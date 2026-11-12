import copy
import unittest

import torch
from torch.nn import functional as F

from hflayers.block_values import InvertibleBlockDiagonal
from examples.vae_parameter_memory.hopfield import make_retriever, retriever_from_checkpoint, retrieve_parameters
from examples.vae_single_image_memory.train_balanced import CompactObjective


class BlockValueTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        torch.set_num_threads(1)

    def test_roundtrip_partial_blocks_and_condition_bound_after_parameter_changes(self):
        for dimension, block in ((23, 4), (7, 16), (1, 1)):
            transform = InvertibleBlockDiagonal(dimension, block, seed=8)
            values = torch.randn(2, 3, dimension, dtype=torch.float64)
            torch.testing.assert_close(transform.decode(transform.encode(values)), values, atol=1e-12, rtol=1e-12)
            if dimension > 1:
                self.assertFalse(torch.equal(transform.encode(values), values))
            with torch.no_grad():
                transform.raw_lower.fill_(10000)
            torch.testing.assert_close(transform.round_trip(values), values, atol=1e-12, rtol=1e-12)
            self.assertLessEqual(float(torch.linalg.cond(transform.blocks()).max().detach()), 5 / 3 + 1e-12)

    def test_soft_batched_retrieval_preserves_the_weighted_mixture_and_dtype(self):
        transform = InvertibleBlockDiagonal(23, 4)
        values = torch.randn(2, 4, 23)
        weights = torch.randn(2, 3, 4).softmax(-1)
        actual = transform.retrieve(weights, values)
        self.assertEqual(actual.dtype, values.dtype)
        torch.testing.assert_close(actual, weights @ values, atol=1e-6, rtol=1e-6)

    def test_full_and_compact_losses_preserve_query_key_gradients_and_cancel_value_gradient(self):
        torch.manual_seed(83)
        baseline = make_retriever(23, 11, 7, projection_mode="separate").double()
        torch.manual_seed(83)
        projected = make_retriever(23, 11, 7, projection_mode="separate", value_projection="block_diagonal",
                                   value_block_size=4, value_seed=17).double()
        cues, memories = torch.randn(9, 23, dtype=torch.float64), torch.randn(4, 23, dtype=torch.float64)
        initial, targets = torch.randn(23, dtype=torch.float64), torch.arange(9) % 4
        original, original_weights = retrieve_parameters(baseline, cues, memories, initial)
        output, weights = retrieve_parameters(projected, cues, memories, initial)
        torch.testing.assert_close(weights, original_weights, atol=0, rtol=0)
        torch.testing.assert_close(output, original, atol=1e-12, rtol=1e-12)

        def loss(retrieved, associations):
            return (F.nll_loss(associations.clamp_min(1e-12).log(), targets)
                    + F.mse_loss(retrieved, memories[targets]) / memories.var(unbiased=False))

        loss(original, original_weights).backward()
        full = loss(output, weights)
        full.backward()
        for name in ("query_encoder.weight", "key_encoder.weight"):
            torch.testing.assert_close(dict(projected.named_parameters())[name].grad,
                                       dict(baseline.named_parameters())[name].grad, atol=1e-10, rtol=1e-10)
        self.assertLess(float(projected.value_projection.raw_lower.grad.norm()), 1e-10)
        full_gradients = {name: parameter.grad.clone() for name, parameter in projected.named_parameters()}
        projected.zero_grad()
        compact = CompactObjective(projected, cues[:, projected.feature_indices], targets, memories, initial)()[0]
        compact.backward()
        torch.testing.assert_close(compact, full, atol=1e-12, rtol=1e-12)
        for name, parameter in projected.named_parameters():
            torch.testing.assert_close(parameter.grad, full_gradients[name], atol=1e-10, rtol=1e-10)

    def test_checkpoint_reload_and_old_identity_checkpoints(self):
        model = make_retriever(23, 11, 7, projection_mode="separate", value_projection="block_diagonal",
                               value_block_size=4, value_seed=17)
        checkpoint = {"parameter_count": 23, "config": {"feature_size": 11, "association_size": 7},
                      "projection_mode": "separate", "value_projection": "block_diagonal",
                      "value_block_size": 4, "value_seed": 17, "hopfield": copy.deepcopy(model.state_dict())}
        loaded = retriever_from_checkpoint(checkpoint, "cpu")
        cues, memories = torch.randn(3, 23), torch.randn(4, 23)
        torch.testing.assert_close(loaded(cues, memories), model(cues, memories), atol=0, rtol=0)
        old = make_retriever(23, 11, 7)
        checkpoint.pop("projection_mode")
        for field in ("value_projection", "value_block_size", "value_seed"):
            checkpoint.pop(field)
        checkpoint["hopfield"] = old.state_dict()
        restored_old = retriever_from_checkpoint(checkpoint, "cpu")
        self.assertIsNone(restored_old.value_projection)
        torch.testing.assert_close(restored_old(cues, memories), old(cues, memories), atol=0, rtol=0)
