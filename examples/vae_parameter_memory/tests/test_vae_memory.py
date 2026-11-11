import copy
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from examples.vae_parameter_memory.cues import fit_block, generate_block_cues
from examples.vae_parameter_memory.evaluate import predict_block
from examples.vae_parameter_memory.hopfield import make_retriever, retrieve_parameters, train_retriever
from examples.vae_parameter_memory.training import train_vae0
from examples.vae_parameter_memory.vae import VAE, initial_vector, parameter_vector, restore_vector, vae_loss


class VAEMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.initial = copy.deepcopy(VAE().state_dict())

    def test_architecture_elbo_and_parameter_roundtrip(self):
        model = VAE()
        self.assertEqual(parameter_vector(model).numel(), 150139)
        x = torch.rand(3, 784)
        logits, mu, logvar = model(x)
        self.assertEqual(mu.shape, (3, 5))
        self.assertEqual(logits.shape, (3, 784))
        losses = vae_loss(logits, x, mu, logvar)
        self.assertEqual(losses.shape, (3,))
        self.assertTrue(torch.isfinite(losses).all())
        losses.mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        clone = VAE()
        restore_vector(clone, parameter_vector(model))
        torch.testing.assert_close(clone(x, sample=False)[0], model(x, sample=False)[0])

    def test_blocks_are_disjoint_reset_and_exactly_50_updates(self):
        data = TensorDataset(torch.rand(220, 784), torch.zeros(220, dtype=torch.long))
        indices = torch.arange(200)  # The last 20 examples stand in for held-out data.
        original = copy.deepcopy(self.initial)
        cues, blocks = generate_block_cues(self.initial, data, indices, 2, 1e-3, "cpu", 9)
        self.assertEqual(cues.shape, (2, 150139))
        self.assertEqual(blocks.shape, (2, 100))
        self.assertEqual(blocks.flatten().unique().numel(), 200)
        self.assertLess(blocks.max().item(), 200)
        # Replaying a block alone with its original seed must reproduce the cue:
        # neither the preceding block's weights nor its Adam state can carry over.
        second = blocks[1]
        batches = (data.tensors[0][part] for part in second.split(2))
        replay = fit_block(self.initial, batches, 1e-3, "cpu", 10)
        torch.testing.assert_close(replay, cues[1])
        for name in original:
            torch.testing.assert_close(self.initial[name], original[name])
        with self.assertRaises(ValueError):
            generate_block_cues(self.initial, data, torch.arange(99), 2, 1e-3, "cpu", 9)
        # Training labels do not enter the VAE2 loss.
        data_with_other_labels = TensorDataset(data.tensors[0], torch.ones(220, dtype=torch.long))
        other, _ = generate_block_cues(self.initial, data_with_other_labels, indices, 2, 1e-3, "cpu", 9)
        torch.testing.assert_close(other, cues)

    def test_vae0_first50_is_not_first_epoch_and_best_is_restored(self):
        batches = [(torch.rand(1, 784), torch.zeros(1, dtype=torch.long)) for _ in range(60)]
        snapshots = []
        def validation(model, *args):
            snapshots.append(parameter_vector(model))
            return float(len(snapshots))  # Epoch one deliberately best.
        with patch("examples.vae_parameter_memory.training.validation_loss", side_effect=validation):
            memory, cue, info = train_vae0(self.initial, batches, batches, 2, 1e-3, "cpu", seed=2)
        expected_cue = fit_block(self.initial, (x for x, _ in batches[:50]), 1e-3, "cpu", 2)
        torch.testing.assert_close(cue, expected_cue)
        torch.testing.assert_close(memory, snapshots[0])
        self.assertFalse(torch.equal(memory, cue + initial_vector(self.initial)))
        self.assertEqual(info["best_epoch"], 1)

    def test_hopfield_learns_multiple_cues_for_each_memory(self):
        memories = torch.randn(3, 16)
        targets = torch.tensor([0, 0, 1, 1, 2, 2])
        cues = memories[targets] + 0.3 * torch.randn(6, 16)
        hopfield = make_retriever(16, 16, 8)
        summary = train_retriever(hopfield, cues, targets, memories, 60, 0.01, torch.zeros(16))
        self.assertEqual(summary["cue_accuracy"], 1.0)
        self.assertEqual(hopfield.update_steps_max, 0)

    def test_block_prediction_uses_deltas_but_restores_full_parameters(self):
        memories = torch.randn(10, 150139) * 0.01
        initial = initial_vector(self.initial)
        checkpoint = {"config": {"batch_size": 2, "lr": 1e-3},
                      "initial_vae": self.initial, "memories": memories, "digits": list(range(10)),
                      "cue_representation": "parameter_delta_v1"}
        hopfield = make_retriever(150139, 128, 16)
        images = [torch.rand(2, 784) for _ in range(50)]
        delta = memories[7] - initial
        with patch("examples.vae_parameter_memory.evaluate.fit_block", return_value=delta) as cue:
            result, returned_cue, retrieved = predict_block(images, checkpoint, hopfield, "cpu", 3)
        self.assertEqual(result["predicted_digit"], 7)
        torch.testing.assert_close(returned_cue, delta)
        torch.testing.assert_close(retrieved, torch.tensor(result["associations"]) @ memories,
                                   atol=1e-7, rtol=1e-5)
        self.assertEqual(len(cue.call_args.args), 5)  # Initial state, images, settings; no label.
        with self.assertRaises(ValueError):
            predict_block(images, {**checkpoint, "cue_representation": "raw"}, hopfield, "cpu", 3)

    def test_delta_origin_and_translation_invariance(self):
        model = VAE()
        model.load_state_dict(self.initial)
        torch.testing.assert_close(initial_vector(self.initial), parameter_vector(model))
        memories, cues, initial = torch.randn(3, 16), torch.randn(4, 16), torch.randn(16)
        hopfield = make_retriever(16, 16, 8)
        expected, expected_weights = hopfield(cues, memories, True)
        restored, weights = retrieve_parameters(hopfield, cues - initial, memories, initial)
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(restored, expected)


if __name__ == "__main__":
    unittest.main()
