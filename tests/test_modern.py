import copy
import unittest

import torch
from torch.nn import functional as F
from torch.nn.utils import vector_to_parameters

from hflayers.modern import ModernHopfield
from examples.split_mnist.experiment import MLP, adapt, train_expert, vector


class ModernHopfieldTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_single_update_matches_attention_and_batches(self):
        layer = ModernHopfield(3, 3, 3, beta=2.0)
        with torch.no_grad():
            layer.encoder.weight.copy_(torch.eye(3))
        memories = torch.tensor([[1., 0., -1.], [-1., 1., 0.], [0., -1., 1.]])
        cues = torch.randn(2, 4, 3)
        center = memories.mean(0)
        scale = memories.std(0, unbiased=False).clamp_min(1e-3)
        q = F.normalize((cues - center) / scale, dim=-1)
        k = F.normalize((memories - center) / scale, dim=-1)
        expected = (2 * q @ k.T).softmax(-1)
        output, weights = layer(cues, memories, True)
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(output, expected @ memories)
        batched = memories.unsqueeze(0).repeat(2, 1, 1)
        torch.testing.assert_close(layer(cues, batched), output)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4))

    def test_training_and_serialization(self):
        layer = ModernHopfield(12, 8, 12)
        memories = torch.randn(4, 12)
        cues = memories + 0.5 * torch.randn_like(memories)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.02)
        _, initial_weights = layer(cues, memories, True)
        initial_loss = F.nll_loss(initial_weights.clamp_min(1e-12).log(), torch.arange(4)).item()
        for _ in range(60):
            optimizer.zero_grad()
            _, weights = layer(cues, memories, True)
            loss = F.nll_loss(weights.clamp_min(1e-12).log(), torch.arange(4))
            loss.backward()
            optimizer.step()
        self.assertTrue(torch.equal(weights.argmax(-1), torch.arange(4)))
        self.assertLess(loss.item(), initial_loss)
        restored = ModernHopfield(12, 8, 12)
        restored.load_state_dict(layer.state_dict())
        torch.testing.assert_close(restored(cues, memories), layer(cues, memories))

    def test_one_memory_and_invalid_inputs(self):
        layer = ModernHopfield(8)
        memory = torch.randn(1, 8)
        torch.testing.assert_close(layer(torch.randn(3, 8), memory), memory.expand(3, 8))
        with self.assertRaises(ValueError):
            ModernHopfield(8, update_steps_max=1)
        with self.assertRaises(ValueError):
            layer(torch.randn(3, 7), memory)
        with self.assertRaises(ValueError):
            layer(torch.randn(3, 8), torch.empty(0, 8))

    def test_parameter_roundtrip_and_paired_adaptation(self):
        from types import SimpleNamespace
        model = MLP()
        self.assertEqual(vector(model).numel(), 478410)
        initial = vector(model)
        cloned = copy.deepcopy(model)
        vector_to_parameters(initial.clone(), cloned.parameters())
        inputs = torch.randn(4, 1, 28, 28)
        torch.testing.assert_close(model(inputs), cloned(inputs))
        batches = [(inputs, torch.tensor([0, 1, 0, 1]))]
        args = SimpleNamespace(lr=1e-3)
        cues = adapt(model, batches, 4, args, "cpu")
        prefix = adapt(model, batches, 2, args, "cpu")
        torch.testing.assert_close(cues["N/2"], prefix["N"])
        self.assertFalse(torch.equal(cues["N"], cues["N/2"]))
        torch.testing.assert_close(vector(model), initial)

    def test_restores_best_validation_checkpoint(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        model = MLP()
        batches = [(torch.randn(4, 1, 28, 28), torch.tensor([0, 1, 0, 1]))]
        args = SimpleNamespace(lr=1e-3, epochs=3)
        # The first epoch is deliberately best; final weights must be rolled back.
        with patch("examples.split_mnist.experiment.evaluate", side_effect=[
                {"loss": 1.0}, {"loss": 2.0}, {"loss": 3.0}]):
            memory, first_epoch_cue, info = train_expert(model, batches, batches, args, "cpu")
        self.assertEqual(info["best_epoch"], 1)
        torch.testing.assert_close(memory, first_epoch_cue)


if __name__ == "__main__":
    unittest.main()
