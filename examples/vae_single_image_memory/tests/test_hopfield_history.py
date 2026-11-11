import copy
import unittest

import torch

from examples.vae_parameter_memory.hopfield import (
    make_retriever, retriever_metrics, train_retriever,
)


class HopfieldHistoryTests(unittest.TestCase):
    def test_validation_logging_never_changes_optimizer_trajectory(self):
        torch.manual_seed(17)
        memories = torch.randn(3, 16)
        targets = torch.arange(3).repeat_interleave(2)
        cues = memories[targets] + 0.2 * torch.randn(6, 16)
        validation = torch.randn(4, 16, requires_grad=True)
        val_targets = torch.tensor([0, 1, 2, 0])
        original = make_retriever(16, 16, 8)
        logged = copy.deepcopy(original)
        summary = train_retriever(original, cues, targets, memories, 5, 0.01, torch.zeros(16))
        diagnostic = train_retriever(
            logged, cues, targets, memories, 5, 0.01, torch.zeros(16),
            validation_cues=validation, validation_targets=val_targets, history_every=2)
        self.assertEqual([row["step"] for row in diagnostic["history"]], [0, 2, 4, 5])
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, logged.state_dict()[name], atol=0, rtol=0)
        self.assertIsNone(validation.grad)
        self.assertEqual(summary["loss"], diagnostic["loss"])
        metrics = retriever_metrics(logged, cues, targets, memories, torch.zeros(16))
        self.assertEqual(diagnostic["history"][-1]["train"], metrics)
        self.assertAlmostEqual(metrics["total_loss"], metrics["association_loss"] +
                               metrics["normalized_parameter_mse"], places=6)

    def test_one_memory_is_trivially_perfect(self):
        model = make_retriever(8, 8, 4)
        summary = train_retriever(
            model, torch.randn(2, 8), torch.zeros(2, dtype=torch.long), torch.randn(1, 8),
            3, 0.01, torch.zeros(8), validation_cues=torch.randn(1, 8),
            validation_targets=torch.zeros(1, dtype=torch.long), history_every=1)
        for row in summary["history"]:
            for split in ("train", "validation"):
                self.assertEqual(row[split]["cue_accuracy"], 1.0)
                self.assertEqual(row[split]["association_loss"], 0.0)
