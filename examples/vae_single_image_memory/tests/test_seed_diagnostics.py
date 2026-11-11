import copy
import unittest

import torch
from torch.utils.data import TensorDataset

from examples.vae_parameter_memory.hopfield import make_retriever, retriever_metrics
from examples.vae_parameter_memory.vae import VAE
from examples.vae_single_image_memory.diagnose_seeds import (
    original_cue_seeds, regenerate_cues, score_cues,
)
from examples.vae_single_image_memory.evaluate import predict_image


class SeedDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def test_original_seed_reconstruction_and_explicit_metadata(self):
        checkpoint = {"config": {"seed": 42}, "targets": torch.tensor([0, 0, 1, 1, 2])}
        expected = torch.tensor([1042, 1043, 11042, 11043, 21042])
        torch.testing.assert_close(original_cue_seeds(checkpoint), expected)
        checkpoint["cue_seeds"] = expected + 10
        torch.testing.assert_close(original_cue_seeds(checkpoint), expected + 10)
        checkpoint["cue_seeds"] = torch.tensor([0])
        with self.assertRaises(ValueError):
            original_cue_seeds(checkpoint)

    def test_replay_keeps_source_order_and_reproduces_inference_cues(self):
        initial = copy.deepcopy(VAE().state_dict())
        data = TensorDataset(torch.rand(2, 784), torch.tensor([1, 0]))
        indices, seeds = torch.tensor([1, 0]), torch.tensor([1042, 11042])
        replay = regenerate_cues(initial, data, indices, seeds, 0.001, "cpu", "mild")
        checkpoint = {"experiment": "single_image_5update_v1", "cue_representation": "parameter_delta_v1",
                      "config": {"batch_size": 16, "lr": 0.001}, "initial_vae": initial,
                      "memories": torch.randn(10, 150139), "digits": list(range(10)),
                      "augmentation_profile": "mild"}
        model = make_retriever(150139, 32, 8)
        for position, index in enumerate(indices.tolist()):
            _, cue, _ = predict_image(data[index][0], checkpoint, model, "cpu", int(seeds[position]))
            torch.testing.assert_close(cue, replay[position], atol=0, rtol=0)

    def test_seed_controls_and_fresh_episode_state(self):
        initial = copy.deepcopy(VAE().state_dict())
        data = TensorDataset(torch.rand(2, 784), torch.zeros(2))
        indices, seeds = torch.tensor([1]), torch.tensor([19])
        original = regenerate_cues(initial, data, indices, seeds, 0.001, "cpu", "mild")
        for vary in ("both", "shuffle", "latent"):
            changed = regenerate_cues(initial, data, indices, seeds, 0.001, "cpu", "mild", offset=1000, vary=vary)
            self.assertFalse(torch.equal(original, changed))
            torch.testing.assert_close(changed, regenerate_cues(
                initial, data, indices, seeds, 0.001, "cpu", "mild", offset=1000, vary=vary), atol=0, rtol=0)
        legacy = regenerate_cues(initial, data, indices, seeds, 0.001, "cpu", "legacy")
        legacy_shuffle = regenerate_cues(initial, data, indices, seeds, 0.001, "cpu", "legacy", offset=1000, vary="shuffle")
        torch.testing.assert_close(legacy, legacy_shuffle, atol=0, rtol=0)

    def test_chunked_losses_match_training_objective_without_updating_weights(self):
        model = make_retriever(16, 16, 8)
        before = copy.deepcopy(model.state_dict())
        memories, initial, cues = torch.randn(10, 16), torch.randn(16), torch.randn(7, 16)
        targets, indices = torch.arange(7), torch.arange(7) + 100
        reference = retriever_metrics(model, cues, targets, memories, initial)
        baseline = score_cues(model, cues, targets, memories, initial, indices, batch_size=3)
        for key, value in reference.items():
            self.assertAlmostEqual(baseline["metrics"][key], value, places=5)
        replay = score_cues(model, cues, targets, memories, initial, indices, batch_size=2,
                            reference_cues=cues, reference_predictions=[r["predicted_digit"] for r in baseline["per_image"]])
        self.assertEqual(replay["metrics"]["relative_cue_l2_change"], 0.0)
        self.assertEqual(replay["metrics"]["prediction_agreement_with_cached"], 1.0)
        self.assertEqual([r["source_index"] for r in replay["per_image"]], indices.tolist())
        for name, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
