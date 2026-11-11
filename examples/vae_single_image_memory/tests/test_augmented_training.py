import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from examples.vae_parameter_memory.data import digit_split
from examples.vae_parameter_memory.hopfield import make_retriever
from examples.vae_parameter_memory.vae import VAE
from examples.vae_single_image_memory.augmented_cues import generate_augmented_cues
from examples.vae_single_image_memory.evaluate import predict_image
from examples.vae_single_image_memory.train_augmented import run


class AugmentedTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_training_cue_matches_test_cue_with_same_image_and_seed(self):
        initial = copy.deepcopy(VAE().state_dict())
        image = torch.rand(784)
        data = TensorDataset(image.unsqueeze(0), torch.tensor([4]))
        checkpoint = {"experiment": "single_image_5update_v1", "cue_representation": "parameter_delta_v1",
                      "config": {"batch_size": 16, "lr": 0.001}, "initial_vae": initial,
                      "memories": torch.randn(10, 150139), "digits": list(range(10))}
        hopfield = make_retriever(150139, 32, 8)
        for profile in ("legacy", "mild"):
            with self.subTest(profile=profile):
                cues, indices = generate_augmented_cues(
                    initial, data, torch.tensor([0]), 1, 0.001, "cpu", 9,
                    augmentation_profile=profile)
                checkpoint["augmentation_profile"] = profile
                _, test_cue, _ = predict_image(image, checkpoint, hopfield, "cpu", 9)
                self.assertEqual(indices.tolist(), [0])
                torch.testing.assert_close(cues[0], test_cue, atol=0, rtol=0)

    def test_sampling_is_reproducible_distinct_and_label_free(self):
        initial = copy.deepcopy(VAE().state_dict())
        images = torch.rand(6, 784)
        data = TensorDataset(images, torch.zeros(6))
        indices = torch.tensor([1, 3, 5])
        cues, selected = generate_augmented_cues(initial, data, indices, 2, 0.001, "cpu", 13)
        other, selected_again = generate_augmented_cues(
            initial, TensorDataset(images, torch.ones(6)), indices, 2, 0.001, "cpu", 13)
        torch.testing.assert_close(cues, other, atol=0, rtol=0)
        torch.testing.assert_close(selected, selected_again)
        self.assertEqual(selected.unique().numel(), 2)
        self.assertTrue(torch.isin(selected, indices).all())
        # Running only the second source with its original seed reproduces its cue:
        # no weights or optimizer state can have carried over from source one.
        replay, _ = generate_augmented_cues(initial, data, selected[1:], 1, 0.001, "cpu", 14)
        torch.testing.assert_close(cues[1], replay[0], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            generate_augmented_cues(initial, data, indices, 4, 0.001, "cpu", 13)

    def test_refit_preserves_experts_and_splits_and_matches_baseline_counts(self):
        # Small synthetic parameter vectors exercise checkpoint/replay orchestration;
        # the real VAE/augmentation equivalence is tested separately above.
        data = TensorDataset(torch.rand(200, 784), torch.arange(10).repeat_interleave(20))
        data.targets = data.tensors[1]
        bank = torch.randn(10, 16)
        initial = {"parameters": torch.randn(16)}
        baseline = {"experiment": "single_image_5update_v1", "cue_representation": "parameter_delta_v1",
                    "config": {"seed": 42, "batch_size": 16, "lr": 0.001, "feature_size": 8,
                               "association_size": 4, "data_dir": "unused", "train_limit": 20,
                               "hopfield_steps": 2, "hopfield_lr": 0.001},
                    "parameter_count": 16, "initial_vae": initial, "memories": bank,
                    "digits": list(range(10)), "targets": torch.arange(10).repeat_interleave(2),
                    "cues": torch.randn(20, 16), "hopfield": make_retriever(16, 8, 4).state_dict(),
                    "block_indices": [digit_split(data, d, 42 + d, 20)[0][:1] for d in range(10)]}

        def fake_cues(state, dataset, indices, count, lr, device, seed, *, augmentation_profile):
            self.assertEqual(augmentation_profile, "mild")
            sources = indices[:count]
            return torch.randn(count, 16), sources

        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "checkpoint.pt", Path(directory) / "augmented"
            torch.save(baseline, source)
            original_bytes = source.read_bytes()
            args = Namespace(source=str(source), output_dir=str(output), device="cpu", threads=1,
                             cues_per_digit=0, validation_cues_per_digit=0, log_every=1, augmentation="mild")
            with patch("examples.vae_single_image_memory.train_augmented.load_mnist", return_value=data), \
                 patch("examples.vae_single_image_memory.train_augmented.generate_augmented_cues", side_effect=fake_cues):
                report = run(args)
            trained = torch.load(output / "checkpoint.pt", weights_only=True)
            validation = torch.load(output / "validation_cues.pt", weights_only=True)
            self.assertEqual(source.read_bytes(), original_bytes)
            torch.testing.assert_close(trained["memories"], bank, atol=0, rtol=0)
            torch.testing.assert_close(trained["initial_vae"]["parameters"], initial["parameters"], atol=0, rtol=0)
            self.assertEqual(trained["targets"].bincount().tolist(), baseline["targets"].bincount().tolist())
            self.assertEqual(trained["hopfield_cue_protocol"], "single_image_augmented_v1")
            self.assertEqual(trained["augmentation_profile"], "mild")
            self.assertEqual(trained["augmentation_config"], validation["augmentation_config"])
            self.assertEqual(report["augmentation_profile"], "mild")
            self.assertNotIn("block_indices", trained)
            self.assertFalse(torch.isin(trained["cue_source_indices"], validation["source_indices"]).any())
            self.assertEqual(len(report["stages"]), 10)
            self.assertEqual([r["step"] for r in report["stages"][-1]["history"]], [0, 1, 2])
