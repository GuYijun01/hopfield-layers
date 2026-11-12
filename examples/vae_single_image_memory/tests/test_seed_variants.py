import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from examples.vae_parameter_memory.hopfield import make_retriever
from examples.vae_single_image_memory.seed_variants import (
    SEED_STRIDE, assemble_cue_banks, prepare_seed_variants,
)
from examples.vae_single_image_memory.train_balanced import run


class SeedVariantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)

    def test_roles_preserve_sources_hold_out_fourth_seed_and_resume_without_regeneration(self):
        base = {"signature": "base", "digits": [{"digit": 0,
            "train": {"features": torch.randn(2, 3), "sources": torch.tensor([8, 2]),
                      "seeds": torch.tensor([1042, 1043])},
            "validation": {"features": torch.randn(1, 3), "sources": torch.tensor([9]), "seeds": torch.tensor([999])}}]}
        captured, saved = [], []

        def generate(initial, dataset, sources, seeds, indices, lr, device):
            captured.append((sources.clone(), seeds.clone()))
            return seeds[:, None].float().expand(-1, 3).clone()

        def interrupt(value):
            saved.append(copy.deepcopy(value))
            raise RuntimeError("simulated interruption after a completed role")

        with patch("examples.vae_single_image_memory.seed_variants.features_for_sources", side_effect=generate):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                prepare_seed_variants(base, None, {}, None, torch.arange(3), 0.001, "cpu", 3, interrupt)
            expanded = prepare_seed_variants(base, saved[-1], {}, None, torch.arange(3), 0.001, "cpu", 3,
                                              lambda value: saved.append(copy.deepcopy(value)))
        self.assertEqual(len(captured), 3)  # Original reused; completed extra role was not repeated.
        for role, (sources, seeds) in enumerate(captured, 1):
            torch.testing.assert_close(sources, base["digits"][0]["train"]["sources"])
            torch.testing.assert_close(seeds, base["digits"][0]["train"]["seeds"] + role * SEED_STRIDE)
        banks = assemble_cue_banks(base, expanded)
        self.assertEqual(banks["train"]["features"].shape, (6, 3))
        self.assertEqual(banks["train"]["sources"].tolist(), [8, 2, 8, 2, 8, 2])
        self.assertEqual(banks["heldout_seed"]["sources"].tolist(), [8, 2])
        self.assertFalse(torch.isin(banks["heldout_seed"]["seeds"], banks["train"]["seeds"]).any())
        torch.testing.assert_close(banks["validation"]["features"], base["digits"][0]["validation"]["features"])
        torch.testing.assert_close(banks["train"]["features"][:2], base["digits"][0]["train"]["features"])
        with patch("examples.vae_single_image_memory.seed_variants.features_for_sources") as regenerate:
            prepare_seed_variants(base, expanded, {}, None, torch.arange(3), 0.001, "cpu", 3, lambda value: None)
        regenerate.assert_not_called()
        with self.assertRaises(ValueError):
            prepare_seed_variants(base, expanded, {}, None, torch.arange(3), 0.001, "cpu", 4, lambda value: None)

    def test_multi_seed_inherits_training_setup_and_heldout_features_never_change_weights(self):
        data = TensorDataset(torch.rand(200, 784), torch.arange(10).repeat_interleave(20))
        data.targets = data.tensors[1]
        model = make_retriever(16, 8, 4, projection_mode="separate")
        source_checkpoint = {"experiment": "single_image_5update_v1", "cue_representation": "parameter_delta_v1",
            "projection_mode": "separate",
            "config": {"seed": 42, "batch_size": 16, "lr": 0.001, "feature_size": 8,
                       "association_size": 4, "data_dir": "unused", "train_limit": 20,
                       "hopfield_steps": 6, "hopfield_lr": 0.0005, "lr_drop_after": 3,
                       "hopfield_lr_final": 0.0003, "projection_mode": "separate"},
            "parameter_count": 16, "initial_vae": {"parameters": torch.randn(16)},
            "memories": torch.randn(10, 16), "digits": list(range(10)), "hopfield": model.state_dict()}

        def base_features(state, dataset, indices, count, coordinates, lr, device, seed):
            return {"features": torch.randn(count, len(coordinates)), "sources": indices[:count],
                    "seeds": torch.arange(count) + seed}

        def extra_features(initial, dataset, sources, seeds, coordinates, lr, device):
            generator = torch.Generator().manual_seed(int(seeds[0]))
            return torch.randn(len(sources), len(coordinates), generator=generator)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            torch.save(source_checkpoint, source)
            original_source = source.read_bytes()
            prefix = str(Path(directory) / "multi")
            args = Namespace(source=str(source), output_prefix=prefix, backup_prefix=None,
                             images_per_digit=3, validation_per_digit=1, hopfield_steps=None,
                             hopfield_lr=None, projection_mode=None, seed_variants=3,
                             lr_drop_after=None, hopfield_lr_final=None, log_every=5, threads=1, device="cpu")
            with patch("examples.vae_single_image_memory.train_balanced.load_mnist", return_value=data), \
                 patch("examples.vae_single_image_memory.train_balanced.cache_features", side_effect=base_features), \
                 patch("examples.vae_single_image_memory.seed_variants.features_for_sources", side_effect=extra_features):
                report = run(args)
            trained = torch.load(prefix + "_checkpoint.pt", weights_only=True)
            base = torch.load(prefix + "_cue_cache.pt", weights_only=True)
            seed_cache = torch.load(prefix + "_seed_cache.pt", weights_only=True)
            self.assertEqual(report["training_images"], 30)
            self.assertEqual(report["train_cues"], 90)
            self.assertEqual(report["heldout_seed_cues"], 30)
            self.assertEqual(report["validation_cues"], 10)
            self.assertEqual(report["projection_mode"], "separate")
            self.assertEqual(report["lr_schedule"], [{"start_step": 1, "end_step": 3, "lr": 0.0005},
                                                    {"start_step": 4, "end_step": 6, "lr": 0.0003}])
            self.assertEqual(trained["cue_features"].shape, (90, 8))
            self.assertEqual(trained["cue_source_indices"].unique().numel(), 30)
            self.assertEqual(trained["heldout_seed_offset"], 3 * SEED_STRIDE)
            self.assertEqual(source.read_bytes(), original_source)
            for stage in report["stages"]:
                self.assertTrue(all("heldout_seed" in row for row in stage["history"]))
            assembled = assemble_cue_banks(base, seed_cache)
            self.assertFalse(torch.isin(assembled["validation"]["sources"], assembled["train"]["sources"]).any())
            # Deliberately change only held-out feature values. Training must be identical.
            for row in seed_cache["digits"]:
                row["roles"]["heldout_seed"]["features"].mul_(-1000)
            torch.save(seed_cache, prefix + "_seed_cache.pt")
            with patch("examples.vae_single_image_memory.train_balanced.load_mnist", return_value=data), \
                 patch("examples.vae_single_image_memory.train_balanced.cache_features") as regenerate_base, \
                 patch("examples.vae_single_image_memory.seed_variants.features_for_sources") as regenerate_extra:
                repeated = run(args)
            regenerate_base.assert_not_called()
            regenerate_extra.assert_not_called()
            repeated_weights = torch.load(prefix + "_checkpoint.pt", weights_only=True)["hopfield"]
            for name, weights in trained["hopfield"].items():
                torch.testing.assert_close(weights, repeated_weights[name], atol=0, rtol=0)
            self.assertEqual(report["stages"][-1]["history"][-1]["train"],
                             repeated["stages"][-1]["history"][-1]["train"])
