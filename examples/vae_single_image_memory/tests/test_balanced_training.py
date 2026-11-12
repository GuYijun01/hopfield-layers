import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from examples.vae_parameter_memory.hopfield import make_retriever, retrieve_parameters
from examples.vae_parameter_memory.vae import VAE
from examples.vae_single_image_memory.balanced_cues import (
    PROFILE, SAMPLING, affine_transforms, augment_image, augmentation_config,
    fit_image_cue, update_indices,
)
from examples.vae_single_image_memory.evaluate import predict_image
from examples.vae_single_image_memory.train_balanced import (
    CompactObjective, learning_rate_schedule, run, set_learning_rate,
)


class BalancedTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)

    def test_geometry_has_sixteen_views_per_family_and_correct_translation_direction(self):
        transforms = affine_transforms()
        self.assertEqual(transforms.shape, (80, 2, 3))
        degrees = torch.atan2(transforms[:16, 1, 0], transforms[:16, 0, 0]).rad2deg()
        self.assertAlmostEqual(float(degrees.min()), -10, places=5)
        self.assertAlmostEqual(float(degrees.max()), 10, places=5)
        for section, axis in ((slice(16, 32), 0), (slice(32, 48), 1), (slice(48, 64), 0)):
            scales = transforms[section, axis, axis].reciprocal()
            self.assertAlmostEqual(float(scales.min()), 0.9, places=6)
            self.assertAlmostEqual(float(scales.max()), 1.1, places=6)
        torch.testing.assert_close(transforms[48:64, 0, 0], transforms[48:64, 1, 1])
        image = torch.zeros(28, 28)
        image[14, 14] = 1
        views = augment_image(image)
        self.assertEqual(views.shape, (80, 784))
        self.assertTrue(((views >= 0) & (views <= 1)).all())
        offsets = torch.cartesian_prod(torch.tensor([-2, -1, 1, 2]), torch.tensor([-2, -1, 1, 2]))
        for view, (dx, dy) in zip(views[64:], offsets):
            row, column = divmod(int(view.argmax()), 28)
            self.assertEqual((row, column), (14 + int(dy), 14 + int(dx)))
            self.assertAlmostEqual(float(view.sum()), 1.0, places=5)

    def test_each_update_gets_a_fresh_permutation(self):
        batches = update_indices(7)
        self.assertEqual(batches.shape, (5, 16))
        generator = torch.Generator().manual_seed(7)
        for batch in batches:
            self.assertEqual(len(batch.unique()), 16)
            torch.testing.assert_close(batch, torch.randperm(80, generator=generator)[:16])
        self.assertLess(len(batches.unique()), 80)  # Across-update repetition is intentional.
        torch.testing.assert_close(batches, update_indices(7))
        self.assertFalse(torch.equal(batches, update_indices(8)))

    def test_five_updates_and_shared_train_test_cue_protocol(self):
        initial = copy.deepcopy(VAE().state_dict())
        image = torch.rand(784)
        original_step = torch.optim.Adam.step
        with patch("torch.optim.Adam.step", autospec=True, side_effect=original_step) as step:
            cue = fit_image_cue(initial, image, 0.001, "cpu", 19)
        self.assertEqual(step.call_count, 5)
        checkpoint = {"experiment": "single_image_5update_v1", "cue_representation": "parameter_delta_v1",
                      "config": {"batch_size": 16, "lr": 0.001}, "initial_vae": initial,
                      "memories": torch.randn(10, 150139), "digits": list(range(10)),
                      "augmentation_profile": PROFILE, "augmentation_config": augmentation_config(),
                      "cue_sampling": SAMPLING}
        _, test_cue, _ = predict_image(image, checkpoint, make_retriever(150139, 32, 8), "cpu", 19)
        torch.testing.assert_close(cue, test_cue, atol=0, rtol=0)

    def test_compact_objective_matches_full_parameter_loss_and_gradients(self):
        original = make_retriever(23, 11, 7).double()
        compact = copy.deepcopy(original)
        memories, initial = torch.randn(4, 23, dtype=torch.float64), torch.randn(23, dtype=torch.float64)
        cues, targets = torch.randn(9, 23, dtype=torch.float64), torch.arange(9) % 4
        retrieved, weights = retrieve_parameters(original, cues, memories, initial)
        association = F.nll_loss(weights.clamp_min(1e-12).log(), targets)
        parameter = F.mse_loss(retrieved, memories[targets]) / memories.var(unbiased=False).clamp_min(1e-8)
        (association + parameter).backward()
        values = CompactObjective(compact, cues[:, compact.feature_indices], targets, memories, initial)()
        values[0].backward()
        torch.testing.assert_close(values[1], association, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(values[2], parameter, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(compact.encoder.weight.grad, original.encoder.weight.grad, atol=1e-10, rtol=1e-10)

    def test_exact_300_300_schedule_preserves_adam_moments(self):
        schedule = learning_rate_schedule(600, 0.001, 300, 0.0003)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=0.001)
        rates = []
        moment_ids = None
        for step in range(1, 601):
            rates.append(set_learning_rate(optimizer, step, schedule))
            if step == 301:
                self.assertEqual(int(optimizer.state[parameter]["step"]), 300)
                self.assertEqual(moment_ids, (id(optimizer.state[parameter]["exp_avg"]),
                                               id(optimizer.state[parameter]["exp_avg_sq"])))
            optimizer.zero_grad()
            parameter.square().sum().backward()
            optimizer.step()
            if step == 300:
                moment_ids = (id(optimizer.state[parameter]["exp_avg"]),
                              id(optimizer.state[parameter]["exp_avg_sq"]))
        self.assertEqual(rates, [0.001] * 300 + [0.0003] * 300)
        self.assertEqual(int(optimizer.state[parameter]["step"]), 600)
        for drop_after, final_lr in ((None, 0.0003), (300, None), (0, 0.0003), (600, 0.0003)):
            with self.assertRaises(ValueError):
                learning_rate_schedule(600, 0.001, drop_after, final_lr)

    def test_cache_resume_and_checkpoint_preserve_experts_without_new_output_directories(self):
        data = TensorDataset(torch.rand(200, 784), torch.arange(10).repeat_interleave(20))
        data.targets = data.tensors[1]
        source_checkpoint = {"experiment": "single_image_5update_v1", "cue_representation": "parameter_delta_v1",
            "config": {"seed": 42, "batch_size": 16, "lr": 0.001, "feature_size": 8,
                       "association_size": 4, "data_dir": "unused", "train_limit": 20,
                       "hopfield_steps": 2, "hopfield_lr": 0.001},
            "parameter_count": 16, "initial_vae": {"parameters": torch.randn(16)},
            "memories": torch.randn(10, 16), "digits": list(range(10)),
            "hopfield": make_retriever(16, 8, 4).state_dict()}

        def fake_features(state, dataset, indices, count, coordinates, lr, device, seed):
            return {"features": torch.randn(count, len(coordinates)), "sources": indices[:count],
                    "seeds": torch.arange(count) + seed}

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            torch.save(source_checkpoint, source)
            source_bytes = source.read_bytes()
            args = Namespace(source=str(source), output_prefix=str(Path(directory)/"balanced"),
                             backup_prefix=str(Path(directory)/"backup"), images_per_digit=3,
                             validation_per_digit=1, hopfield_steps=2, hopfield_lr=None,
                             log_every=1, threads=1, device="cpu")
            with patch("examples.vae_single_image_memory.train_balanced.load_mnist", return_value=data), \
                 patch("examples.vae_single_image_memory.train_balanced.cache_features", side_effect=fake_features):
                report = run(args)
            trained = torch.load(args.output_prefix + "_checkpoint.pt", weights_only=True)
            cache = torch.load(args.output_prefix + "_cue_cache.pt", weights_only=True)
            torch.testing.assert_close(trained["memories"], source_checkpoint["memories"], atol=0, rtol=0)
            self.assertEqual(source_bytes, source.read_bytes())
            self.assertEqual(trained["cue_features"].shape, (30, 8))
            self.assertEqual(trained["cue_sampling"], SAMPLING)
            self.assertEqual(report["train_cues"], 30)
            self.assertEqual(report["validation_cues"], 10)
            for row in cache["digits"]:
                self.assertFalse(torch.isin(row["train"]["sources"], row["validation"]["sources"]).any())
            self.assertFalse(any(path.is_dir() for path in Path(directory).iterdir()))
            with patch("examples.vae_single_image_memory.train_balanced.load_mnist", return_value=data), \
                 patch("examples.vae_single_image_memory.train_balanced.cache_features") as regenerate:
                repeated = run(args)
            regenerate.assert_not_called()
            self.assertEqual(report["stages"][-1]["history"], repeated["stages"][-1]["history"])
            # A new schedule reuses the original cache under a separate file prefix.
            original_cache_bytes = Path(args.output_prefix + "_cue_cache.pt").read_bytes()
            args.cue_cache = args.output_prefix + "_cue_cache.pt"
            args.output_prefix += "_scheduled"
            args.backup_prefix += "_scheduled"
            args.hopfield_steps, args.log_every = 6, 5
            args.lr_drop_after, args.hopfield_lr_final = 3, 0.0003
            with patch("examples.vae_single_image_memory.train_balanced.load_mnist", return_value=data), \
                 patch("examples.vae_single_image_memory.train_balanced.cache_features") as regenerate:
                scheduled = run(args)
            regenerate.assert_not_called()
            self.assertEqual(Path(args.cue_cache).read_bytes(), original_cache_bytes)
            for stage in scheduled["stages"][1:]:
                self.assertEqual([row["step"] for row in stage["history"]], [0, 3, 4, 5, 6])
                self.assertEqual([row["lr"] for row in stage["history"]], [0.001, 0.001, 0.0003, 0.0003, 0.0003])
            trained = torch.load(args.output_prefix + "_checkpoint.pt", weights_only=True)
            self.assertEqual(trained["config"]["lr_drop_after"], 3)
            self.assertEqual(trained["lr_schedule"], scheduled["lr_schedule"])
