import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from examples.vae_single_image_memory.augment import augment_image
from examples.vae_single_image_memory.evaluate import validate_checkpoint


class MildAugmentationTests(unittest.TestCase):
    def test_geometry_and_seeded_shuffle_of_the_same_eighty_views(self):
        image = torch.zeros(28, 28)
        image[7:19, 10:16] = 1
        with patch("torch.nn.functional.affine_grid", wraps=F.affine_grid) as grid:
            first = augment_image(image, profile="mild", seed=7)
        transforms = grid.call_args.args[0]
        angles = torch.atan2(transforms[:40, 1, 0], transforms[:40, 0, 0])
        self.assertAlmostEqual(angles.min().item(), -math.pi / 12, places=6)
        self.assertAlmostEqual(angles.max().item(), math.pi / 12, places=6)
        self.assertAlmostEqual(transforms[40, 0, 0].item(), 1 / 0.9, places=6)
        self.assertAlmostEqual(transforms[59, 0, 0].item(), 1 / 1.1, places=6)
        self.assertAlmostEqual(transforms[60, 1, 1].item(), 1 / 0.9, places=6)
        self.assertAlmostEqual(transforms[79, 1, 1].item(), 1 / 1.1, places=6)
        self.assertEqual(first.shape, (80, 784))
        self.assertTrue(((first >= 0) & (first <= 1)).all())
        torch.testing.assert_close(first, augment_image(image, profile="mild", seed=7), atol=0, rtol=0)
        second = augment_image(image, profile="mild", seed=8)
        self.assertFalse(torch.equal(first, second))
        orders = [torch.randperm(80, generator=torch.Generator().manual_seed(seed)) for seed in (7, 8)]
        torch.testing.assert_close(first[orders[0].argsort()], second[orders[1].argsort()], atol=0, rtol=0)
        self.assertEqual(len(first.split(16)), 5)

    def test_old_checkpoints_keep_legacy_and_inconsistent_settings_are_rejected(self):
        old = {"experiment": "single_image_5update_v1", "cue_representation": "parameter_delta_v1",
               "config": {"batch_size": 16}}
        validate_checkpoint(old)
        image = torch.rand(784)
        torch.testing.assert_close(augment_image(image), augment_image(image, profile="legacy", seed=100),
                                   atol=0, rtol=0)
        validate_checkpoint({**old, "augmentation_profile": "mild"})
        with self.assertRaises(ValueError):
            validate_checkpoint({**old, "augmentation_profile": "mild", "augmentation_config": {}})
        with self.assertRaises(ValueError):
            validate_checkpoint({**old, "augmentation_profile": "unknown"})
