import copy
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import TensorDataset
from examples.vae_single_image_memory.augment import augment_image
from examples.vae_single_image_memory.cues import fit_block, generate_block_cues
from examples.vae_single_image_memory.evaluate import predict_image
from examples.vae_parameter_memory.vae import VAE, initial_vector
from examples.vae_parameter_memory.hopfield import make_retriever


class SingleImageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.initial = copy.deepcopy(VAE().state_dict())

    def test_augmentations_are_deterministic_and_axis_specific(self):
        image = torch.zeros(28, 28)
        image[10:18, 12:16] = 1
        augmented = augment_image(image)
        self.assertEqual(augmented.shape, (80, 784))
        self.assertTrue(((augmented >= 0) & (augmented <= 1)).all())
        torch.testing.assert_close(augmented, augment_image(image))
        self.assertEqual(augmented.unique(dim=0).shape[0], 80)
        yy, xx = torch.meshgrid(torch.arange(28.), torch.arange(28.), indexing="ij")
        def moments(x):
            x = x.reshape(28, 28)
            return ((x * (xx - 13.5).square()).sum() / x.sum(),
                    (x * (yy - 13.5).square()).sum() / x.sum())
        original = moments(image)
        horizontal, vertical = moments(augmented[59]), moments(augmented[79])
        self.assertGreater(horizontal[0], original[0])
        self.assertGreater(vertical[1], original[1])
        torch.testing.assert_close(horizontal[1], original[1])
        torch.testing.assert_close(vertical[0], original[0])

    def test_five_updates_disjoint_blocks_and_reset(self):
        images = torch.rand(170, 784)
        data = TensorDataset(images, torch.zeros(170))
        with patch("torch.optim.Adam.step", autospec=True,
                   side_effect=torch.optim.Adam.step) as step:
            cues, blocks = generate_block_cues(self.initial, data, torch.arange(160),
                                               16, 1e-3, "cpu", 11)
        self.assertEqual(step.call_count, 10)
        self.assertEqual(blocks.shape, (2, 80))
        self.assertEqual(blocks.unique().numel(), 160)
        replay = fit_block(self.initial, images[blocks[1]].split(16), 1e-3, "cpu", 12)
        torch.testing.assert_close(replay, cues[1])
        with self.assertRaises(ValueError):
            fit_block(self.initial, images[:64].split(16), 1e-3, "cpu", 12)
        with self.assertRaises(ValueError):
            fit_block(self.initial, images[:75].split(15), 1e-3, "cpu", 12)

    def test_prediction_is_label_free_and_returns_full_parameters(self):
        initial = initial_vector(self.initial)
        memories = initial + torch.randn(10, len(initial)) * 0.01
        checkpoint = {"experiment": "single_image_5update_v1", "cue_representation": "parameter_delta_v1",
                      "config": {"batch_size": 16, "lr": 1e-3}, "initial_vae": self.initial,
                      "memories": memories, "digits": list(range(10))}
        hopfield = make_retriever(len(initial), 128, 16)
        with patch("examples.vae_single_image_memory.evaluate.fit_block",
                   return_value=memories[3] - initial) as fit:
            result, _, retrieved = predict_image(torch.rand(784), checkpoint, hopfield, "cpu", 7)
        batches = list(fit.call_args.args[1])
        self.assertEqual([len(x) for x in batches], [16] * 5)
        self.assertEqual(result["predicted_digit"], 3)
        torch.testing.assert_close(retrieved, torch.tensor(result["associations"]) @ memories)
        with self.assertRaises(ValueError):
            predict_image(torch.rand(784), {**checkpoint, "experiment": "old"}, hopfield, "cpu", 7)
