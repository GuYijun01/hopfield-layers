import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from examples.vae_parameter_memory.hopfield import make_retriever, retriever_from_checkpoint
from examples.vae_single_image_memory.block_value_control import run


class BlockControlTests(unittest.TestCase):
    def test_control_preserves_trained_projections_predictions_source_and_caches(self):
        torch.manual_seed(67)
        torch.set_num_threads(1)
        base = {"signature": "control", "digits": []}
        for digit in range(10):
            base["digits"].append({"digit": digit,
                "train": {"features": torch.randn(3, 8), "sources": torch.arange(3) + digit * 10,
                          "seeds": torch.arange(3) + 1042 + digit * 10000},
                "validation": {"features": torch.randn(1, 8), "sources": torch.tensor([digit * 10 + 9]),
                               "seeds": torch.tensor([1000000 + digit])}})
        model = make_retriever(16, 8, 4, projection_mode="separate")
        checkpoint = {"experiment": "single_image_5update_v1", "cue_representation": "parameter_delta_v1",
                      "projection_mode": "separate", "config": {"seed": 42, "batch_size": 16,
                        "feature_size": 8, "association_size": 4, "projection_mode": "separate"},
                      "parameter_count": 16, "initial_vae": {"parameters": torch.randn(16)},
                      "memories": torch.randn(10, 16), "digits": list(range(10)), "hopfield": model.state_dict(),
                      "cue_features": torch.cat([row["train"]["features"] for row in base["digits"]]),
                      "cue_source_indices": torch.cat([row["train"]["sources"] for row in base["digits"]]),
                      "cue_seeds": torch.cat([row["train"]["seeds"] for row in base["digits"]]),
                      "targets": torch.arange(10).repeat_interleave(3)}
        with tempfile.TemporaryDirectory() as directory:
            source, cache = Path(directory)/"source.pt", Path(directory)/"cache.pt"
            torch.save(checkpoint, source)
            torch.save(base, cache)
            source_bytes, cache_bytes = source.read_bytes(), cache.read_bytes()
            args = Namespace(source=str(source), cue_cache=str(cache), seed_cache=None,
                             output_prefix=str(Path(directory)/"block"), backup_prefix=None,
                             value_block_size=4, threads=1, device="cpu")
            report = run(args)
            self.assertEqual(report["optimizer_updates"], 0)
            self.assertLess(report["roundtrip"]["max_abs_roundtrip_error"], 1e-12)
            self.assertLess(report["max_soft_mixture_roundtrip_error"], 1e-12)
            for case in report["comparisons"].values():
                self.assertTrue(case["same_predictions"])
                self.assertEqual(case["max_association_difference"], 0)
            attached = torch.load(args.output_prefix + "_checkpoint.pt", weights_only=True)
            loaded = retriever_from_checkpoint(attached, "cpu")
            for name, value in checkpoint["hopfield"].items():
                torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(cache.read_bytes(), cache_bytes)
            self.assertFalse(any(path.is_dir() for path in Path(directory).iterdir()))
