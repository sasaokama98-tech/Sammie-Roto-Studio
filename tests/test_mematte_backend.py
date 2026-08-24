import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from sammie.mematte_backend import (
    MematteBackend,
    MematteUnavailableError,
    resolve_mematte_paths,
)


class _FakeBackbone:
    max_number_token = 0


class _FakeModel:
    def __init__(self, alpha=0.375):
        self.alpha = alpha
        self.backbone = _FakeBackbone()
        self.batch = None

    def __call__(self, batch, patch_decoder=True):
        self.batch = batch
        height, width = batch["trimap"].shape[-2:]
        alpha = torch.full((1, 1, height, width), self.alpha)
        return {"phas": alpha}, [], []


class MematteBackendTests(unittest.TestCase):
    def test_explicit_paths_are_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = resolve_mematte_paths(root / "repo", root / "model.pth")
            self.assertEqual(paths.repo, (root / "repo").resolve())
            self.assertEqual(paths.checkpoint, (root / "model.pth").resolve())

    def test_environment_paths_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "MEMATTE_REPO_PATH": str(root / "source"),
                "MEMATTE_CHECKPOINT_PATH": str(root / "weights.pth"),
            }
            with patch.dict(os.environ, environment, clear=False):
                paths = resolve_mematte_paths()
            self.assertEqual(paths.repo, (root / "source").resolve())
            self.assertEqual(paths.checkpoint, (root / "weights.pth").resolve())

    def test_single_checkpoint_is_discovered_without_renaming(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            downloaded = checkpoint_dir / "model_final.pth"
            downloaded.touch()
            with patch(
                "sammie.mematte_backend.DEFAULT_CHECKPOINT_DIR", checkpoint_dir
            ):
                paths = resolve_mematte_paths()
            self.assertEqual(paths.checkpoint, downloaded.resolve())

    def test_invalid_memory_controls_are_rejected(self):
        with self.assertRaises(ValueError):
            MematteBackend(torch.device("cpu"), max_number_token=0)
        with self.assertRaises(ValueError):
            MematteBackend(torch.device("cpu"), precision="Int8")

    def test_predict_quantizes_trimap_and_applies_token_limit(self):
        backend = MematteBackend(
            torch.device("cpu"), max_number_token=8192, precision="Float32"
        )
        backend.model = _FakeModel()
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        trimap = np.array(
            [[0, 84, 85, 128, 255]] * 4,
            dtype=np.uint8,
        )

        alpha = backend.predict(rgb, trimap)

        self.assertEqual(alpha.shape, trimap.shape)
        self.assertTrue(np.allclose(alpha, 0.375))
        self.assertEqual(backend.model.backbone.max_number_token, 8192)
        normalized = backend.model.batch["trimap"][0, 0, 0].numpy()
        self.assertTrue(np.array_equal(normalized, [0.0, 0.0, 0.5, 0.5, 1.0]))

    def test_tiled_integration_preserves_known_regions(self):
        backend = MematteBackend(torch.device("cpu"), precision="Float32")
        backend.model = _FakeModel(alpha=0.5001)
        rgb = np.zeros((64, 96, 3), dtype=np.uint8)
        trimap = np.zeros((64, 96), dtype=np.uint8)
        trimap[8:56, 10:86] = 128
        trimap[20:44, 30:66] = 255

        alpha = backend.predict_multi_roi_float(
            rgb, trimap, margin=4, max_tile_size=32, tile_overlap=8
        )

        self.assertEqual(alpha.dtype, np.float32)
        self.assertEqual(float(alpha[0, 0]), 0.0)
        self.assertEqual(float(alpha[32, 48]), 1.0)
        self.assertAlmostEqual(float(alpha[10, 12]), 0.5001, places=4)

    def test_checkpoint_state_filters_wrappers_and_teacher(self):
        state = MematteBackend._checkpoint_state(
            {
                "model": {
                    "module.backbone.weight": torch.ones(1),
                    "module.decoder.weight": torch.ones(1),
                    "module.teacher_backbone.weight": torch.ones(1),
                }
            }
        )
        self.assertEqual(set(state), {"backbone.weight", "decoder.weight"})

    def test_missing_source_has_actionable_error(self):
        backend = MematteBackend(
            torch.device("cpu"),
            repo_path="missing-mematte-source",
            checkpoint_path="missing-mematte-checkpoint.pth",
        )
        with self.assertRaisesRegex(MematteUnavailableError, "MEMATTE_REPO_PATH"):
            backend.load()


if __name__ == "__main__":
    unittest.main()
