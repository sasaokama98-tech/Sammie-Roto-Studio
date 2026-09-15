import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sammie.ground_truth_metrics import (
    GroundTruthResolutionMismatchError,
    connectivity_error,
    dtssd_metric,
    evaluate_ground_truth,
    resolve_ground_truth_path,
    spatial_metrics,
)


class GroundTruthMetricTests(unittest.TestCase):
    def test_identical_alpha_has_zero_error(self):
        alpha = np.zeros((16, 20), dtype=np.float32)
        alpha[3:13, 5:15] = 1.0
        alpha[3:13, 4] = 0.25

        metrics = spatial_metrics(alpha, alpha)

        self.assertEqual(metrics["sad"], 0.0)
        self.assertEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["gradient"], 0.0, places=12)
        self.assertEqual(metrics["connectivity"], 0.0)

    def test_spatial_errors_increase_for_wrong_alpha(self):
        ground_truth = np.zeros((16, 20), dtype=np.float32)
        ground_truth[3:13, 5:15] = 1.0
        prediction = np.zeros_like(ground_truth)

        metrics = spatial_metrics(prediction, ground_truth)

        self.assertGreater(metrics["sad"], 0.0)
        self.assertGreater(metrics["mse"], 0.0)
        self.assertGreater(metrics["gradient"], 0.0)
        self.assertGreater(metrics["connectivity"], 0.0)

    def test_dtssd_measures_temporal_error_not_static_bias(self):
        ground_truth_previous = np.zeros((4, 4), dtype=np.float32)
        ground_truth_current = np.full((4, 4), 0.25, dtype=np.float32)
        prediction_previous = np.full((4, 4), 0.1, dtype=np.float32)
        prediction_current = np.full((4, 4), 0.35, dtype=np.float32)

        stable_bias = dtssd_metric(
            prediction_current,
            prediction_previous,
            ground_truth_current,
            ground_truth_previous,
        )
        flicker = dtssd_metric(
            prediction_current + 0.2,
            prediction_previous,
            ground_truth_current,
            ground_truth_previous,
        )

        self.assertAlmostEqual(stable_bias, 0.0, places=5)
        self.assertGreater(flicker, 0.0)

    def test_sequence_report_compares_final_and_temporal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_dir = root / "final"
            temporal_dir = root / "temporal"
            ground_truth_dir = root / "ground_truth"
            for frame in range(2):
                ground_truth = np.zeros((12, 16), dtype=np.uint16)
                ground_truth[:, 4:12] = 65535
                temporal = ground_truth.copy()
                final = ground_truth.copy()
                if frame == 1:
                    final[:, 3:4] = 32768
                for base, value in (
                    (final_dir, final),
                    (temporal_dir, temporal),
                    (ground_truth_dir, ground_truth),
                ):
                    target = base / f"{frame:05d}"
                    target.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target / "0.png"), value)

            report = evaluate_ground_truth(
                final_dir=final_dir,
                temporal_dir=temporal_dir,
                ground_truth_dir=ground_truth_dir,
                frame_range=(0, 1),
                object_ids=[0],
            )

        item = report["objects"][0]
        self.assertEqual(item["temporal"]["aggregate"]["sad_mean"], 0.0)
        self.assertGreater(item["final"]["aggregate"]["sad_mean"], 0.0)
        self.assertGreater(item["final_minus_temporal"]["dtssd_mean"], 0.0)

    def test_flat_layout_is_only_allowed_for_single_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(
                str(root / "00000.png"), np.zeros((2, 2), dtype=np.uint8)
            )

            resolved = resolve_ground_truth_path(
                root, 0, 0, allow_flat=True
            )

            self.assertEqual(resolved.name, "00000.png")
            with self.assertRaises(FileNotFoundError):
                resolve_ground_truth_path(root, 0, 1, allow_flat=False)

    def test_source_sequence_number_is_preferred_over_internal_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(
                str(root / "0998.png"), np.zeros((2, 2), dtype=np.uint8)
            )
            cv2.imwrite(
                str(root / "00000.png"), np.ones((2, 2), dtype=np.uint8)
            )

            resolved = resolve_ground_truth_path(
                root,
                0,
                0,
                allow_flat=True,
                source_frame=998,
                source_padding=4,
            )

        self.assertEqual(resolved.name, "0998.png")

    def test_sequence_requires_every_ground_truth_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for base in (root / "final", root / "temporal"):
                for frame in range(2):
                    target = base / f"{frame:05d}"
                    target.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(target / "0.png"),
                        np.zeros((4, 4), dtype=np.uint8),
                    )
            target = root / "ground_truth" / "00000"
            target.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(target / "0.png"), np.zeros((4, 4), dtype=np.uint8)
            )

            with self.assertRaisesRegex(FileNotFoundError, "frame 1"):
                evaluate_ground_truth(
                    final_dir=root / "final",
                    temporal_dir=root / "temporal",
                    ground_truth_dir=root / "ground_truth",
                    frame_range=(0, 1),
                    object_ids=[0],
                )

    def test_sequence_rejects_ground_truth_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for base, shape in (
                (root / "final", (4, 4)),
                (root / "temporal", (4, 4)),
                (root / "ground_truth", (3, 4)),
            ):
                target = base / "00000"
                target.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(target / "0.png"), np.zeros(shape, dtype=np.uint8)
                )

            with self.assertRaisesRegex(
                GroundTruthResolutionMismatchError, "does not match"
            ) as caught:
                evaluate_ground_truth(
                    final_dir=root / "final",
                    temporal_dir=root / "temporal",
                    ground_truth_dir=root / "ground_truth",
                    frame_range=(0, 0),
                    object_ids=[0],
                )
            self.assertEqual(caught.exception.ground_truth_shape, (3, 4))
            self.assertEqual(caught.exception.matte_shape, (4, 4))
            self.assertIn("No automatic resize", str(caught.exception))

    def test_connectivity_rejects_shape_change(self):
        self.assertGreater(
            connectivity_error(
                np.zeros((4, 4), dtype=np.float32),
                np.ones((4, 4), dtype=np.float32),
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
