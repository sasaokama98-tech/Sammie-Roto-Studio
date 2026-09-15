import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sammie.hybrid_evaluation import (
    HybridEvaluationCancelled,
    evaluate_hybrid_run,
    sanitize_run_label,
)


class HybridEvaluationTests(unittest.TestCase):
    def _make_sequence(self, root: Path):
        frames = root / "frames"
        mattes = root / "matting"
        temporal = root / "hybrid_temporal"
        trimaps = root / "hybrid_trimaps"
        confidence = root / "hybrid_motion_confidence"
        frames.mkdir()
        for frame in range(3):
            image = np.tile(np.arange(32, dtype=np.uint8), (24, 1))
            cv2.imwrite(str(frames / f"{frame:05d}.png"), image)
            base = np.zeros((24, 32), dtype=np.float32)
            base[:, 10:22] = 1.0
            base[:, 8:10] = 0.25
            base[:, 22:24] = 0.75
            trimap = np.zeros((24, 32), dtype=np.uint8)
            trimap[:, 6:26] = 128
            trimap[:, 12:20] = 255
            final = base.copy()
            final[trimap == 128] = np.clip(
                final[trimap == 128] + frame * 0.01, 0.0, 1.0
            )
            for directory, value in (
                (mattes, np.round(final * 65535).astype(np.uint16)),
                (temporal, np.round(base * 65535).astype(np.uint16)),
                (trimaps, trimap),
            ):
                target = directory / f"{frame:05d}"
                target.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target / "0.png"), value)
            if frame == 1:
                target = confidence / f"{frame:05d}"
                target.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(target / "0.png"),
                    np.full((24, 32), 192, dtype=np.uint8),
                )
        return frames, mattes, temporal, trimaps, confidence

    def test_archives_run_and_writes_no_reference_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._make_sequence(root)
            report_path = evaluate_hybrid_run(
                frames_dir=paths[0],
                matting_dir=paths[1],
                temporal_dir=paths[2],
                trimap_dir=paths[3],
                confidence_dir=paths[4],
                output_root=root / "evaluation",
                frame_range=(0, 2),
                object_ids=[0],
                frame_extension="png",
                run_label="VideoMaMa / Preserve Temporal",
                settings={
                    "temporal_model": "VideoMaMa",
                    "stability_preset": "Preserve Temporal",
                    "motion_enabled": True,
                    "flow_resolution": 32,
                },
                flow_resolution=32,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            aggregate = report["objects"][0]["aggregate"]
            self.assertEqual(report["phase"], "4.3")
            self.assertIn("not ground-truth", report["ground_truth_notice"])
            self.assertGreater(aggregate["unknown_residual_mean"], 0.0)
            self.assertEqual(aggregate["known_region_drift_mean"], 0.0)
            self.assertIsNotNone(aggregate["motion_confidence_mean"])
            self.assertTrue(
                (report_path.parent / "final" / "00002" / "0.png").exists()
            )
            self.assertTrue(
                (report_path.parent / "temporal" / "00000" / "0.png").exists()
            )
            self.assertTrue(
                (report_path.parent / "confidence" / "00001" / "0.png").exists()
            )
            self.assertTrue((root / "evaluation" / "summary.csv").exists())

    def test_cancel_removes_only_new_incomplete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._make_sequence(root)
            output = root / "evaluation"
            existing = output / "keep_me"
            existing.mkdir(parents=True)
            with self.assertRaises(HybridEvaluationCancelled):
                evaluate_hybrid_run(
                    frames_dir=paths[0],
                    matting_dir=paths[1],
                    temporal_dir=paths[2],
                    trimap_dir=paths[3],
                    confidence_dir=paths[4],
                    output_root=output,
                    frame_range=(0, 2),
                    object_ids=[0],
                    frame_extension="png",
                    run_label="cancelled",
                    settings={},
                    cancel_callback=lambda: True,
                )
            self.assertTrue(existing.exists())
            self.assertFalse((output / "cancelled").exists())

    def test_ground_truth_metrics_and_archive_are_added_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._make_sequence(root)
            ground_truth = root / "ground_truth"
            for frame in range(3):
                source = paths[2] / f"{frame:05d}" / "0.png"
                target = ground_truth / f"{frame:05d}"
                target.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(target / "0.png"),
                    cv2.imread(str(source), cv2.IMREAD_UNCHANGED),
                )

            report_path = evaluate_hybrid_run(
                frames_dir=paths[0],
                matting_dir=paths[1],
                temporal_dir=paths[2],
                trimap_dir=paths[3],
                confidence_dir=paths[4],
                output_root=root / "evaluation",
                frame_range=(0, 2),
                object_ids=[0],
                frame_extension="png",
                run_label="ground_truth",
                settings={"temporal_model": "MatAnyone2"},
                ground_truth_dir=ground_truth,
                flow_resolution=32,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertIn("ground_truth_metrics", report)
            self.assertEqual(report["phase"], "4.4")
            result = report["ground_truth_metrics"]["objects"][0]
            self.assertEqual(result["temporal"]["aggregate"]["sad_mean"], 0.0)
            self.assertGreater(result["final"]["aggregate"]["sad_mean"], 0.0)
            self.assertTrue(
                (report_path.parent / "ground_truth" / "00002" / "0.png").exists()
            )
            self.assertTrue(
                (root / "evaluation" / "ground_truth_summary.csv").exists()
            )

    def test_run_label_is_sanitized(self):
        self.assertEqual(sanitize_run_label("  A/B: C  "), "A_B_C")


if __name__ == "__main__":
    unittest.main()
