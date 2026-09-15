import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sammie.auto_pregrade import (
    Pregrade, apply_pregrade, estimate_pregrade, inference_frames_dir,
)


class AutoPregradeTests(unittest.TestCase):
    def test_disabled_uses_original_directory(self):
        self.assertEqual(
            inference_frames_dir("source", "temp", "png", 10, "matting", False),
            "source",
        )

    def test_one_grade_is_shared_across_frames_and_source_is_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            source.mkdir()
            originals = []
            for index, value in enumerate((42, 48, 54)):
                frame = np.full((24, 32, 3), value, dtype=np.uint8)
                originals.append(frame.copy())
                self.assertTrue(cv2.imwrite(str(source / f"{index:05d}.png"), frame))
            grade = estimate_pregrade(
                [str(source / f"{index:05d}.png") for index in range(3)]
            )
            self.assertGreater(grade.gain, 1.0)
            stage = inference_frames_dir(
                str(source), root, "png", 3, "matting", True
            )
            self.assertNotEqual(stage, str(source))
            self.assertEqual(len(list(Path(stage).glob("*.png"))), 3)
            for index, original in enumerate(originals):
                np.testing.assert_array_equal(
                    cv2.imread(str(source / f"{index:05d}.png")), original
                )
                np.testing.assert_array_equal(
                    cv2.imread(str(Path(stage) / f"{index:05d}.png")),
                    apply_pregrade(original, grade),
                )
            # A workspace cleanup must not leave a stale cached stage path.
            os.remove(Path(stage) / "00001.png")
            stage_again = inference_frames_dir(
                str(source), root, "png", 3, "matting", True
            )
            self.assertTrue((Path(stage_again) / "00001.png").is_file())

    def test_grade_is_bounded_and_rejects_non_bgr(self):
        image = np.full((8, 8, 3), 220, dtype=np.uint8)
        graded = apply_pregrade(image, Pregrade(gain=2.0))
        self.assertEqual(int(graded.max()), 255)
        with self.assertRaises(ValueError):
            apply_pregrade(np.zeros((8, 8), dtype=np.uint8), Pregrade())

    def test_jpeg_sequence_keeps_predictor_frame_count(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            source.mkdir()
            for index in range(2):
                frame = np.full((16, 24, 3), 80 + index * 10, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(source / f"{index:05d}.jpg"), frame))
            stage = Path(inference_frames_dir(
                str(source), root, "jpg", 2, "segmentation", True
            ))
            self.assertEqual([path.name for path in sorted(stage.glob("*.jpg"))],
                             ["00000.jpg", "00001.jpg"])
            self.assertTrue((stage / "_pending").is_dir())


if __name__ == "__main__":
    unittest.main()
