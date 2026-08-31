import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sammie import core
from sammie import sammie as sammie_runtime
from sammie.removal import RemovalManager


class RemovalOutputTests(unittest.TestCase):
    def test_output_path_is_png_independent_of_source_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            core, "removal_dir", temp_dir
        ), patch.object(core, "get_frame_extension", return_value="jpg"):
            path = RemovalManager.removal_output_path(12)

        self.assertEqual(path, os.path.join(temp_dir, "00012.png"))

    def test_write_removal_frame_creates_lossless_readable_output(self):
        frame = np.zeros((5, 7, 3), dtype=np.uint8)
        frame[:, :, 1] = 173

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            core, "removal_dir", temp_dir
        ):
            path = RemovalManager.write_removal_frame(3, frame)
            loaded = cv2.imread(path, cv2.IMREAD_COLOR)

        self.assertTrue(path.endswith("00003.png"))
        np.testing.assert_array_equal(loaded, frame)

    def test_png_output_is_loaded_by_removal_preview_for_jpg_source(self):
        frame_bgr = np.zeros((4, 6, 3), dtype=np.uint8)
        frame_bgr[:, :, 2] = 211

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            core, "removal_dir", temp_dir
        ), patch.object(core, "get_frame_extension", return_value="jpg"), patch.object(
            core, "load_base_frame", side_effect=AssertionError("unexpected fallback")
        ):
            RemovalManager.write_removal_frame(4, frame_bgr)
            loaded_rgb = sammie_runtime.load_removal_frame(4)

        np.testing.assert_array_equal(
            loaded_rgb, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )

    def test_write_removal_frame_raises_when_opencv_cannot_save(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            core, "removal_dir", temp_dir
        ), patch("sammie.removal.cv2.imwrite", return_value=False):
            with self.assertRaisesRegex(IOError, "Failed to write removal frame"):
                RemovalManager.write_removal_frame(8, frame)


if __name__ == "__main__":
    unittest.main()
