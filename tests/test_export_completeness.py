import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import av
import Imath
import numpy as np
import OpenEXR

from sammie import core, sammie
from sammie.export_formats import ExportSettings
from sammie.export_workers import SequenceExportWorker, VideoExportWorker


def export_settings(directory, format_id, output_type):
    return ExportSettings(
        format_id=format_id,
        output_dir=directory,
        filename_template="shot",
        output_type=output_type,
        object_id=0,
        antialias=False,
        quality=14,
        use_inout=False,
        in_point=None,
        out_point=None,
        sequence_start_number=1001,
    )


class ExportCompletenessTests(unittest.TestCase):
    def test_exr_layer_set_keeps_authored_objects_without_any_matte(self):
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = os.path.join(directory, "00000")
            os.makedirs(frame_dir)
            cv2.imwrite(
                os.path.join(frame_dir, "00000.png"),
                np.full((2, 2), 255, dtype=np.uint8),
            )
            with patch.object(core, "matting_dir", directory):
                worker = SequenceExportWorker(
                    export_settings(directory, "exr", "Matting-Matte"),
                    [
                        {"frame": 0, "object_id": 0},
                        {"frame": 0, "object_id": 1},
                    ],
                    1,
                    "shot",
                )

                object_ids = worker._exr_object_ids()

        self.assertEqual(object_ids, [0, 1])

    def test_missing_segmentation_matte_renders_as_zero(self):
        with patch.object(core.VideoInfo, "height", 4), patch.object(
            core.VideoInfo, "width", 6
        ), patch.object(core, "load_masks_for_frame", return_value=None):
            frame = sammie.update_image(
                0,
                {"view_mode": "Segmentation-Matte", "antialias": False},
                [{"frame": 0, "object_id": 0}],
                return_numpy=True,
                object_id_filter=0,
            )

        self.assertEqual(frame.shape, (4, 6, 3))
        self.assertFalse(np.any(frame))

    def test_missing_alpha_mask_keeps_rgb_with_zero_alpha(self):
        source = np.full((4, 6, 3), 73, dtype=np.uint8)
        with patch.object(core, "load_base_frame", return_value=source), patch.object(
            core, "load_masks_for_frame", return_value=None
        ):
            frame = sammie.update_image(
                0,
                {"view_mode": "Matting-Alpha"},
                [{"frame": 0, "object_id": 0}],
                return_numpy=True,
                object_id_filter=0,
            )

        np.testing.assert_array_equal(frame[:, :, :3], source)
        self.assertFalse(np.any(frame[:, :, 3]))

    def test_png_sequence_writes_a_zero_frame_for_every_missing_mask(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            core.VideoInfo, "height", 4
        ), patch.object(core.VideoInfo, "width", 6), patch.object(
            core, "load_masks_for_frame", return_value=None
        ):
            worker = SequenceExportWorker(
                export_settings(directory, "png", "Segmentation-Matte"),
                [{"frame": 0, "object_id": 0}],
                3,
                "shot",
            )
            worker._export_png_sequence()

            paths = [
                os.path.join(directory, f"shot.{number:04d}.png")
                for number in range(1001, 1004)
            ]
            frames = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in paths]

        self.assertTrue(all(frame is not None for frame in frames))
        self.assertTrue(all(not np.any(frame) for frame in frames))

    def test_exr_sequence_keeps_zero_layer_and_exact_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            matte_dir = os.path.join(directory, "mattes")
            os.makedirs(matte_dir)
            with patch.object(core.VideoInfo, "height", 4), patch.object(
                core.VideoInfo, "width", 6
            ), patch.object(core, "matting_dir", matte_dir), patch.object(
                core, "load_matte_for_export", return_value=None
            ):
                worker = SequenceExportWorker(
                    export_settings(directory, "exr", "Matting-Matte"),
                    [{"frame": 0, "object_id": 0}],
                    3,
                    "shot",
                )
                worker._export_exr_sequence()

            paths = [
                os.path.join(directory, f"shot.{number:04d}.exr")
                for number in range(1001, 1004)
            ]
            restored = []
            for path in paths:
                exr = OpenEXR.InputFile(path)
                raw = exr.channel(
                    "Object_0.Y", Imath.PixelType(Imath.PixelType.FLOAT)
                )
                restored.append(np.frombuffer(raw, dtype=np.float32))
                exr.close()

        self.assertEqual(len(restored), 3)
        self.assertTrue(all(not np.any(frame) for frame in restored))

    def test_png_write_failure_is_not_reported_as_success(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory, patch(
            "sammie.export_workers.cv2.imwrite", return_value=False
        ):
            with self.assertRaisesRegex(IOError, "Failed to write PNG frame"):
                SequenceExportWorker._atomic_write_png(
                    os.path.join(directory, "shot.1001.png"), frame
                )

    def test_video_export_decodes_to_exact_requested_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "shot.mkv")
            settings = export_settings(
                directory, "ffv1", "Segmentation-Matte"
            )
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            with patch.object(core.VideoInfo, "height", 16), patch.object(
                core.VideoInfo, "width", 16
            ), patch.object(core.VideoInfo, "fps", 24.0), patch.object(
                core.VideoInfo, "color_space", 1
            ), patch(
                "sammie.export_workers.sammie.update_image",
                return_value=frame,
            ):
                worker = VideoExportWorker(
                    settings, [], 3, [path], [-1]
                )
                worker._export_single(path, -1)

            with av.open(path) as container:
                decoded_count = sum(1 for _ in container.decode(video=0))

        self.assertEqual(decoded_count, 3)


if __name__ == "__main__":
    unittest.main()
