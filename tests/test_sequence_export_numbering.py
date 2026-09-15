import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMainWindow

from sammie.export_dialog import ExportDialog
from sammie.export_formats import ExportSettings
from sammie.export_workers import SequenceExportWorker
from sammie.core import VideoInfo


def make_settings(format_id="png", start_number=1001, in_point=None, out_point=None):
    return ExportSettings(
        format_id=format_id,
        output_dir="output",
        filename_template="shot",
        output_type="Segmentation-Matte",
        object_id=-1,
        antialias=False,
        quality=14,
        use_inout=in_point is not None and out_point is not None,
        in_point=in_point,
        out_point=out_point,
        sequence_start_number=start_number,
    )


class SequenceExportNumberingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_full_sequence_starts_at_requested_number(self):
        worker = SequenceExportWorker(
            make_settings(start_number=1001), [], 3, "shot"
        )

        self.assertEqual(worker.frame_filename(0), "shot.1001.png")
        self.assertEqual(worker.frame_filename(2), "shot.1003.png")

    def test_in_out_source_range_is_renumbered_from_requested_start(self):
        worker = SequenceExportWorker(
            make_settings(start_number=1010, in_point=50, out_point=52),
            [],
            100,
            "shot",
        )

        self.assertEqual(worker.frame_filename(50), "shot.1010.png")
        self.assertEqual(worker.frame_filename(52), "shot.1012.png")

    def test_exr_and_negative_start_numbers_use_the_same_mapping(self):
        worker = SequenceExportWorker(
            make_settings(format_id="exr", start_number=-2), [], 2, "matte"
        )

        self.assertEqual(worker.frame_filename(0), "matte.-002.exr")
        self.assertEqual(worker.frame_filename(1), "matte.-001.exr")

    def test_export_dialog_exposes_start_number_for_sequences(self):
        class StubSettings:
            @staticmethod
            def get_app_setting(_key, default=None):
                return default

            @staticmethod
            def get_session_setting(_key, default=None):
                return default

        parent = QMainWindow()
        parent.settings_mgr = StubSettings()
        parent.point_manager = type(
            "StubPoints", (), {"get_all_points": lambda self: []}
        )()

        with patch.object(VideoInfo, "total_frames", 3):
            dialog = ExportDialog(parent)
            png_index = dialog.format_combo.findData("png")
            dialog.format_combo.setCurrentIndex(png_index)
            dialog.sequence_start_spin.setValue(1001)
            self.app.processEvents()

            self.assertFalse(dialog.advanced_export_content.isVisibleTo(dialog))
            self.assertFalse(dialog.save_settings_btn.isVisibleTo(dialog))
            self.assertTrue(dialog.sequence_start_spin.isVisibleTo(dialog))
            self.assertIn(".1001.png", dialog.filename_preview_label.text())
            self.assertIn(".1003.png", dialog.filename_preview_label.text())
            self.assertEqual(
                dialog._build_export_settings().sequence_start_number, 1001
            )
            dialog.advanced_export_button.setChecked(True)
            self.app.processEvents()
            self.assertTrue(dialog.save_settings_btn.isVisibleTo(dialog))
            self.assertTrue(dialog.object_id_combo.isVisibleTo(dialog))

            prores_index = dialog.format_combo.findData("prores")
            dialog.format_combo.setCurrentIndex(prores_index)
            self.assertFalse(dialog.sequence_start_spin.isVisibleTo(dialog))
            dialog.close()
        parent.close()


if __name__ == "__main__":
    unittest.main()
