import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from sammie.ground_truth_metrics import GroundTruthResolutionMismatchError
from sammie.matting import HybridHQManager


class GroundTruthResolutionWarningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_resolution_mismatch_warns_and_keeps_completed_matte(self):
        settings = MagicMock()
        values = {"hybrid_ground_truth_dir": "gt", "hybrid_evaluation_label": "run"}
        settings.get_session_setting.side_effect = (
            lambda key, default=None: values.get(key, default)
        )
        manager = HybridHQManager()
        manager._get_frame_range = MagicMock(return_value=(0, 0, 1))
        manager._active_output_ids = MagicMock(return_value=[0])
        mismatch = GroundTruthResolutionMismatchError(
            frame=0,
            object_id=0,
            source=Path("gt/00000/0.png"),
            ground_truth_shape=(2160, 3840),
            matte_shape=(1080, 1920),
            matte_name="final",
        )
        with (
            patch("sammie.matting.get_settings_manager", return_value=settings),
            patch("sammie.matting.evaluate_hybrid_run", side_effect=mismatch),
            patch("sammie.matting.QProgressDialog"),
            patch("sammie.matting.QMessageBox.warning") as warning,
        ):
            result = manager._run_evaluation_stage(
                [], parent_window=MagicMock(), combined=False,
                temporal_model="MatAnyone2",
            )
        self.assertIsNone(result)
        warning.assert_called_once()
        message = warning.call_args.args[2]
        self.assertIn("3840x2160", message)
        self.assertIn("1920x1080", message)
        self.assertIn("GT scoring was skipped", message)
        self.assertIn("matte is preserved", message)


if __name__ == "__main__":
    unittest.main()
