import os
import json
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from sammie.settings_manager import ApplicationSettings, SessionSettings, SettingsManager
from sammie.settings_dialog import SettingsDialog
from sammie_main import MattingTab, SegmentationTab


class WorkflowDefaultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_new_workflow_diagnostics_and_pregrade_are_opt_in(self):
        app = ApplicationSettings()
        session = SessionSettings()
        self.assertFalse(app.default_performance_metrics_enabled)
        self.assertFalse(app.default_hybrid_evaluation_enabled)
        self.assertFalse(app.default_segmentation_auto_pregrade_enabled)
        self.assertFalse(app.default_matting_auto_pregrade_enabled)
        self.assertFalse(session.performance_metrics_enabled)
        self.assertFalse(session.hybrid_evaluation_enabled)
        self.assertFalse(session.segmentation_auto_pregrade_enabled)
        self.assertFalse(session.matting_auto_pregrade_enabled)

    def test_pregrade_is_accessible_in_basic_model_controls(self):
        seg = SegmentationTab()
        mat = MattingTab()
        seg.show()
        mat.show()
        self.app.processEvents()
        self.assertTrue(seg.segmentation_pregrade_checkbox.isVisible())
        self.assertTrue(mat.matting_pregrade_checkbox.isVisible())
        self.assertFalse(mat.advanced_processing_content.isVisible())
        seg.close()
        mat.close()

    def test_legacy_always_on_metrics_migrate_off_once(self):
        with tempfile.TemporaryDirectory() as root:
            manager = SettingsManager(root)
            manager.app_settings_file = os.path.join(root, "app.conf")
            manager.session_settings_file = os.path.join(root, "session.conf")
            with open(manager.app_settings_file, "w", encoding="utf-8") as file:
                json.dump({"default_performance_metrics_enabled": True}, file)
            with open(manager.session_settings_file, "w", encoding="utf-8") as file:
                json.dump({"performance_metrics_enabled": True}, file)
            self.assertTrue(manager.load_app_settings())
            self.assertTrue(manager.load_session_settings())
            self.assertFalse(manager.app_settings.default_performance_metrics_enabled)
            self.assertFalse(manager.session_settings.performance_metrics_enabled)
            manager.save_app_settings()
            manager.save_session_settings()
            manager.app_settings.default_performance_metrics_enabled = True
            manager.session_settings.performance_metrics_enabled = True
            manager.save_app_settings()
            manager.save_session_settings()
            self.assertTrue(manager.load_app_settings())
            self.assertTrue(manager.load_session_settings())
            self.assertTrue(manager.app_settings.default_performance_metrics_enabled)
            self.assertTrue(manager.session_settings.performance_metrics_enabled)

    def test_global_defaults_keep_diagnostics_under_advanced(self):
        with tempfile.TemporaryDirectory() as root:
            manager = SettingsManager(root)
            dialog = SettingsDialog(manager)
            dialog.show()
            dialog.tab_widget.setCurrentIndex(1)
            self.app.processEvents()
            self.assertFalse(dialog.advanced_defaults_content.isVisible())
            self.assertFalse(dialog.default_performance_metrics_checkbox.isVisible())
            dialog.show_advanced_defaults_checkbox.setChecked(True)
            self.app.processEvents()
            self.assertTrue(dialog.default_performance_metrics_checkbox.isVisible())
            self.assertTrue(dialog.default_hybrid_evaluation_checkbox.isVisible())
            dialog.reject()


if __name__ == "__main__":
    unittest.main()
