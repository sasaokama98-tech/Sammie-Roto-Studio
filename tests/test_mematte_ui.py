import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from sammie_main import MattingTab


class MematteUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_mematte_controls_follow_model_selection(self):
        tab = MattingTab()
        tab.show()
        tab.matany_model_combo.setCurrentText("MEMatte")
        self.app.processEvents()

        self.assertTrue(tab.mematte_margin_spin.isVisible())
        self.assertTrue(tab.mematte_tile_spin.isVisible())
        self.assertTrue(tab.mematte_overlap_spin.isVisible())
        self.assertTrue(tab.mematte_tokens_spin.isVisible())
        self.assertTrue(tab.mematte_precision_combo.isVisible())
        self.assertFalse(tab.vitmatte_margin_spin.isVisible())
        tab.close()

    def test_memory_profiles_apply_as_a_group_and_manual_edit_becomes_custom(self):
        tab = MattingTab()
        tab.show()
        tab.memory_profile_combo.setCurrentText("Memory Safe")
        self.app.processEvents()

        self.assertEqual(tab.matany_res_combo.currentText(), "480")
        self.assertEqual(tab.mematte_tile_spin.value(), 1024)
        self.assertEqual(tab.mematte_tokens_spin.value(), 6144)
        self.assertEqual(tab.hybrid_flow_resolution_combo.currentText(), "480")

        tab.mematte_tile_spin.setValue(1152)
        self.app.processEvents()
        self.assertEqual(tab.memory_profile_combo.currentText(), "Custom")

        tab.memory_profile_combo.setCurrentText("Fast")
        self.app.processEvents()
        self.assertEqual(tab.matany_res_combo.currentText(), "1080")
        self.assertEqual(tab.chunk_combo.currentText(), "32")
        self.assertEqual(tab.mematte_tile_spin.value(), 3072)
        self.assertEqual(tab.mematte_tokens_spin.value(), 18000)
        tab.close()

    def test_hybrid_controls_include_temporal_and_mematte_settings(self):
        tab = MattingTab()
        tab.show()
        tab.matany_model_combo.setCurrentText("Hybrid HQ")
        self.app.processEvents()

        self.assertTrue(tab.hybrid_temporal_combo.isVisible())
        self.assertTrue(tab.hybrid_stability_combo.isVisible())
        self.assertEqual(
            tab.hybrid_stability_combo.currentText(), "Preserve Temporal"
        )
        self.assertTrue(tab.hybrid_motion_checkbox.isVisible())
        self.assertFalse(tab.hybrid_motion_checkbox.isChecked())
        self.assertFalse(tab.hybrid_flow_resolution_combo.isEnabled())
        self.assertTrue(tab.hybrid_evaluation_checkbox.isVisible())
        self.assertFalse(tab.hybrid_evaluation_checkbox.isChecked())
        self.assertFalse(tab.hybrid_evaluation_edit.isEnabled())
        self.assertTrue(tab.performance_metrics_checkbox.isVisible())
        tab.hybrid_motion_checkbox.setChecked(True)
        self.app.processEvents()
        self.assertTrue(tab.hybrid_flow_resolution_combo.isEnabled())
        tab.hybrid_motion_checkbox.setChecked(False)
        tab.hybrid_evaluation_checkbox.setChecked(True)
        self.app.processEvents()
        self.assertTrue(tab.hybrid_evaluation_edit.isEnabled())
        self.assertTrue(tab.hybrid_edge_spin.isVisible())
        self.assertTrue(tab.hybrid_feather_spin.isVisible())
        self.assertTrue(tab.mematte_tokens_spin.isVisible())
        self.assertFalse(tab.vitmatte_margin_spin.isVisible())

        tab.hybrid_temporal_combo.setCurrentText("VideoMaMa")
        self.app.processEvents()
        self.assertTrue(tab.overlap_combo.isVisible())
        self.assertTrue(tab.chunk_combo.isVisible())
        tab.close()

    def test_processing_settings_use_fixed_rows_inside_scrollable_tab(self):
        tab = MattingTab()
        tab.resize(320, 240)
        tab.show()
        tab.matany_model_combo.setCurrentText("Hybrid HQ")
        self.app.processEvents()

        self.assertTrue(tab.content_scroll.widgetResizable())
        self.assertEqual(
            tab.content_scroll.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff
        )
        self.assertEqual(tab.matany_model_combo.minimumHeight(), 26)
        self.assertEqual(tab.matany_model_combo.maximumHeight(), 26)
        self.assertEqual(tab.memory_profile_combo.minimumHeight(), 26)
        self.assertEqual(tab.memory_profile_combo.maximumHeight(), 26)
        self.assertEqual(tab.performance_metrics_checkbox.minimumHeight(), 26)
        self.assertEqual(tab.performance_metrics_checkbox.maximumHeight(), 26)
        self.assertEqual(tab.hybrid_edge_spin.minimumHeight(), 26)
        self.assertEqual(tab.hybrid_edge_spin.maximumHeight(), 26)
        self.assertGreaterEqual(tab.content_scroll.verticalScrollBar().maximum(), 1)
        tab.close()


if __name__ == "__main__":
    unittest.main()
