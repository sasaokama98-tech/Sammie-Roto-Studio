import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from sammie_main import SegmentationTab


class Sam31UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_anchor_notice_is_only_visible_for_sam31(self):
        tab = SegmentationTab()
        tab.show()

        tab.sam_model_combo.setCurrentText("Base")
        self.app.processEvents()
        self.assertFalse(tab.sam31_anchor_notice.isVisible())

        tab.sam_model_combo.setCurrentText("SAM 3.1")
        self.app.processEvents()
        self.assertTrue(tab.sam31_anchor_notice.isVisible())
        self.assertIn("last (Out) frame", tab.sam31_anchor_notice.text())
        self.assertIn("track backward", tab.sam31_anchor_notice.text())
        tab.close()


if __name__ == "__main__":
    unittest.main()
