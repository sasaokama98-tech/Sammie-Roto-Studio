import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from sammie import sammie
from sammie_main import MainWindow, SegmentationTab


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

    def test_prompt_controls_are_only_visible_for_sam31(self):
        tab = SegmentationTab()
        tab.show()

        tab.sam_model_combo.setCurrentText("Base")
        self.app.processEvents()
        self.assertFalse(tab.sam31_prompt_group.isVisible())

        tab.sam_model_combo.setCurrentText("SAM 3.1")
        self.app.processEvents()
        self.assertTrue(tab.sam31_prompt_group.isVisible())
        self.assertIn("non-destructive", tab.sam31_prompt_edit.toolTip().lower())
        tab.close()

    def test_candidate_list_supports_multiple_selection(self):
        tab = SegmentationTab()
        tab.set_prompt_candidates(
            [
                {"candidate_id": 3, "score": 0.8, "area": 20, "mask": np.ones((2, 2))},
                {"candidate_id": 7, "score": 0.9, "area": 30, "mask": np.ones((2, 2))},
            ]
        )
        tab.sam31_prompt_candidates.item(1).setSelected(True)

        selected = tab.selected_prompt_candidates()

        self.assertEqual([item["candidate_id"] for item in selected], [3, 7])
        tab.close()

    def test_prompt_seed_point_is_inside_candidate(self):
        mask = np.zeros((9, 11), dtype=np.uint8)
        mask[2:8, 3:10] = 1

        x, y = MainWindow._prompt_seed_point(mask)

        self.assertEqual(mask[y, x], 1)

    def test_boolean_prompt_mask_is_normalized_before_opencv_display(self):
        image = np.zeros((9, 11, 3), dtype=np.uint8)
        preview_mask = np.zeros((9, 11), dtype=bool)
        preview_mask[2:8, 3:10] = True

        def assert_uint8(mask):
            self.assertEqual(mask.dtype, np.uint8)
            self.assertEqual(set(np.unique(mask)), {0, 255})
            return mask

        with patch.object(
            sammie.core, "load_masks_for_frame", return_value={}
        ), patch.object(
            sammie.core,
            "apply_mask_postprocessing",
            side_effect=assert_uint8,
        ):
            result = sammie.apply_postprocessing_to_display(
                image,
                frame_number=0,
                points=[],
                view_options={"show_masks": False, "show_outlines": True},
                preview_mask=preview_mask,
                preview_object_id=0,
            )

        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.any(result != image))


if __name__ == "__main__":
    unittest.main()
