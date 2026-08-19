import unittest

import numpy as np

from sammie.trimap import TrimapConfig, generate_trimap, render_trimap_preview


class TrimapTests(unittest.TestCase):
    def test_generates_three_class_trimap(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255

        trimap = generate_trimap(
            mask,
            TrimapConfig(erode_width=2, dilate_width=2),
        )

        self.assertEqual(trimap.dtype, np.uint8)
        self.assertEqual(set(np.unique(trimap)), {0, 128, 255})
        self.assertEqual(trimap[16, 16], 255)
        self.assertEqual(trimap[0, 0], 0)
        self.assertEqual(trimap[7, 16], 128)

    def test_accepts_normalized_float_masks(self):
        mask = np.zeros((16, 16), dtype=np.float32)
        mask[4:12, 4:12] = 1.0

        trimap = generate_trimap(
            mask,
            TrimapConfig(erode_width=1, dilate_width=1),
        )

        self.assertEqual(trimap[8, 8], 255)
        self.assertEqual(trimap[3, 8], 128)

    def test_auto_width_is_resolution_aware_and_bounded(self):
        config = TrimapConfig(auto_scale=0.01, min_width=2, max_width=12)
        self.assertEqual(config.resolve_widths((100, 200)), (2, 2))
        self.assertEqual(config.resolve_widths((800, 1600)), (8, 8))
        self.assertEqual(config.resolve_widths((4000, 8000)), (12, 12))

    def test_rejects_negative_manual_width(self):
        with self.assertRaises(ValueError):
            TrimapConfig(erode_width=-1).resolve_widths((100, 100))

    def test_preview_distinguishes_all_three_classes(self):
        image = np.full((1, 3, 3), 100, dtype=np.uint8)
        trimap = np.array([[0, 128, 255]], dtype=np.uint8)
        preview = render_trimap_preview(image, trimap)
        self.assertEqual(len({tuple(pixel) for pixel in preview[0]}), 3)


if __name__ == "__main__":
    unittest.main()
