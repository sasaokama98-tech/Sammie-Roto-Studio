import unittest

import numpy as np

from sammie.vitmatte_backend import (
    Roi,
    VitMatteBackend,
    integrate_unknown_alpha,
    integrate_unknown_alpha_float,
    split_roi,
    unknown_roi,
    unknown_rois,
)


class VitMatteRoiTests(unittest.TestCase):
    def test_unknown_roi_is_padded_and_clamped(self):
        trimap = np.zeros((20, 30), dtype=np.uint8)
        trimap[3:8, 5:10] = 128
        self.assertEqual(unknown_roi(trimap, margin=4), Roi(1, 0, 14, 12))

    def test_no_unknown_region_has_no_roi(self):
        self.assertIsNone(unknown_roi(np.zeros((8, 8), dtype=np.uint8)))

    def test_merge_preserves_known_regions(self):
        trimap = np.zeros((8, 8), dtype=np.uint8)
        trimap[2:6, 2:6] = 128
        trimap[3:5, 3:5] = 255
        predicted = np.full((8, 8), 0.25, dtype=np.float32)
        alpha = integrate_unknown_alpha(trimap, predicted)
        self.assertEqual(alpha[0, 0], 0)
        self.assertEqual(alpha[3, 3], 255)
        self.assertEqual(alpha[2, 2], 64)

    def test_roi_merge_updates_only_unknown_crop(self):
        trimap = np.zeros((10, 10), dtype=np.uint8)
        trimap[3:7, 3:7] = 128
        trimap[4:6, 4:6] = 255
        roi = Roi(2, 2, 8, 8)
        alpha = integrate_unknown_alpha(
            trimap, np.full((6, 6), 200, dtype=np.uint8), roi
        )
        self.assertEqual(alpha[3, 3], 200)
        self.assertEqual(alpha[4, 4], 255)
        self.assertEqual(alpha[0, 0], 0)

    def test_disconnected_unknown_regions_stay_separate(self):
        trimap = np.zeros((100, 160), dtype=np.uint8)
        trimap[10:20, 10:20] = 128
        trimap[70:80, 130:140] = 128
        self.assertEqual(
            unknown_rois(trimap, margin=5),
            [Roi(5, 5, 25, 25), Roi(125, 65, 145, 85)],
        )

    def test_large_roi_is_fully_covered_by_tiles(self):
        roi = Roi(0, 0, 2200, 1300)
        tiles = split_roi(roi, max_tile_size=1024, overlap=128)
        coverage = np.zeros((1300, 2200), dtype=np.uint8)
        for tile in tiles:
            coverage[tile.slices] = 1
            self.assertLessEqual(tile.x1 - tile.x0, 1024)
            self.assertLessEqual(tile.y1 - tile.y0, 1024)
        self.assertTrue(np.all(coverage))

    def test_multi_roi_feathers_predictions_and_preserves_known_pixels(self):
        class FakeBackend(VitMatteBackend):
            def __init__(self):
                pass

            def predict(self, rgb, trimap):
                return np.full(trimap.shape, 0.5, dtype=np.float32)

        trimap = np.zeros((64, 96), dtype=np.uint8)
        trimap[8:56, 10:86] = 128
        trimap[20:44, 30:66] = 255
        rgb = np.zeros((64, 96, 3), dtype=np.uint8)
        alpha = FakeBackend().predict_multi_roi(
            rgb, trimap, margin=4, max_tile_size=32, tile_overlap=8
        )
        self.assertEqual(alpha[0, 0], 0)
        self.assertEqual(alpha[32, 48], 255)
        self.assertIn(alpha[10, 12], (127, 128))

    def test_float_merge_retains_sub_8_bit_precision(self):
        trimap = np.full((4, 4), 128, dtype=np.uint8)
        prediction = np.full((4, 4), 0.5001, dtype=np.float32)
        alpha = integrate_unknown_alpha_float(trimap, prediction)
        self.assertEqual(alpha.dtype, np.float32)
        self.assertAlmostEqual(float(alpha[0, 0]), 0.5001, places=5)


if __name__ == "__main__":
    unittest.main()
