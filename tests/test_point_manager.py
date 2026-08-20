import unittest
from unittest.mock import Mock, patch

from sammie.core import PointManager


class PointManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = PointManager()
        self.manager.points = [
            {"frame": 0, "object_id": 1, "positive": True, "x": 10, "y": 10},
            {"frame": 0, "object_id": 2, "positive": False, "x": 14, "y": 10},
            {"frame": 1, "object_id": 1, "positive": True, "x": 10, "y": 10},
        ]

    @patch("sammie.core.get_settings_manager")
    def test_removes_nearest_point_on_current_frame(self, settings_factory):
        settings_factory.return_value = Mock()

        removed = self.manager.remove_nearest_point(
            frame=0,
            x=13,
            y=10,
            max_distance=5,
            preferred_object_id=1,
        )

        self.assertEqual(removed["object_id"], 2)
        self.assertEqual(len(self.manager.points), 2)

    @patch("sammie.core.get_settings_manager")
    def test_preferred_object_breaks_equal_distance_tie(self, settings_factory):
        settings_factory.return_value = Mock()

        removed = self.manager.remove_nearest_point(
            frame=0,
            x=12,
            y=10,
            max_distance=5,
            preferred_object_id=1,
        )

        self.assertEqual(removed["object_id"], 1)

    @patch("sammie.core.get_settings_manager")
    def test_click_outside_hit_radius_does_nothing(self, settings_factory):
        settings_factory.return_value = Mock()

        removed = self.manager.remove_nearest_point(
            frame=0,
            x=100,
            y=100,
            max_distance=5,
            preferred_object_id=1,
        )

        self.assertIsNone(removed)
        self.assertEqual(len(self.manager.points), 3)


if __name__ == "__main__":
    unittest.main()
