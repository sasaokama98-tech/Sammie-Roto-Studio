import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPixmap
from PySide6.QtWidgets import QApplication

from sammie.gui_widgets import ImageViewer


class ImageViewerNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.viewer = ImageViewer()
        self.viewer.resize(320, 240)
        self.viewer.show()
        self.viewer.load_image_reset_zoom(QPixmap(640, 480))
        self.viewer.set_zoom(1.0)
        self.app.processEvents()
        self.points = []
        self.deletions = []
        self.viewer.point_clicked.connect(lambda *point: self.points.append(point))
        self.viewer.point_delete_requested.connect(
            lambda *position: self.deletions.append(position)
        )

    def tearDown(self):
        self.viewer.close()

    def _event(self, event_type, position, button, buttons, modifiers=Qt.NoModifier):
        position = QPointF(position)
        return QMouseEvent(
            event_type,
            position,
            self.viewer.viewport().mapToGlobal(position.toPoint()),
            button,
            buttons,
            modifiers,
        )

    def test_plain_point_clicks_are_unchanged(self):
        center = QPointF(160, 120)
        self.viewer.mousePressEvent(
            self._event(
                QEvent.MouseButtonPress,
                center,
                Qt.LeftButton,
                Qt.LeftButton,
            )
        )
        self.viewer.mousePressEvent(
            self._event(
                QEvent.MouseButtonPress,
                center,
                Qt.RightButton,
                Qt.RightButton,
            )
        )
        self.assertEqual([point[2] for point in self.points], [True, False])

    def test_alt_right_is_unassigned(self):
        start = QPointF(120, 100)
        initial_scale = self.viewer.current_scale
        self.viewer.mousePressEvent(
            self._event(
                QEvent.MouseButtonPress,
                start,
                Qt.RightButton,
                Qt.RightButton,
                Qt.AltModifier,
            )
        )
        self.viewer.mouseMoveEvent(
            self._event(
                QEvent.MouseMove,
                QPointF(220, 100),
                Qt.NoButton,
                Qt.RightButton,
                Qt.AltModifier,
            )
        )
        self.viewer.mouseReleaseEvent(
            self._event(
                QEvent.MouseButtonRelease,
                QPointF(220, 100),
                Qt.RightButton,
                Qt.NoButton,
                Qt.AltModifier,
            )
        )
        self.assertEqual(self.viewer.current_scale, initial_scale)
        self.assertEqual(self.points, [])
        self.assertEqual(self.deletions, [])
        self.assertFalse(self.viewer._is_zooming)

    def test_ctrl_left_and_right_request_point_deletion(self):
        center = QPointF(160, 120)
        for button in (Qt.LeftButton, Qt.RightButton):
            self.viewer.mousePressEvent(
                self._event(
                    QEvent.MouseButtonPress,
                    center,
                    button,
                    button,
                    Qt.ControlModifier,
                )
            )

        self.assertEqual(len(self.deletions), 2)
        self.assertEqual(self.points, [])

    def test_alt_left_and_middle_start_pan_without_points(self):
        start = QPointF(120, 100)
        alt_left = self._event(
            QEvent.MouseButtonPress,
            start,
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.AltModifier,
        )
        self.viewer.mousePressEvent(alt_left)
        self.assertTrue(self.viewer._is_panning)
        self.assertEqual(self.points, [])
        self.viewer.mouseReleaseEvent(
            self._event(
                QEvent.MouseButtonRelease,
                start,
                Qt.LeftButton,
                Qt.NoButton,
                Qt.AltModifier,
            )
        )

        middle = self._event(
            QEvent.MouseButtonPress,
            start,
            Qt.MiddleButton,
            Qt.MiddleButton,
        )
        self.viewer.mousePressEvent(middle)
        self.assertTrue(self.viewer._is_panning)
        self.viewer.mouseMoveEvent(
            self._event(
                QEvent.MouseMove,
                QPointF(90, 80),
                Qt.NoButton,
                Qt.MiddleButton,
            )
        )
        self.viewer.mouseReleaseEvent(
            self._event(
                QEvent.MouseButtonRelease,
                start,
                Qt.MiddleButton,
                Qt.NoButton,
            )
        )
        self.assertFalse(self.viewer._is_panning)

    def test_alt_middle_drag_remains_zoom(self):
        start = QPointF(120, 100)
        initial_scale = self.viewer.current_scale
        self.viewer.mousePressEvent(
            self._event(
                QEvent.MouseButtonPress,
                start,
                Qt.MiddleButton,
                Qt.MiddleButton,
                Qt.AltModifier,
            )
        )
        self.viewer.mouseMoveEvent(
            self._event(
                QEvent.MouseMove,
                QPointF(220, 100),
                Qt.NoButton,
                Qt.MiddleButton,
                Qt.AltModifier,
            )
        )
        self.viewer.mouseReleaseEvent(
            self._event(
                QEvent.MouseButtonRelease,
                QPointF(220, 100),
                Qt.MiddleButton,
                Qt.NoButton,
                Qt.AltModifier,
            )
        )
        self.assertGreater(self.viewer.current_scale, initial_scale)
        self.assertEqual(self.points, [])
        self.assertEqual(self.deletions, [])

    def test_zoom_anchor_stays_on_the_same_image_position(self):
        anchor = QPointF(80, 70)
        scene_before = self.viewer.mapToScene(anchor.toPoint())
        self.viewer.set_zoom(self.viewer.current_scale * 1.5, anchor)
        scene_after = self.viewer.mapToScene(anchor.toPoint())
        self.assertAlmostEqual(scene_before.x(), scene_after.x(), delta=2.0)
        self.assertAlmostEqual(scene_before.y(), scene_after.y(), delta=2.0)


if __name__ == "__main__":
    unittest.main()
