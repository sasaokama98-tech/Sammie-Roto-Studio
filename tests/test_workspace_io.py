import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from sammie import core, sammie
from sammie.workspace_io import (
    create_load_workspace,
    finalize_workspace_swap,
    rollback_workspace_swap,
    swap_workspace,
    validate_frame_workspace,
    workspace_paths,
)
from sammie_main import MainWindow


class WorkspaceIoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _write_frame(path, value=0, size=(6, 4)):
        frame = np.full((size[1], size[0], 3), value, dtype=np.uint8)
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(path)

    def test_validation_requires_exact_readable_frame_range(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = create_load_workspace(os.path.join(directory, "temp"))
            frames = Path(workspace_paths(workspace)["frames"])
            self._write_frame(frames / "00000.png", 10)
            self._write_frame(frames / "00001.png", 20)

            result = validate_frame_workspace(workspace, 2)

        self.assertEqual(result.frame_count, 2)
        self.assertEqual((result.width, result.height), (6, 4))
        self.assertEqual(result.frame_format, "png")

    def test_validation_rejects_missing_or_mismatched_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = create_load_workspace(os.path.join(directory, "temp"))
            frames = Path(workspace_paths(workspace)["frames"])
            self._write_frame(frames / "00000.png")
            self._write_frame(frames / "00002.png")

            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                validate_frame_workspace(workspace, 3)

            (frames / "00002.png").unlink()
            self._write_frame(frames / "00001.png", size=(8, 4))
            with self.assertRaisesRegex(RuntimeError, "expected 6x4"):
                validate_frame_workspace(workspace, 2)

    def test_workspace_swap_can_be_rolled_back_or_finalized(self):
        with tempfile.TemporaryDirectory() as directory:
            live = os.path.join(directory, "temp")
            os.makedirs(live)
            Path(live, "old.txt").write_text("old", encoding="utf-8")

            staging = create_load_workspace(live)
            Path(staging, "new.txt").write_text("new", encoding="utf-8")
            backup = swap_workspace(staging, live)
            self.assertTrue(Path(live, "new.txt").exists())

            rollback_workspace_swap(live, backup)
            self.assertTrue(Path(live, "old.txt").exists())

            staging = create_load_workspace(live)
            Path(staging, "new.txt").write_text("new", encoding="utf-8")
            backup = swap_workspace(staging, live)
            finalize_workspace_swap(backup)
            self.assertTrue(Path(live, "new.txt").exists())
            self.assertFalse(backup and os.path.exists(backup))

    def test_failed_main_window_load_preserves_live_session_and_points(self):
        class StubSettings:
            def __init__(self):
                self.session_settings = SimpleNamespace(name="old")

            def create_new_session(self, _path):
                self.session_settings = SimpleNamespace(name="new")

        fake_window = SimpleNamespace(settings_mgr=StubSettings())
        old_video = (640, 360, 24.0, 12, 1)

        with tempfile.TemporaryDirectory() as directory:
            live = os.path.join(directory, "temp")
            os.makedirs(live)
            sentinel = Path(live, "existing-session.txt")
            sentinel.write_text("keep", encoding="utf-8")

            with patch.object(core, "temp_dir", live), patch.object(
                core.VideoInfo, "width", old_video[0]
            ), patch.object(core.VideoInfo, "height", old_video[1]), patch.object(
                core.VideoInfo, "fps", old_video[2]
            ), patch.object(
                core.VideoInfo, "total_frames", old_video[3]
            ), patch.object(
                core.VideoInfo, "color_space", old_video[4]
            ), patch(
                "sammie_main.sammie.load_image_sequence",
                side_effect=RuntimeError("bad frame"),
            ):
                result = MainWindow.load_file(fake_window, "broken.1001.jpg")

                self.assertFalse(result)
                self.assertEqual(fake_window.settings_mgr.session_settings.name, "old")
                self.assertEqual(core.VideoInfo.total_frames, old_video[3])
                self.assertTrue(sentinel.exists())
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_successful_main_window_load_swaps_only_after_validation(self):
        class StubSettings:
            def __init__(self):
                self.session_settings = SimpleNamespace(name="old")

            def create_new_session(self, path):
                self.session_settings = SimpleNamespace(
                    name="new", video_file_path=path
                )

            def set_session_setting(self, key, value):
                setattr(self.session_settings, key, value)
                return True

            def update_video_info(
                self, width, height, fps, count, color_space, path
            ):
                self.session_settings.video_width = width
                self.session_settings.video_height = height
                self.session_settings.video_fps = fps
                self.session_settings.total_frames = count
                self.session_settings.color_space = color_space
                self.session_settings.video_file_path = path

            @staticmethod
            def save_session_settings():
                Path(core.temp_dir, "session_settings.conf").write_text(
                    "new", encoding="utf-8"
                )
                return True

        point_manager = SimpleNamespace(clear_all=Mock(), points=[])
        fake_window = SimpleNamespace(
            settings_mgr=StubSettings(),
            point_manager=point_manager,
            sam_manager=SimpleNamespace(
                propagated=True, initialize_predictor=Mock()
            ),
            matany_manager=SimpleNamespace(propagated=True),
            removal_manager=SimpleNamespace(propagated=True),
            frame_slider=SimpleNamespace(setRange=Mock(), setValue=Mock()),
            viewer=SimpleNamespace(clear_image=Mock()),
            sidebar=SimpleNamespace(
                load_values_from_settings=Mock(),
                tab_widget=SimpleNamespace(setCurrentIndex=Mock()),
                segmentation_tab=SimpleNamespace(
                    sam_model_btn=SimpleNamespace(setEnabled=Mock())
                ),
            ),
            _refresh_frame_display_controls=Mock(),
            _reset_show_all_points_button_state=Mock(),
            clear_markers=Mock(),
            _update_dynamic_widgets=Mock(),
            get_view_options=Mock(return_value={"view_mode": "None"}),
        )

        def stage_one_frame(_path, parent_window, sequence_files, workspace_dir):
            frame_path = Path(workspace_paths(workspace_dir)["frames"], "00000.png")
            self._write_frame(frame_path, 42)
            core.VideoInfo.width = 6
            core.VideoInfo.height = 4
            core.VideoInfo.fps = 24.0
            core.VideoInfo.total_frames = 1
            core.VideoInfo.color_space = 1
            return 1

        with tempfile.TemporaryDirectory() as directory:
            live = os.path.join(directory, "temp")
            os.makedirs(live)
            Path(live, "old.txt").write_text("old", encoding="utf-8")

            with patch.object(core, "temp_dir", live), patch(
                "sammie_main.sammie.load_image_sequence",
                side_effect=stage_one_frame,
            ), patch(
                "sammie_main.sammie.update_image", return_value=None
            ):
                result = MainWindow.load_file(fake_window, "shot.1001.png")

            self.assertTrue(result)
            self.assertFalse(Path(live, "old.txt").exists())
            self.assertTrue(Path(live, "frames", "00000.png").exists())
            self.assertTrue(Path(live, "session_settings.conf").exists())
            point_manager.clear_all.assert_called_once()

    def test_image_sequence_loader_stages_every_frame_without_skipping(self):
        class StubSettings:
            def __init__(self):
                self.values = {}

            @staticmethod
            def get_app_setting(_key, default=None):
                return default

            def set_session_setting(self, key, value):
                self.values[key] = value
                return True

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "source")
            source.mkdir()
            first = source / "shot.1001.jpg"
            second = source / "shot.1002.jpg"
            self._write_frame(first, 10)
            self._write_frame(second, 20)
            workspace = create_load_workspace(os.path.join(directory, "temp"))

            with patch(
                "sammie.sammie.get_settings_manager",
                return_value=StubSettings(),
            ):
                count = sammie.load_image_sequence(
                    str(first),
                    parent_window=None,
                    sequence_files=[str(first), str(second)],
                    workspace_dir=workspace,
                )

            validated = validate_frame_workspace(workspace, count)

        self.assertEqual(count, 2)
        self.assertEqual(validated.frame_count, 2)
        self.assertEqual(validated.frame_format, "jpg")

    def test_image_sequence_loader_rejects_unreadable_middle_frame(self):
        class StubSettings:
            @staticmethod
            def get_app_setting(_key, default=None):
                return default

            @staticmethod
            def set_session_setting(_key, _value):
                return True

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, "shot.1001.jpg")
            broken = Path(directory, "shot.1002.jpg")
            self._write_frame(first, 10)
            broken.write_bytes(b"not an image")
            workspace = create_load_workspace(os.path.join(directory, "temp"))

            with patch(
                "sammie.sammie.get_settings_manager",
                return_value=StubSettings(),
            ), self.assertRaisesRegex(RuntimeError, "Could not read image frame"):
                sammie.load_image_sequence(
                    str(first),
                    parent_window=None,
                    sequence_files=[str(first), str(broken)],
                    workspace_dir=workspace,
                )


if __name__ == "__main__":
    unittest.main()
