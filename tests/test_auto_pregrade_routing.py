import unittest
from unittest.mock import MagicMock, patch

from sammie.matting import MattingManager
from sammie.sammie import SamManager


class AutoPregradeRoutingTests(unittest.TestCase):
    def test_sam2_initializes_on_inference_directory(self):
        manager = SamManager()
        manager.predictor = MagicMock()
        manager.predictor.init_state.return_value = {"ready": True}
        with patch.object(manager, "_segmentation_inference_dir", return_value="graded"):
            manager.initialize_predictor()
        self.assertEqual(manager.inference_state, {"ready": True})
        self.assertEqual(
            manager.predictor.init_state.call_args.kwargs["video_path"], "graded"
        )

    def test_sam31_initializes_on_inference_directory(self):
        manager = SamManager()
        manager.sam31_backend = MagicMock()
        manager.sam31_backend.session_id = "test-session"
        with patch.object(manager, "_segmentation_inference_dir", return_value="graded"):
            manager.initialize_predictor()
        manager.sam31_backend.start_session.assert_called_once_with("graded")
        self.assertEqual(manager.inference_state, {"session_id": "test-session"})

    def test_matting_frame_paths_use_shared_inference_directory(self):
        manager = MattingManager()
        with patch.object(manager, "_inference_frames_dir", return_value="graded"):
            self.assertEqual(
                manager._inference_frame_path(12, "png"), "graded\\00012.png"
            )
            self.assertEqual(manager._collect_image_paths(0, 0), [])


if __name__ == "__main__":
    unittest.main()
