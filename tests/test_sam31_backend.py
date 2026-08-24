import os
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sammie.sam31_backend import Sam31Backend, _CheckpointKeyOutputFilter
from sammie.sammie import SamManager


class _Device:
    type = "cuda"


class _FakePredictor:
    def __init__(self):
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "add_prompt":
            return {
                "frame_index": request["frame_index"],
                "outputs": {
                    "out_obj_ids": np.array([request["obj_id"]]),
                    "out_binary_masks": np.ones((1, 4, 5), dtype=bool),
                },
            }
        return {"is_success": True}

    def handle_stream_request(self, request):
        self.requests.append(request)
        yield {
            "frame_index": 3,
            "outputs": {
                "out_obj_ids": np.array([7]),
                "out_binary_masks": np.ones((1, 4, 5), dtype=bool),
            },
        }


class _MultiplexModelWithoutStateOffload:
    def __init__(self):
        self.init_kwargs = None

    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
    ):
        self.init_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "async_loading_frames": async_loading_frames,
        }
        return {"frames": []}


class _PredictorWithIncompatibleBaseSession:
    def __init__(self):
        self.model = _MultiplexModelWithoutStateOffload()

    def handle_request(self, request):
        self.model.init_state(
            resource_path=request["resource_path"],
            offload_video_to_cpu=request.get("offload_video_to_cpu", False),
            offload_state_to_cpu=False,
            async_loading_frames=True,
        )
        return {"session_id": "compatible-session"}


class Sam31BackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = Sam31Backend(_Device(), "png")
        self.backend.predictor = _FakePredictor()
        self.backend.session_id = "session"
        self.backend.frame_size = (1920, 1080)

    def test_squeezes_singleton_mask_channel(self):
        outputs = {
            "out_obj_ids": np.array([4]),
            "out_binary_masks": np.ones((1, 1, 4, 5), dtype=bool),
        }

        masks = Sam31Backend.masks_from_outputs(outputs)

        self.assertEqual(masks[0][0], 4)
        self.assertEqual(masks[0][1].shape, (4, 5))

    def test_explicit_checkpoint_path_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam3.1_multiplex.pt"
            checkpoint.touch()
            backend = Sam31Backend(_Device(), "png", checkpoint)
            self.assertEqual(backend._resolve_checkpoint_path(), checkpoint.resolve())

    def test_environment_checkpoint_path_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sam3.1_multiplex.pt"
            checkpoint.touch()
            with patch.dict(os.environ, {"SAM31_CHECKPOINT_PATH": str(checkpoint)}):
                backend = Sam31Backend(_Device(), "png")
                self.assertEqual(
                    backend._resolve_checkpoint_path(), checkpoint.resolve()
                )

    def test_missing_configured_checkpoint_has_actionable_error(self):
        backend = Sam31Backend(_Device(), "png", "missing-sam31.pt")
        with self.assertRaisesRegex(Exception, "checkpoint was not found"):
            backend._resolve_checkpoint_path()

    def test_checkpoint_key_filter_compacts_verbose_non_strict_lists(self):
        target = io.StringIO()
        output = _CheckpointKeyOutputFilter(target)
        output.write("ordinary builder message\n")
        output.write("Missing keys: ['layer.a', 'layer.b', 'layer.c']\n")
        output.write("Unexpected keys (12): ['tracker.a', 'tracker.b']...\n")
        output.finish()

        self.assertEqual(target.getvalue(), "ordinary builder message\n")
        self.assertEqual(output.missing_count, 3)
        self.assertEqual(output.unexpected_count, 12)
        self.assertIn("2 non-strict key lists suppressed", output.summary())
        self.assertNotIn("tracker.a", output.summary())

    def test_add_points_normalizes_non_square_source_coordinates(self):
        masks = self.backend.add_points(
            frame_number=2,
            object_id=7,
            points=[[960, 270]],
            labels=[1],
        )

        request = self.backend.predictor.requests[-1]
        self.assertEqual(request["type"], "add_prompt")
        self.assertTrue(request["rel_coordinates"])
        self.assertTrue(
            np.allclose(request["points"], np.asarray([[0.5, 0.25]]))
        )
        self.assertEqual(request["obj_id"], 7)
        self.assertEqual(masks[0][0], 7)
        self.assertEqual(masks[0][1].shape, (4, 5))

    def test_add_points_rejects_coordinates_outside_source_frame(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            self.backend.add_points(
                frame_number=2,
                object_id=7,
                points=[[1921, 270]],
                labels=[1],
            )

    def test_propagation_direction_is_mapped(self):
        frames = list(self.backend.propagate(2, 4, reverse=True))

        request = self.backend.predictor.requests[-1]
        self.assertEqual(request["propagation_direction"], "backward")
        self.assertEqual(request["start_frame_index"], 2)
        self.assertEqual(request["max_frame_num_to_track"], 4)
        self.assertEqual(frames[0][0], 3)
        self.assertEqual(frames[0][1][0][0], 7)

    def test_first_frame_anchor_selects_forward_tracking(self):
        plan = SamManager._sam31_anchor_plan(
            [{"frame": 10, "object_id": 0}], 10, 20
        )

        self.assertEqual(plan, (10, 10, False))

    def test_last_frame_anchor_selects_backward_tracking(self):
        plan = SamManager._sam31_anchor_plan(
            [{"frame": 20, "object_id": 0}], 10, 20
        )

        self.assertEqual(plan, (20, 10, True))

    def test_middle_frame_anchor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only supports point anchors"):
            SamManager._sam31_anchor_plan(
                [{"frame": 15, "object_id": 0}], 10, 20
            )

    def test_both_endpoint_anchors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one anchor edge at a time"):
            SamManager._sam31_anchor_plan(
                [
                    {"frame": 10, "object_id": 0},
                    {"frame": 20, "object_id": 0},
                ],
                10,
                20,
            )

    def test_anchor_only_output_is_not_valid_propagation_coverage(self):
        self.assertFalse(
            SamManager._has_sam31_propagation_coverage({10}, 10, 11)
        )

    def test_non_anchor_output_is_valid_propagation_coverage(self):
        self.assertTrue(
            SamManager._has_sam31_propagation_coverage({10, 11}, 10, 11)
        )

    def test_reset_and_remove_use_session_api(self):
        self.backend.reset()
        self.backend.remove_object(9, frame_number=4)

        self.assertEqual(self.backend.predictor.requests[-2]["type"], "reset_session")
        self.assertEqual(self.backend.predictor.requests[-1]["type"], "remove_object")
        self.assertEqual(self.backend.predictor.requests[-1]["obj_id"], 9)

    def test_session_filters_unsupported_state_offload_argument(self):
        predictor = _PredictorWithIncompatibleBaseSession()
        self.backend.predictor = predictor

        response = self.backend._handle_start_session(
            {
                "type": "start_session",
                "resource_path": "frames",
                "offload_video_to_cpu": True,
            }
        )

        self.assertEqual(response["session_id"], "compatible-session")
        self.assertEqual(predictor.model.init_kwargs["resource_path"], "frames")
        self.assertTrue(predictor.model.init_kwargs["offload_video_to_cpu"])
        self.assertTrue(predictor.model.init_kwargs["async_loading_frames"])


if __name__ == "__main__":
    unittest.main()
