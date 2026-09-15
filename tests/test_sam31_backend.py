import os
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

from sammie import core
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


class _TrackerFlags:
    multimask_output_in_sam = True


class _SinglePointEmptyPredictor:
    def __init__(
        self,
        recover_with_single_mask=True,
        recover_with_adjacent_point=True,
    ):
        self.requests = []
        self.model = type("Model", (), {"tracker": _TrackerFlags()})()
        self.recover_with_single_mask = recover_with_single_mask
        self.recover_with_adjacent_point = recover_with_adjacent_point

    def handle_request(self, request):
        self.requests.append(request)
        points = np.asarray(request["points"])
        use_multimask = self.model.tracker.multimask_output_in_sam
        succeeds = (len(points) > 1 and self.recover_with_adjacent_point) or (
            not use_multimask and self.recover_with_single_mask
        )
        masks = (
            np.ones((1, 4, 5), dtype=bool)
            if succeeds
            else np.zeros((0, 4, 5), dtype=bool)
        )
        object_ids = (
            np.asarray([request["obj_id"]], dtype=np.int64)
            if succeeds
            else np.zeros(0, dtype=np.int64)
        )
        return {
            "frame_index": request["frame_index"],
            "outputs": {
                "out_obj_ids": object_ids,
                "out_binary_masks": masks,
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


class _PromptPredictor:
    def __init__(self):
        self.requests = []
        self.session_count = 0

    def handle_request(self, request):
        self.requests.append(request)
        request_type = request["type"]
        if request_type == "start_session":
            self.session_count += 1
            return {"session_id": f"preview-{self.session_count}"}
        if request_type == "add_prompt" and request.get("text"):
            masks = np.zeros((2, 4, 5), dtype=bool)
            masks[0, 0:2, 0:2] = True
            masks[1, 1:4, 2:5] = True
            return {
                "frame_index": request["frame_index"],
                "outputs": {
                    "out_obj_ids": np.array([3, 7]),
                    "out_binary_masks": masks,
                    "out_probs": np.array([0.75, 0.95], dtype=np.float32),
                    "out_boxes_xywh": np.array(
                        [[0.0, 0.0, 0.4, 0.5], [0.4, 0.25, 0.6, 0.75]],
                        dtype=np.float32,
                    ),
                },
            }
        if request_type == "add_prompt":
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
            "frame_index": 6,
            "outputs": {
                "out_obj_ids": np.array([7]),
                "out_binary_masks": np.ones((1, 4, 5), dtype=bool),
            },
        }


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

    def test_single_positive_retries_empty_multimask_as_single_mask(self):
        predictor = _SinglePointEmptyPredictor(recover_with_single_mask=True)
        self.backend.predictor = predictor

        masks = self.backend.add_points(2, 7, [[960, 270]], [1])

        self.assertEqual(len(predictor.requests), 2)
        self.assertTrue(predictor.model.tracker.multimask_output_in_sam)
        self.assertTrue(masks[0][1].any())

    def test_single_positive_uses_adjacent_seed_if_single_mask_is_empty(self):
        predictor = _SinglePointEmptyPredictor(recover_with_single_mask=False)
        self.backend.predictor = predictor

        masks = self.backend.add_points(2, 7, [[960, 270]], [1])

        self.assertEqual(len(predictor.requests), 3)
        retry_points = np.asarray(predictor.requests[-1]["points"])
        self.assertEqual(retry_points.shape, (2, 2))
        self.assertTrue(masks[0][1].any())

    def test_single_negative_does_not_trigger_positive_recovery(self):
        predictor = _SinglePointEmptyPredictor(recover_with_single_mask=True)
        self.backend.predictor = predictor

        masks = self.backend.add_points(2, 7, [[960, 270]], [0])

        self.assertEqual(masks, [])
        self.assertEqual(len(predictor.requests), 1)

    def test_single_positive_failure_is_not_silent_after_recovery(self):
        predictor = _SinglePointEmptyPredictor(
            recover_with_single_mask=False,
            recover_with_adjacent_point=False,
        )
        self.backend.predictor = predictor

        with self.assertRaisesRegex(RuntimeError, "could not create a mask"):
            self.backend.add_points(2, 7, [[960, 270]], [1])

        self.assertEqual(len(predictor.requests), 3)

    def test_recovered_single_positive_writes_mask_and_trimap(self):
        predictor = _SinglePointEmptyPredictor(recover_with_single_mask=True)
        self.backend.predictor = predictor
        manager = SamManager()
        manager.sam31_backend = self.backend
        manager.predictor = predictor
        manager.inference_state = {"session_id": "session"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            masks = root / "masks"
            trimaps = root / "trimaps"
            frames.mkdir()
            cv2.imwrite(
                str(frames / "00000.png"),
                np.zeros((4, 5, 3), dtype=np.uint8),
            )
            with patch.object(core, "frames_dir", str(frames)), patch.object(
                core, "mask_dir", str(masks)
            ), patch.object(core, "trimap_dir", str(trimaps)), patch.object(
                core, "get_frame_extension", return_value="png"
            ):
                manager.segment_image(0, 7, [[960, 270]], [1])

            mask = cv2.imread(
                str(masks / "00000" / "7.png"), cv2.IMREAD_GRAYSCALE
            )
            trimap = cv2.imread(
                str(trimaps / "00000" / "7.png"), cv2.IMREAD_GRAYSCALE
            )

        self.assertIsNotNone(mask)
        self.assertTrue(mask.any())
        self.assertIsNotNone(trimap)

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

    def test_prompt_preview_is_isolated_until_candidate_commit(self):
        predictor = _PromptPredictor()
        self.backend.predictor = predictor
        self.backend.resource_path = "frames"

        candidates = self.backend.preview_text_prompt(5, "person")

        self.assertEqual(self.backend.session_id, "session")
        self.assertEqual(self.backend.prompt_preview_session_id, "preview-1")
        self.assertEqual([item["candidate_id"] for item in candidates], [3, 7])
        self.assertAlmostEqual(candidates[1]["score"], 0.95, places=5)

        masks = self.backend.commit_prompt_candidates([7], [2])

        self.assertEqual(self.backend.session_id, "preview-1")
        self.assertEqual(masks[0][0], 2)
        self.assertEqual(self.backend.prompt_seed_metadata()["text"], "person")
        self.assertEqual(
            self.backend.prompt_seed_metadata()["mappings"][0]["object_id"], 2
        )
        close_requests = [
            request for request in predictor.requests if request["type"] == "close_session"
        ]
        self.assertEqual(close_requests[-1]["session_id"], "session")
        removed = [
            request for request in predictor.requests if request["type"] == "remove_object"
        ]
        self.assertEqual(removed[-1]["obj_id"], 3)

    def test_committed_candidate_maps_point_refinement_and_propagation(self):
        predictor = _PromptPredictor()
        self.backend.predictor = predictor
        self.backend.resource_path = "frames"
        self.backend.preview_text_prompt(5, "person")
        self.backend.commit_prompt_candidates([7], [2])

        masks = self.backend.add_points(5, 2, [[960, 540]], [1])
        point_request = [
            request
            for request in predictor.requests
            if request["type"] == "add_prompt" and request.get("points") is not None
        ][-1]
        propagated = list(self.backend.propagate(5, 1, reverse=False))

        self.assertEqual(point_request["obj_id"], 7)
        self.assertEqual(masks[0][0], 2)
        self.assertEqual(propagated[0][1][0][0], 2)

    def test_saved_prompt_seed_is_replayed_before_point_refinement(self):
        predictor = _PromptPredictor()
        self.backend.predictor = predictor
        self.backend.configure_prompt_seed(
            {
                "text": "person",
                "frame": 5,
                "mappings": [
                    {
                        "candidate_index": 1,
                        "candidate_id": 7,
                        "object_id": 4,
                        "score": 0.95,
                    }
                ],
            }
        )

        masks = self.backend.add_points(5, 4, [[960, 540]], [1])
        prompt_requests = [
            request for request in predictor.requests if request["type"] == "add_prompt"
        ]

        self.assertEqual(prompt_requests[0]["text"], "person")
        self.assertEqual(prompt_requests[1]["obj_id"], 7)
        self.assertEqual(masks[0][0], 4)


if __name__ == "__main__":
    unittest.main()
