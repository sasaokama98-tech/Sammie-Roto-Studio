import unittest

import numpy as np

from sammie.sam31_backend import Sam31Backend


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


class Sam31BackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = Sam31Backend(_Device(), "png")
        self.backend.predictor = _FakePredictor()
        self.backend.session_id = "session"

    def test_squeezes_singleton_mask_channel(self):
        outputs = {
            "out_obj_ids": np.array([4]),
            "out_binary_masks": np.ones((1, 1, 4, 5), dtype=bool),
        }

        masks = Sam31Backend.masks_from_outputs(outputs)

        self.assertEqual(masks[0][0], 4)
        self.assertEqual(masks[0][1].shape, (4, 5))


    def test_add_points_uses_absolute_coordinates(self):
        masks = self.backend.add_points(
            frame_number=2,
            object_id=7,
            points=[[10, 20]],
            labels=[1],
        )

        request = self.backend.predictor.requests[-1]
        self.assertEqual(request["type"], "add_prompt")
        self.assertFalse(request["rel_coordinates"])
        self.assertEqual(request["obj_id"], 7)
        self.assertEqual(masks[0][0], 7)
        self.assertEqual(masks[0][1].shape, (4, 5))

    def test_propagation_direction_is_mapped(self):
        frames = list(self.backend.propagate(2, 4, reverse=True))

        request = self.backend.predictor.requests[-1]
        self.assertEqual(request["propagation_direction"], "backward")
        self.assertEqual(request["start_frame_index"], 2)
        self.assertEqual(request["max_frame_num_to_track"], 4)
        self.assertEqual(frames[0][0], 3)
        self.assertEqual(frames[0][1][0][0], 7)

    def test_reset_and_remove_use_session_api(self):
        self.backend.reset()
        self.backend.remove_object(9, frame_number=4)

        self.assertEqual(self.backend.predictor.requests[-2]["type"], "reset_session")
        self.assertEqual(self.backend.predictor.requests[-1]["type"], "remove_object")
        self.assertEqual(self.backend.predictor.requests[-1]["obj_id"], 9)


if __name__ == "__main__":
    unittest.main()
