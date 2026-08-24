import unittest

import numpy as np

from sammie.motion_confidence import (
    calculate_bidirectional_alignment,
    forward_backward_confidence,
    motion_confidence_gate,
    warp_source_to_target,
)


class MotionConfidenceTests(unittest.TestCase):
    def test_forward_backward_cycle_marks_consistent_flow(self):
        forward = np.zeros((12, 16, 2), dtype=np.float32)
        backward = np.zeros_like(forward)
        forward[..., 0] = 1.0
        backward[..., 0] = -1.0

        confidence = forward_backward_confidence(
            forward, backward, cycle_sigma=0.5
        )

        self.assertGreater(float(confidence[:, :-1].mean()), 0.99)
        self.assertTrue(np.all(confidence[:, -1] == 0.0))

    def test_forward_backward_cycle_rejects_inconsistent_flow(self):
        forward = np.zeros((8, 10, 2), dtype=np.float32)
        backward = np.zeros_like(forward)
        forward[..., 0] = 1.0

        confidence = forward_backward_confidence(
            forward, backward, cycle_sigma=0.25
        )

        self.assertLess(float(confidence[:, :-1].max()), 1e-5)

    def test_warp_source_to_target_uses_target_to_source_flow(self):
        source = np.zeros((8, 10), dtype=np.float32)
        source[2:6, 2:4] = 1.0
        flow = np.zeros((8, 10, 2), dtype=np.float32)
        flow[..., 0] = -1.0
        confidence = np.ones((8, 10), dtype=np.float32)

        warped, warped_confidence = warp_source_to_target(
            source, flow, confidence, source.shape
        )

        self.assertTrue(np.allclose(warped[2:6, 3:5], 1.0))
        self.assertTrue(np.allclose(warped_confidence, 1.0))

    def test_motion_gate_suppresses_supported_single_frame_residual(self):
        current = np.full((8, 8), 0.08, dtype=np.float32)
        neighbor = np.zeros_like(current)
        high_confidence = np.ones_like(current)

        gated, previous, following, confidence = motion_confidence_gate(
            current,
            neighbor,
            high_confidence,
            neighbor,
            high_confidence,
            agreement_sigma=0.04,
        )

        self.assertLess(float(gated.max()), 0.002)
        self.assertTrue(np.allclose(previous, 0.0))
        self.assertTrue(np.allclose(following, 0.0))
        self.assertLess(float(confidence.max()), 0.02)

    def test_low_flow_confidence_falls_back_to_phase41_residual(self):
        current = np.full((8, 8), 0.08, dtype=np.float32)
        neighbor = np.zeros_like(current)
        no_confidence = np.zeros_like(current)

        gated, previous, following, confidence = motion_confidence_gate(
            current,
            neighbor,
            no_confidence,
            neighbor,
            no_confidence,
            agreement_sigma=0.04,
            previous_fallback=neighbor,
            next_fallback=neighbor,
        )

        self.assertTrue(np.array_equal(gated, current))
        self.assertTrue(np.array_equal(previous, neighbor))
        self.assertTrue(np.array_equal(following, neighbor))
        self.assertTrue(np.all(confidence == 0.0))

    def test_dis_alignment_is_stable_for_identical_frames(self):
        image = np.zeros((48, 64), dtype=np.uint8)
        image[12:36, 20:44] = 200

        alignment = calculate_bidirectional_alignment(
            image, image.copy(), max_short_side=32
        )

        self.assertEqual(alignment.full_shape, image.shape)
        self.assertLess(float(np.abs(alignment.flow_current_to_previous).max()), 1e-3)
        self.assertGreater(float(alignment.confidence_current.mean()), 0.99)


if __name__ == "__main__":
    unittest.main()
