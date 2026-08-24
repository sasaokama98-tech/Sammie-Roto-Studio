import unittest
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sammie.hybrid_hq import (
    BALANCED,
    MAXIMUM_DETAIL,
    PRESERVE_TEMPORAL,
    alpha_to_float,
    apply_edge_residual,
    build_edge_residual,
    merge_edge_refinement,
    stabilize_edge_residual,
    temporal_alpha_to_trimap,
)
from sammie.matting import HybridHQManager, create_matting_manager


class HybridHqTests(unittest.TestCase):
    def test_alpha_to_float_preserves_16_bit_precision(self):
        alpha = np.asarray([[0, 1, 32768, 65535]], dtype=np.uint16)
        result = alpha_to_float(alpha)
        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(float(result[0, 1]), 1.0 / 65535.0, places=8)
        self.assertEqual(float(result[0, -1]), 1.0)

    def test_soft_alpha_and_hard_boundary_form_unknown_band(self):
        alpha = np.zeros((48, 64), dtype=np.float32)
        alpha[:, 24:40] = 1.0
        alpha[:, 22:24] = 0.25
        alpha[:, 40:42] = 0.75

        trimap = temporal_alpha_to_trimap(alpha, edge_width=3)

        self.assertEqual(set(np.unique(trimap).tolist()), {0, 128, 255})
        self.assertTrue(np.all(trimap[:, 31] == 255))
        self.assertTrue(np.all(trimap[:, 23] == 128))
        self.assertTrue(np.all(trimap[:, 0] == 0))

    def test_merge_changes_only_unknown_region(self):
        temporal = np.zeros((32, 48), dtype=np.float32)
        temporal[:, 18:30] = 1.0
        trimap = temporal_alpha_to_trimap(temporal, edge_width=4)
        spatial = np.full_like(temporal, 0.375)

        merged = merge_edge_refinement(
            temporal, spatial, trimap, feather_width=2
        )

        known = trimap != 128
        self.assertTrue(np.array_equal(merged[known], temporal[known]))
        self.assertTrue(np.any(np.abs(merged[~known] - temporal[~known]) > 0.01))

    def test_merge_without_unknown_is_noop(self):
        temporal = np.ones((8, 8), dtype=np.float32)
        trimap = np.full((8, 8), 255, dtype=np.uint8)
        spatial = np.zeros((8, 8), dtype=np.float32)
        merged = merge_edge_refinement(temporal, spatial, trimap)
        self.assertTrue(np.array_equal(merged, temporal))

    def test_preserve_temporal_bounds_and_gates_mematte_residual(self):
        temporal = np.full((8, 8), 0.5, dtype=np.float32)
        trimap = np.full((8, 8), 128, dtype=np.uint8)
        moderate = np.full((8, 8), 0.65, dtype=np.float32)
        extreme = np.ones((8, 8), dtype=np.float32)

        moderate_residual = build_edge_residual(
            temporal,
            moderate,
            trimap,
            feather_width=0,
            preset=PRESERVE_TEMPORAL,
        )
        extreme_residual = build_edge_residual(
            temporal,
            extreme,
            trimap,
            feather_width=0,
            preset=PRESERVE_TEMPORAL,
        )

        self.assertGreater(float(moderate_residual.max()), 0.0)
        self.assertLessEqual(float(np.abs(moderate_residual).max()), 0.1)
        self.assertTrue(np.allclose(extreme_residual, 0.0))

    def test_three_frame_stabilization_suppresses_single_frame_impulse(self):
        trimap = np.full((8, 8), 128, dtype=np.uint8)
        previous = np.zeros((8, 8), dtype=np.float32)
        impulse = np.full((8, 8), 0.08, dtype=np.float32)
        following = np.zeros((8, 8), dtype=np.float32)

        preserve = stabilize_edge_residual(
            impulse,
            trimap,
            preset=PRESERVE_TEMPORAL,
            previous_residual=previous,
            next_residual=following,
        )
        balanced = stabilize_edge_residual(
            impulse,
            trimap,
            preset=BALANCED,
            previous_residual=previous,
            next_residual=following,
        )

        self.assertTrue(np.allclose(preserve, 0.02, atol=1e-6))
        self.assertTrue(np.allclose(balanced, 0.052, atol=1e-6))

    def test_maximum_detail_matches_legacy_merge(self):
        temporal = np.zeros((16, 24), dtype=np.float32)
        temporal[:, 8:16] = 1.0
        spatial = np.full_like(temporal, 0.375)
        trimap = temporal_alpha_to_trimap(temporal, edge_width=3)

        legacy = merge_edge_refinement(
            temporal, spatial, trimap, feather_width=2
        )
        residual = build_edge_residual(
            temporal,
            spatial,
            trimap,
            feather_width=2,
            preset=MAXIMUM_DETAIL,
        )
        result = apply_edge_residual(temporal, residual)

        self.assertTrue(np.allclose(result, legacy, atol=1e-7))

    def test_combined_mode_maps_multiple_objects_to_zero(self):
        points = [
            {"object_id": 2, "frame": 4},
            {"object_id": 7, "frame": 8},
        ]
        ids = HybridHQManager._active_output_ids(
            points, 0, 10, combined=True, temporal="MatAnyone2"
        )
        self.assertEqual(ids, [0])

    def test_factory_selects_hybrid_manager(self):
        settings = Mock()
        settings.get_session_setting.return_value = "Hybrid HQ"
        with patch("sammie.matting.get_settings_manager", return_value=settings):
            manager = create_matting_manager()
        self.assertIsInstance(manager, HybridHQManager)

    def test_temporal_stage_is_unloaded_before_edge_stage(self):
        events = []
        temporal = Mock()
        temporal.load_matting_model.side_effect = lambda **_kwargs: events.append(
            "temporal_load"
        ) or True
        temporal.run_matting.side_effect = lambda *_args, **_kwargs: events.append(
            "temporal_run"
        ) or 1
        temporal.unload_matting_model.side_effect = lambda: events.append(
            "temporal_unload"
        )
        temporal.propagated = True

        settings = Mock()
        values = {
            "hybrid_temporal_model": "MatAnyone2",
            "performance_metrics_enabled": False,
        }
        settings.get_session_setting.side_effect = (
            lambda key, default=None: values.get(key, default)
        )
        manager = HybridHQManager()
        manager._create_temporal_manager = Mock(return_value=temporal)
        manager._run_edge_stage = Mock(
            side_effect=lambda *_args: events.append("edge_load_and_run") or 1
        )

        with patch("sammie.matting.get_settings_manager", return_value=settings):
            result = manager.run_matting([], parent_window=Mock(), combined=False)

        self.assertEqual(result, 1)
        self.assertLess(events.index("temporal_unload"), events.index("edge_load_and_run"))
        self.assertTrue(manager.propagated)

    def test_enabled_phase43_runs_after_edge_stage(self):
        events = []
        temporal = Mock()
        temporal.load_matting_model.return_value = True
        temporal.run_matting.side_effect = (
            lambda *_args, **_kwargs: events.append("temporal") or 1
        )
        temporal.propagated = True

        values = {
            "hybrid_temporal_model": "MatAnyone2",
            "hybrid_evaluation_enabled": True,
            "performance_metrics_enabled": False,
        }
        settings = Mock()
        settings.get_session_setting.side_effect = (
            lambda key, default=None: values.get(key, default)
        )
        manager = HybridHQManager()
        manager._create_temporal_manager = Mock(return_value=temporal)
        manager._run_edge_stage = Mock(
            side_effect=lambda *_args: events.append("edge") or 1
        )
        manager._run_evaluation_stage = Mock(
            side_effect=lambda *_args: events.append("evaluation")
        )

        with (
            patch("sammie.matting.get_settings_manager", return_value=settings),
            patch(
                "sammie.matting.core.DeviceManager.get_device",
                return_value=torch.device("cpu"),
            ),
            patch("sammie.matting.core.DeviceManager.clear_cache"),
        ):
            result = manager.run_matting([], parent_window=Mock(), combined=False)

        self.assertEqual(result, 1)
        self.assertEqual(events, ["temporal", "edge", "evaluation"])
        manager._run_evaluation_stage.assert_called_once()

    def test_hybrid_run_writes_stage_performance_when_enabled(self):
        temporal = Mock()
        temporal.load_matting_model.return_value = True
        temporal.run_matting.return_value = 1
        temporal.propagated = True
        values = {
            "hybrid_temporal_model": "MatAnyone2",
            "hybrid_evaluation_enabled": False,
            "performance_metrics_enabled": True,
            "memory_profile": "Memory Safe",
            "in_point": 2,
            "out_point": 4,
        }
        settings = Mock()
        settings.get_session_setting.side_effect = (
            lambda key, default=None: values.get(key, default)
        )
        manager = HybridHQManager()
        manager._create_temporal_manager = Mock(return_value=temporal)
        manager._run_edge_stage = Mock(return_value=1)

        with (
            patch("sammie.matting.get_settings_manager", return_value=settings),
            patch("sammie.matting.core.VideoInfo.total_frames", 8),
            patch(
                "sammie.matting.core.DeviceManager.get_device",
                return_value=torch.device("cpu"),
            ),
            patch("sammie.matting.core.DeviceManager.clear_cache"),
            patch("sammie.matting.write_performance_report") as write_report,
        ):
            result = manager.run_matting(
                [{"object_id": 0, "frame": 2}],
                parent_window=Mock(),
                combined=False,
            )

        self.assertEqual(result, 1)
        write_report.assert_called_once()
        kwargs = write_report.call_args.kwargs
        self.assertEqual(kwargs["frame_equivalents"], 3)
        self.assertEqual(kwargs["metadata"]["memory_profile"], "Memory Safe")
        self.assertEqual(
            [stage["name"] for stage in kwargs["profiler"].stages],
            ["temporal", "edge"],
        )

    def test_edge_stage_writes_16_bit_hybrid_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            mattes = root / "matting" / "00000"
            frames.mkdir()
            mattes.mkdir(parents=True)

            image = np.zeros((32, 48, 3), dtype=np.uint8)
            temporal = np.zeros((32, 48), dtype=np.uint8)
            temporal[:, 18:30] = 255
            cv2.imwrite(str(frames / "00000.png"), image)
            cv2.imwrite(str(mattes / "0.png"), temporal)

            values = {
                "in_point": 0,
                "out_point": 0,
                "hybrid_edge_width": 3,
                "hybrid_edge_feather": 1,
                "mematte_roi_margin": 4,
                "mematte_tile_size": 32,
                "mematte_tile_overlap": 4,
                "mematte_max_tokens": 1024,
                "mematte_precision": "Float32",
            }
            settings = Mock()
            settings.get_session_setting.side_effect = (
                lambda key, default=None: values.get(key, default)
            )
            settings.get_app_setting.side_effect = lambda _key, default=None: default

            backend = Mock()
            backend.predict_multi_roi_float.return_value = np.full(
                temporal.shape, 0.375, dtype=np.float32
            )
            progress = Mock()
            progress.wasCanceled.return_value = False
            pbar = Mock()
            manager = HybridHQManager()
            manager._make_progress_dialog = Mock(return_value=(progress, pbar))

            with (
                patch("sammie.matting.get_settings_manager", return_value=settings),
                patch("sammie.matting.MematteBackend", return_value=backend),
                patch("sammie.matting.core.temp_dir", str(root)),
                patch("sammie.matting.core.frames_dir", str(frames)),
                patch("sammie.matting.core.matting_dir", str(root / "matting")),
                patch("sammie.matting.core.get_frame_extension", return_value="png"),
                patch("sammie.matting.core.VideoInfo.total_frames", 1),
                patch(
                    "sammie.matting.core.DeviceManager.get_device",
                    return_value=torch.device("cpu"),
                ),
                patch("sammie.matting.core.DeviceManager.clear_cache"),
            ):
                result = manager._run_edge_stage(
                    [{"object_id": 0, "frame": 0}],
                    parent_window=Mock(),
                    combined=False,
                    temporal_model="MatAnyone2",
                )

            hybrid = cv2.imread(str(mattes / "0.png"), cv2.IMREAD_UNCHANGED)
            diagnostic = cv2.imread(
                str(root / "hybrid_temporal" / "00000" / "0.png"),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(result, 1)
            self.assertEqual(hybrid.dtype, np.uint16)
            self.assertEqual(diagnostic.dtype, np.uint16)
            backend.load.assert_called_once()
            backend.predict_multi_roi_float.assert_called_once()
            backend.unload.assert_called_once()

    def test_edge_stage_motion_confidence_suppresses_middle_impulse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            matting_root = root / "matting"
            frames.mkdir()
            for frame_number in range(3):
                matte_dir = matting_root / f"{frame_number:05d}"
                matte_dir.mkdir(parents=True)
                cv2.imwrite(
                    str(frames / f"{frame_number:05d}.png"),
                    np.zeros((12, 16, 3), dtype=np.uint8),
                )
                cv2.imwrite(
                    str(matte_dir / "0.png"),
                    np.full((12, 16), 32768, dtype=np.uint16),
                )

            values = {
                "in_point": 0,
                "out_point": 2,
                "hybrid_edge_width": 0,
                "hybrid_edge_feather": 0,
                "hybrid_stability_preset": PRESERVE_TEMPORAL,
                "hybrid_motion_enabled": True,
                "hybrid_flow_resolution": 32,
                "mematte_roi_margin": 4,
                "mematte_tile_size": 32,
                "mematte_tile_overlap": 4,
                "mematte_max_tokens": 1024,
                "mematte_precision": "Float32",
            }
            settings = Mock()
            settings.get_session_setting.side_effect = (
                lambda key, default=None: values.get(key, default)
            )
            settings.get_app_setting.side_effect = lambda _key, default=None: default

            backend = Mock()
            backend.predict_multi_roi_float.side_effect = [
                np.full((12, 16), 0.5, dtype=np.float32),
                np.full((12, 16), 0.65, dtype=np.float32),
                np.full((12, 16), 0.5, dtype=np.float32),
            ]
            progress = Mock()
            progress.wasCanceled.return_value = False
            manager = HybridHQManager()
            manager._make_progress_dialog = Mock(
                return_value=(progress, Mock())
            )

            with (
                patch("sammie.matting.get_settings_manager", return_value=settings),
                patch("sammie.matting.MematteBackend", return_value=backend),
                patch("sammie.matting.core.temp_dir", str(root)),
                patch("sammie.matting.core.frames_dir", str(frames)),
                patch("sammie.matting.core.matting_dir", str(matting_root)),
                patch("sammie.matting.core.get_frame_extension", return_value="png"),
                patch("sammie.matting.core.VideoInfo.total_frames", 3),
                patch(
                    "sammie.matting.core.DeviceManager.get_device",
                    return_value=torch.device("cpu"),
                ),
                patch("sammie.matting.core.DeviceManager.clear_cache"),
            ):
                result = manager._run_edge_stage(
                    [{"object_id": 0, "frame": 0}],
                    parent_window=Mock(),
                    combined=False,
                    temporal_model="MatAnyone2",
                )

            middle = cv2.imread(
                str(matting_root / "00001" / "0.png"), cv2.IMREAD_UNCHANGED
            ).astype(np.float32) / 65535.0
            confidence = cv2.imread(
                str(root / "hybrid_motion_confidence" / "00001" / "0.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            report_path = (
                root
                / "hybrid_diagnostics"
                / "matanyone2_preserve_temporal_motion.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(result, 1)
            self.assertGreater(float(middle.mean()), 0.499)
            self.assertLess(float(middle.mean()), 0.505)
            self.assertIsNotNone(confidence)
            self.assertEqual(report["phase"], "4.2")
            self.assertTrue(report["motion_enabled"])
            self.assertEqual(report["frame_range"], [0, 2])
            self.assertEqual(backend.predict_multi_roi_float.call_count, 3)


if __name__ == "__main__":
    unittest.main()
