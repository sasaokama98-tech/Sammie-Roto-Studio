import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from sammie.performance_metrics import MattingRunProfiler, write_performance_report


class PerformanceMetricTests(unittest.TestCase):
    def test_cuda_stage_records_peak_and_delta(self):
        device = Mock(type="cuda")
        profiler = MattingRunProfiler(device)
        with (
            patch(
                "sammie.performance_metrics.time.perf_counter",
                side_effect=[4.0, 5.25],
            ),
            patch("sammie.performance_metrics.torch.cuda.is_available", return_value=True),
            patch("sammie.performance_metrics.torch.cuda.synchronize") as sync,
            patch(
                "sammie.performance_metrics.torch.cuda.memory_allocated",
                return_value=100,
            ),
            patch(
                "sammie.performance_metrics.torch.cuda.memory_reserved",
                return_value=200,
            ),
            patch("sammie.performance_metrics.torch.cuda.reset_peak_memory_stats") as reset,
            patch(
                "sammie.performance_metrics.torch.cuda.max_memory_allocated",
                return_value=500,
            ),
            patch(
                "sammie.performance_metrics.torch.cuda.max_memory_reserved",
                return_value=800,
            ),
        ):
            with profiler.stage("edge", work_items=1):
                pass

        metric = profiler.stages[0]
        self.assertEqual(metric["cuda_peak_allocated_bytes"], 500)
        self.assertEqual(metric["cuda_peak_reserved_bytes"], 800)
        self.assertEqual(metric["cuda_peak_allocated_delta_bytes"], 400)
        self.assertEqual(metric["cuda_peak_reserved_delta_bytes"], 600)
        self.assertEqual(sync.call_count, 2)
        reset.assert_called_once_with(device)

    def test_cpu_stage_timing_and_report_are_written(self):
        profiler = MattingRunProfiler(torch.device("cpu"))
        with patch(
            "sammie.performance_metrics.time.perf_counter",
            side_effect=[10.0, 12.5, 20.0, 21.0],
        ):
            with profiler.stage("temporal", work_items=5):
                pass
            with profiler.stage("edge", work_items=5):
                pass

        self.assertEqual(profiler.stages[0]["elapsed_seconds"], 2.5)
        self.assertEqual(profiler.stages[0]["seconds_per_work_item"], 0.5)
        self.assertIsNone(profiler.stages[0]["cuda_peak_reserved_bytes"])

        with tempfile.TemporaryDirectory() as directory:
            path = write_performance_report(
                directory,
                label="VideoMaMa / Balanced",
                metadata={
                    "status": "completed",
                    "model": "Hybrid HQ",
                    "temporal_model": "VideoMaMa",
                    "memory_profile": "Balanced",
                    "start_frame": 10,
                    "end_frame": 14,
                    "objects": 1,
                },
                profiler=profiler,
                frame_equivalents=5,
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["phase"], "5.1")
            self.assertEqual(report["aggregate"]["elapsed_seconds"], 3.5)
            self.assertEqual(
                report["aggregate"]["seconds_per_frame_equivalent"], 0.7
            )
            with (Path(directory) / "summary.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["memory_profile"], "Balanced")
            self.assertEqual(rows[0]["temporal_seconds"], "2.5")


if __name__ == "__main__":
    unittest.main()
