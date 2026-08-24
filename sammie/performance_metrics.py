"""Low-overhead stage timing and CUDA peak-memory reports."""

from __future__ import annotations

import csv
import json
import re
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path

import torch


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    return label.strip("._-")[:80] or "matting"


class _StageMeasurement(AbstractContextManager):
    def __init__(self, owner, name: str, work_items: int):
        self.owner = owner
        self.name = name
        self.work_items = max(0, int(work_items))
        self.started = 0.0
        self.start_allocated = 0
        self.start_reserved = 0
        self.cuda_measured = False

    def _prepare_cuda(self):
        device = self.owner.device
        if getattr(device, "type", str(device)) != "cuda":
            return
        try:
            if not torch.cuda.is_available():
                return
            torch.cuda.synchronize(device)
            self.start_allocated = int(torch.cuda.memory_allocated(device))
            self.start_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
            self.cuda_measured = True
        except (RuntimeError, TypeError, AttributeError):
            self.cuda_measured = False

    def __enter__(self):
        self._prepare_cuda()
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        peak_allocated = None
        peak_reserved = None
        if self.cuda_measured:
            try:
                torch.cuda.synchronize(self.owner.device)
                peak_allocated = int(
                    torch.cuda.max_memory_allocated(self.owner.device)
                )
                peak_reserved = int(
                    torch.cuda.max_memory_reserved(self.owner.device)
                )
            except (RuntimeError, TypeError, AttributeError):
                peak_allocated = None
                peak_reserved = None
        elapsed = max(0.0, time.perf_counter() - self.started)
        self.owner.stages.append(
            {
                "name": self.name,
                "status": "failed" if exc_type is not None else "completed",
                "elapsed_seconds": float(elapsed),
                "work_items": self.work_items,
                "seconds_per_work_item": (
                    float(elapsed / self.work_items) if self.work_items else None
                ),
                "cuda_peak_allocated_bytes": peak_allocated,
                "cuda_peak_reserved_bytes": peak_reserved,
                "cuda_peak_allocated_delta_bytes": (
                    max(0, peak_allocated - self.start_allocated)
                    if peak_allocated is not None else None
                ),
                "cuda_peak_reserved_delta_bytes": (
                    max(0, peak_reserved - self.start_reserved)
                    if peak_reserved is not None else None
                ),
            }
        )
        return False


class MattingRunProfiler:
    """Collect sequential stage measurements for one matting run."""

    def __init__(self, device):
        self.device = device
        self.stages: list[dict] = []

    def stage(self, name: str, work_items: int = 0):
        return _StageMeasurement(self, name, work_items)

    def aggregate(self, frame_equivalents: int) -> dict:
        total = float(sum(stage["elapsed_seconds"] for stage in self.stages))
        allocated = [
            stage["cuda_peak_allocated_bytes"]
            for stage in self.stages
            if stage["cuda_peak_allocated_bytes"] is not None
        ]
        reserved = [
            stage["cuda_peak_reserved_bytes"]
            for stage in self.stages
            if stage["cuda_peak_reserved_bytes"] is not None
        ]
        return {
            "elapsed_seconds": total,
            "frame_equivalents": int(frame_equivalents),
            "seconds_per_frame_equivalent": (
                total / frame_equivalents if frame_equivalents > 0 else None
            ),
            "cuda_peak_allocated_bytes": max(allocated) if allocated else None,
            "cuda_peak_reserved_bytes": max(reserved) if reserved else None,
        }


def write_performance_report(
    output_root: str | Path,
    *,
    label: str,
    metadata: dict,
    profiler: MattingRunProfiler,
    frame_equivalents: int,
) -> Path:
    """Write one detailed JSON and append one comparison CSV row."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    timestamp = created.strftime("%Y%m%dT%H%M%S_%fZ")
    report_path = root / f"{timestamp}_{_safe_label(label)}.json"
    aggregate = profiler.aggregate(frame_equivalents)
    report = {
        "phase": "5.1",
        "created_utc": created.isoformat(),
        "metadata": dict(metadata),
        "aggregate": aggregate,
        "stages": list(profiler.stages),
    }
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    stage_map = {stage["name"]: stage for stage in profiler.stages}
    csv_path = root / "summary.csv"
    fields = [
        "created_utc", "label", "status", "model", "temporal_model",
        "memory_profile", "start_frame", "end_frame", "objects",
        "frame_equivalents", "elapsed_seconds", "seconds_per_frame_equivalent",
        "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
        "temporal_seconds", "temporal_peak_reserved_bytes",
        "edge_seconds", "edge_peak_reserved_bytes", "evaluation_seconds",
    ]
    row = {
        "created_utc": report["created_utc"],
        "label": label,
        **{key: metadata.get(key) for key in fields if key in metadata},
        **{key: aggregate.get(key) for key in fields if key in aggregate},
        "temporal_seconds": stage_map.get("temporal", {}).get("elapsed_seconds"),
        "temporal_peak_reserved_bytes": stage_map.get("temporal", {}).get(
            "cuda_peak_reserved_bytes"
        ),
        "edge_seconds": stage_map.get("edge", {}).get("elapsed_seconds"),
        "edge_peak_reserved_bytes": stage_map.get("edge", {}).get(
            "cuda_peak_reserved_bytes"
        ),
        "evaluation_seconds": stage_map.get("evaluation", {}).get(
            "elapsed_seconds"
        ),
    }
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return report_path
