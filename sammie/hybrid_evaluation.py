"""No-reference production diagnostics for Hybrid HQ Phase 4.3.

These metrics compare a Hybrid HQ result with its temporal base.  They are not
substitutes for ground-truth SAD, MSE, gradient, connectivity, or dtSSD scores.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from sammie.hybrid_hq import alpha_to_float
from sammie.motion_confidence import (
    calculate_bidirectional_alignment,
    warp_source_to_target,
)


class HybridEvaluationCancelled(RuntimeError):
    """Raised when an optional Phase 4.3 evaluation is cancelled."""


ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


def sanitize_run_label(value: str) -> str:
    """Return a filesystem-safe, stable evaluation label."""

    label = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    label = label.strip("._-")
    return label[:80] or "hybrid_hq"


def make_unique_run_dir(output_root: str | os.PathLike, label: str) -> Path:
    """Create and return a run directory without overwriting earlier results."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    base = sanitize_run_label(label)
    candidate = root / base
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base}_{suffix:03d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _read_alpha(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"Unable to read alpha: {path}")
    if raw.ndim == 3:
        raw = raw[..., 0]
    return alpha_to_float(raw)


def _read_trimap(path: Path, shape: tuple[int, int]) -> np.ndarray:
    trimap = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if trimap is None:
        raise OSError(f"Unable to read Hybrid HQ trimap: {path}")
    if trimap.shape != shape:
        trimap = cv2.resize(
            trimap, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return trimap


def _masked_values(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)[mask]
    return values[np.isfinite(values)]


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "count": int(values.size),
    }


def _gradient_magnitude(alpha: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _confidence_path(confidence_dir: Path, object_id: int, frame: int) -> Path:
    current = confidence_dir / f"{frame:05d}" / f"{object_id}.png"
    if current.exists():
        return current
    # A short-lived Phase 4.2 build used object/frame ordering. Continue reading
    # it so an existing diagnostic can still be evaluated after the contract is
    # restored to Sammie's established frame/object ordering.
    return confidence_dir / str(object_id) / f"{frame:05d}.png"


def _weighted_mean(items: list[tuple[float, int]]) -> float:
    count = sum(count for _, count in items)
    if count <= 0:
        return 0.0
    return float(sum(value * count for value, count in items) / count)


def _aggregate_object(frame_metrics: list[dict], pair_metrics: list[dict]) -> dict:
    def frame_mean(key: str) -> float:
        return _weighted_mean(
            [(float(item[key]["mean"]), int(item[key]["count"])) for item in frame_metrics]
        )

    def frame_max(key: str) -> float:
        return max((float(item[key]["max"]) for item in frame_metrics), default=0.0)

    def pair_mean(key: str) -> float:
        return _weighted_mean(
            [(float(item[key]["mean"]), int(item[key]["count"])) for item in pair_metrics]
        )

    base_gradient = frame_mean("temporal_boundary_gradient")
    final_gradient = frame_mean("final_boundary_gradient")
    base_flow_error = pair_mean("temporal_flow_warped_error")
    final_flow_error = pair_mean("final_flow_warped_error")
    confidence_values = [
        item["motion_confidence"]
        for item in frame_metrics
        if item.get("motion_confidence") is not None
    ]
    return {
        "frames": len(frame_metrics),
        "adjacent_pairs": len(pair_metrics),
        "unknown_residual_mean": frame_mean("unknown_residual"),
        "unknown_residual_max": frame_max("unknown_residual"),
        "known_region_drift_mean": frame_mean("known_region_drift"),
        "known_region_drift_max": frame_max("known_region_drift"),
        "temporal_boundary_gradient_mean": base_gradient,
        "final_boundary_gradient_mean": final_gradient,
        "boundary_gradient_ratio_final_to_temporal": (
            final_gradient / base_gradient if base_gradient > 1e-8 else 1.0
        ),
        "temporal_unwarped_delta_mean": pair_mean("temporal_unwarped_delta"),
        "final_unwarped_delta_mean": pair_mean("final_unwarped_delta"),
        "temporal_flow_warped_error_mean": base_flow_error,
        "final_flow_warped_error_mean": final_flow_error,
        "flow_warped_error_ratio_final_to_temporal": (
            final_flow_error / base_flow_error if base_flow_error > 1e-8 else 1.0
        ),
        "residual_flow_warped_error_mean": pair_mean(
            "residual_flow_warped_error"
        ),
        "motion_confidence_mean": (
            _weighted_mean(
                [(float(item["mean"]), int(item["count"])) for item in confidence_values]
            )
            if confidence_values else None
        ),
        "motion_confidence_low_fraction": (
            _weighted_mean(
                [(float(item["low_fraction"]), int(item["count"])) for item in confidence_values]
            )
            if confidence_values else None
        ),
        "motion_confidence_high_fraction": (
            _weighted_mean(
                [(float(item["high_fraction"]), int(item["count"])) for item in confidence_values]
            )
            if confidence_values else None
        ),
    }


def _write_summary_csv(output_root: Path, report: dict) -> None:
    path = output_root / "summary.csv"
    fields = [
        "run_label", "created_utc", "temporal_model", "stability_preset",
        "motion_enabled", "flow_resolution", "start_frame", "end_frame",
        "object_id", "unknown_residual_mean", "known_region_drift_mean",
        "boundary_gradient_ratio_final_to_temporal",
        "temporal_flow_warped_error_mean", "final_flow_warped_error_mean",
        "flow_warped_error_ratio_final_to_temporal",
        "residual_flow_warped_error_mean", "motion_confidence_mean",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for item in report["objects"]:
            aggregate = item["aggregate"]
            writer.writerow(
                {
                    "run_label": report["run_label"],
                    "created_utc": report["created_utc"],
                    "temporal_model": report["settings"].get("temporal_model"),
                    "stability_preset": report["settings"].get("stability_preset"),
                    "motion_enabled": report["settings"].get("motion_enabled"),
                    "flow_resolution": report["settings"].get("flow_resolution"),
                    "start_frame": report["frame_range"][0],
                    "end_frame": report["frame_range"][1],
                    "object_id": item["object_id"],
                    **{field: aggregate.get(field) for field in fields if field in aggregate},
                }
            )


def evaluate_hybrid_run(
    *,
    frames_dir: str | os.PathLike,
    matting_dir: str | os.PathLike,
    temporal_dir: str | os.PathLike,
    trimap_dir: str | os.PathLike,
    confidence_dir: str | os.PathLike,
    output_root: str | os.PathLike,
    frame_range: tuple[int, int],
    object_ids: list[int],
    frame_extension: str,
    run_label: str,
    settings: dict,
    flow_resolution: int = 720,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> Path:
    """Archive and evaluate a completed Hybrid HQ run.

    Flow is calculated once per adjacent frame pair and reused across objects.
    The newly-created run directory is removed if evaluation fails or is cancelled.
    """

    start_frame, end_frame = frame_range
    if end_frame < start_frame:
        raise ValueError("Hybrid evaluation frame range is invalid")
    if not object_ids:
        raise ValueError("Hybrid evaluation requires at least one object")

    frames_path = Path(frames_dir)
    mattes_path = Path(matting_dir)
    temporal_path = Path(temporal_dir)
    trimaps_path = Path(trimap_dir)
    confidence_path = Path(confidence_dir)
    output_path = Path(output_root)
    run_dir = make_unique_run_dir(output_path, run_label)
    frames = list(range(start_frame, end_frame + 1))
    total = len(frames) * len(object_ids) + max(0, len(frames) - 1)
    completed = 0

    def update(message: str) -> None:
        nonlocal completed
        if cancel_callback is not None and cancel_callback():
            raise HybridEvaluationCancelled("Hybrid HQ evaluation cancelled")
        if progress_callback is not None:
            progress_callback(completed, max(total, 1), message)

    objects = {
        int(object_id): {"frame_metrics": [], "pair_metrics": []}
        for object_id in object_ids
    }
    try:
        for object_id in object_ids:
            for frame in frames:
                update(f"Archiving object {object_id}, frame {frame}")
                final_source = mattes_path / f"{frame:05d}" / f"{object_id}.png"
                temporal_source = temporal_path / f"{frame:05d}" / f"{object_id}.png"
                trimap_source = trimaps_path / f"{frame:05d}" / f"{object_id}.png"
                final = _read_alpha(final_source)
                temporal = _read_alpha(temporal_source)
                if final.shape != temporal.shape:
                    raise ValueError(
                        f"Hybrid alpha dimensions differ at frame {frame}, object {object_id}"
                    )
                trimap = _read_trimap(trimap_source, final.shape)
                unknown = trimap == 128
                known = ~unknown
                residual = np.abs(final - temporal)
                temporal_gradient = _gradient_magnitude(temporal)
                final_gradient = _gradient_magnitude(final)

                confidence_metrics = None
                source_confidence = _confidence_path(
                    confidence_path, object_id, frame
                )
                if source_confidence.exists():
                    confidence = cv2.imread(
                        str(source_confidence), cv2.IMREAD_GRAYSCALE
                    )
                    if confidence is not None:
                        if confidence.shape != final.shape:
                            confidence = cv2.resize(
                                confidence,
                                (final.shape[1], final.shape[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        values = _masked_values(confidence / 255.0, unknown)
                        confidence_metrics = _stats(values)
                        confidence_metrics["low_fraction"] = (
                            float(np.mean(values < 0.2)) if values.size else 0.0
                        )
                        confidence_metrics["high_fraction"] = (
                            float(np.mean(values > 0.8)) if values.size else 0.0
                        )

                objects[int(object_id)]["frame_metrics"].append(
                    {
                        "frame": int(frame),
                        "unknown_residual": _stats(_masked_values(residual, unknown)),
                        "known_region_drift": _stats(_masked_values(residual, known)),
                        "temporal_boundary_gradient": _stats(
                            _masked_values(temporal_gradient, unknown)
                        ),
                        "final_boundary_gradient": _stats(
                            _masked_values(final_gradient, unknown)
                        ),
                        "motion_confidence": confidence_metrics,
                    }
                )

                final_dest = run_dir / "final" / f"{frame:05d}" / f"{object_id}.png"
                temporal_dest = (
                    run_dir / "temporal" / f"{frame:05d}" / f"{object_id}.png"
                )
                final_dest.parent.mkdir(parents=True, exist_ok=True)
                temporal_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(final_source, final_dest)
                shutil.copy2(temporal_source, temporal_dest)
                if source_confidence.exists():
                    confidence_dest = (
                        run_dir / "confidence" / f"{frame:05d}" / f"{object_id}.png"
                    )
                    confidence_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_confidence, confidence_dest)
                completed += 1

        for previous_frame, current_frame in zip(frames, frames[1:]):
            update(f"Evaluating motion {previous_frame}-{current_frame}")
            previous_image = cv2.imread(
                str(frames_path / f"{previous_frame:05d}.{frame_extension}"),
                cv2.IMREAD_GRAYSCALE,
            )
            current_image = cv2.imread(
                str(frames_path / f"{current_frame:05d}.{frame_extension}"),
                cv2.IMREAD_GRAYSCALE,
            )
            if previous_image is None or current_image is None:
                raise OSError(
                    f"Unable to read source frames {previous_frame}-{current_frame}"
                )
            alignment = calculate_bidirectional_alignment(
                previous_image, current_image, max_short_side=flow_resolution
            )
            for object_id in object_ids:
                previous_temporal = _read_alpha(
                    temporal_path / f"{previous_frame:05d}" / f"{object_id}.png"
                )
                current_temporal = _read_alpha(
                    temporal_path / f"{current_frame:05d}" / f"{object_id}.png"
                )
                previous_final = _read_alpha(
                    mattes_path / f"{previous_frame:05d}" / f"{object_id}.png"
                )
                current_final = _read_alpha(
                    mattes_path / f"{current_frame:05d}" / f"{object_id}.png"
                )
                current_trimap = _read_trimap(
                    trimaps_path / f"{current_frame:05d}" / f"{object_id}.png",
                    current_final.shape,
                )
                unknown = current_trimap == 128
                previous_temporal_warped, flow_confidence = warp_source_to_target(
                    previous_temporal,
                    alignment.flow_current_to_previous,
                    alignment.confidence_current,
                    current_temporal.shape,
                )
                previous_final_warped, _ = warp_source_to_target(
                    previous_final,
                    alignment.flow_current_to_previous,
                    alignment.confidence_current,
                    current_final.shape,
                )
                previous_residual = previous_final - previous_temporal
                current_residual = current_final - current_temporal
                previous_residual_warped, _ = warp_source_to_target(
                    previous_residual,
                    alignment.flow_current_to_previous,
                    alignment.confidence_current,
                    current_residual.shape,
                )
                reliable_unknown = unknown & (flow_confidence >= 0.2)
                objects[int(object_id)]["pair_metrics"].append(
                    {
                        "previous_frame": int(previous_frame),
                        "current_frame": int(current_frame),
                        "flow_confidence": _stats(
                            _masked_values(flow_confidence, unknown)
                        ),
                        "temporal_unwarped_delta": _stats(
                            _masked_values(
                                np.abs(current_temporal - previous_temporal), unknown
                            )
                        ),
                        "final_unwarped_delta": _stats(
                            _masked_values(
                                np.abs(current_final - previous_final), unknown
                            )
                        ),
                        "temporal_flow_warped_error": _stats(
                            _masked_values(
                                np.abs(current_temporal - previous_temporal_warped),
                                reliable_unknown,
                            )
                        ),
                        "final_flow_warped_error": _stats(
                            _masked_values(
                                np.abs(current_final - previous_final_warped),
                                reliable_unknown,
                            )
                        ),
                        "residual_flow_warped_error": _stats(
                            _masked_values(
                                np.abs(current_residual - previous_residual_warped),
                                reliable_unknown,
                            )
                        ),
                    }
                )
            completed += 1

        created_utc = datetime.now(timezone.utc).isoformat()
        report = {
            "phase": "4.3",
            "metric_type": "no-reference comparative diagnostics",
            "ground_truth_notice": (
                "These values compare final Hybrid HQ alpha with its temporal base. "
                "They are not ground-truth SAD, MSE, gradient, connectivity, or dtSSD scores."
            ),
            "run_label": run_dir.name,
            "created_utc": created_utc,
            "frame_range": [int(start_frame), int(end_frame)],
            "settings": dict(settings),
            "objects": [],
        }
        for object_id in object_ids:
            frame_metrics = objects[int(object_id)]["frame_metrics"]
            pair_metrics = objects[int(object_id)]["pair_metrics"]
            report["objects"].append(
                {
                    "object_id": int(object_id),
                    "aggregate": _aggregate_object(frame_metrics, pair_metrics),
                    "frames": frame_metrics,
                    "pairs": pair_metrics,
                }
            )
        report_path = run_dir / "report.json"
        with report_path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        _write_summary_csv(output_path, report)
        if progress_callback is not None:
            progress_callback(max(total, 1), max(total, 1), "Evaluation complete")
        return report_path
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
