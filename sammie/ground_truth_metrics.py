"""Reference alpha-matting metrics for Hybrid HQ evaluation.

The spatial metrics follow the conventional alpha-matting benchmark units:
SAD, gradient, and connectivity are divided by 1000, while MSE is the mean
squared error.  dtSSD follows the video-matting convention used by the
MatAnyone2/RVM evaluation code: RMS temporal-derivative error multiplied by
100.  Lower is better for every metric.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


GROUND_TRUTH_EXTENSIONS = (".png", ".tif", ".tiff", ".exr")


def read_alpha(path: str | os.PathLike) -> np.ndarray:
    """Read an alpha image as clipped float32 in the 0..1 range."""

    source = Path(path)
    raw = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"Unable to read ground-truth alpha: {source}")
    if raw.ndim == 3:
        raw = raw[..., 0]
    if raw.dtype == np.uint8:
        alpha = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        alpha = raw.astype(np.float32) / 65535.0
    else:
        alpha = raw.astype(np.float32)
    if not np.all(np.isfinite(alpha)):
        raise ValueError(f"Ground-truth alpha contains non-finite values: {source}")
    return np.clip(alpha, 0.0, 1.0)


def resolve_ground_truth_path(
    root: str | os.PathLike,
    frame: int,
    object_id: int,
    *,
    allow_flat: bool,
    source_frame: int | None = None,
    source_padding: int = 0,
) -> Path:
    """Resolve supported frame/object or object/frame ground-truth layouts."""

    base = Path(root)
    stems = []
    if source_frame is not None:
        if source_padding > 0:
            stems.append(f"{source_frame:0{source_padding}d}")
        stems.append(str(source_frame))
    stems.extend((f"{frame:05d}", str(frame)))
    stems = tuple(dict.fromkeys(stems))
    candidates = []
    for extension in GROUND_TRUTH_EXTENSIONS:
        for stem in stems:
            candidates.extend(
                (
                    base / stem / f"{object_id}{extension}",
                    base / str(object_id) / f"{stem}{extension}",
                )
            )
            if allow_flat:
                candidates.append(base / f"{stem}{extension}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing ground-truth alpha for frame {frame}, object {object_id} "
        f"under {base}"
    )


def _gaussian_derivative_filters(
    sigma: float = 1.4, epsilon: float = 1e-2
) -> tuple[np.ndarray, np.ndarray]:
    half_size = int(
        np.ceil(
            sigma
            * np.sqrt(-2.0 * np.log(np.sqrt(2.0 * np.pi) * sigma * epsilon))
        )
    )
    coordinates = np.arange(-half_size, half_size + 1, dtype=np.float64)
    gaussian = np.exp(-(coordinates**2) / (2.0 * sigma**2)) / (
        sigma * np.sqrt(2.0 * np.pi)
    )
    derivative = -coordinates * gaussian / sigma**2
    filter_x = np.outer(gaussian, derivative)
    norm = float(np.sqrt(np.sum(filter_x**2)))
    if norm > 0.0:
        filter_x /= norm
    return filter_x.astype(np.float32), filter_x.T.astype(np.float32)


_GRADIENT_FILTER_X, _GRADIENT_FILTER_Y = _gaussian_derivative_filters()


def _gradient_magnitude(alpha: np.ndarray) -> np.ndarray:
    gradient_x = cv2.filter2D(
        alpha, cv2.CV_32F, _GRADIENT_FILTER_X, borderType=cv2.BORDER_REPLICATE
    )
    gradient_y = cv2.filter2D(
        alpha, cv2.CV_32F, _GRADIENT_FILTER_Y, borderType=cv2.BORDER_REPLICATE
    )
    return cv2.magnitude(gradient_x, gradient_y)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=4
    )
    if component_count <= 1:
        return np.zeros(binary.shape, dtype=bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def connectivity_error(
    prediction: np.ndarray, ground_truth: np.ndarray, step: float = 0.1
) -> float:
    """Compute the alpha-matting benchmark connectivity error."""

    round_down = np.full(ground_truth.shape, -1.0, dtype=np.float32)
    thresholds = np.arange(0.0, 1.0 + step, step, dtype=np.float32)
    for index in range(1, len(thresholds)):
        intersection = (
            (ground_truth >= thresholds[index])
            & (prediction >= thresholds[index])
        )
        omega = _largest_component(intersection)
        newly_excluded = (round_down < 0.0) & ~omega
        round_down[newly_excluded] = thresholds[index - 1]
    round_down[round_down < 0.0] = 1.0

    ground_truth_difference = ground_truth - round_down
    prediction_difference = prediction - round_down
    ground_truth_phi = 1.0 - ground_truth_difference * (
        ground_truth_difference >= 0.15
    )
    prediction_phi = 1.0 - prediction_difference * (
        prediction_difference >= 0.15
    )
    return float(np.sum(np.abs(ground_truth_phi - prediction_phi)) / 1000.0)


def spatial_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Return conventional SAD/MSE/gradient/connectivity alpha metrics."""

    prediction = np.asarray(prediction, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match ground truth "
            f"{ground_truth.shape}"
        )
    difference = prediction - ground_truth
    prediction_gradient = _gradient_magnitude(prediction)
    ground_truth_gradient = _gradient_magnitude(ground_truth)
    return {
        "sad": float(np.sum(np.abs(difference)) / 1000.0),
        "mse": float(np.mean(difference**2)),
        "gradient": float(
            np.sum((prediction_gradient - ground_truth_gradient) ** 2) / 1000.0
        ),
        "connectivity": connectivity_error(prediction, ground_truth),
    }


def dtssd_metric(
    prediction: np.ndarray,
    previous_prediction: np.ndarray,
    ground_truth: np.ndarray,
    previous_ground_truth: np.ndarray,
) -> float:
    """Return direct temporal derivative SSD in standard video-matting units."""

    shapes = {
        np.asarray(value).shape
        for value in (
            prediction,
            previous_prediction,
            ground_truth,
            previous_ground_truth,
        )
    }
    if len(shapes) != 1:
        raise ValueError(f"dtSSD inputs have inconsistent shapes: {sorted(shapes)}")
    temporal_error = (
        np.asarray(prediction, dtype=np.float32)
        - np.asarray(previous_prediction, dtype=np.float32)
        - np.asarray(ground_truth, dtype=np.float32)
        + np.asarray(previous_ground_truth, dtype=np.float32)
    )
    return float(np.sqrt(np.mean(temporal_error**2)) * 100.0)


def _aggregate_variant(frames: list[dict], pairs: list[dict]) -> dict:
    return {
        "frames": len(frames),
        "adjacent_pairs": len(pairs),
        "sad_mean": float(np.mean([item["sad"] for item in frames])),
        "sad_sum": float(np.sum([item["sad"] for item in frames])),
        "mse_mean": float(np.mean([item["mse"] for item in frames])),
        "gradient_mean": float(np.mean([item["gradient"] for item in frames])),
        "connectivity_mean": float(
            np.mean([item["connectivity"] for item in frames])
        ),
        "dtssd_mean": (
            float(np.mean([item["dtssd"] for item in pairs])) if pairs else 0.0
        ),
    }


def evaluate_ground_truth(
    *,
    final_dir: str | os.PathLike,
    temporal_dir: str | os.PathLike,
    ground_truth_dir: str | os.PathLike,
    frame_range: tuple[int, int],
    object_ids: list[int],
    source_frame_numbers: list[int] | None = None,
    source_frame_padding: int = 0,
) -> dict:
    """Evaluate final and temporal alpha sequences against complete GT mattes."""

    ground_truth_root = Path(ground_truth_dir)
    if not ground_truth_root.is_dir():
        raise NotADirectoryError(
            f"Ground-truth alpha folder does not exist: {ground_truth_root}"
        )
    start_frame, end_frame = frame_range
    frames = list(range(start_frame, end_frame + 1))
    if not frames:
        raise ValueError("Ground-truth evaluation frame range is empty")
    allow_flat = len(object_ids) == 1
    report_objects = []

    for object_id in object_ids:
        variants = {
            "final": {"frames": [], "pairs": []},
            "temporal": {"frames": [], "pairs": []},
        }
        previous = None
        for frame in frames:
            source_frame = None
            if source_frame_numbers and frame < len(source_frame_numbers):
                source_frame = int(source_frame_numbers[frame])
            ground_truth_path = resolve_ground_truth_path(
                ground_truth_root,
                frame,
                object_id,
                allow_flat=allow_flat,
                source_frame=source_frame,
                source_padding=source_frame_padding,
            )
            ground_truth = read_alpha(ground_truth_path)
            final = read_alpha(
                Path(final_dir) / f"{frame:05d}" / f"{object_id}.png"
            )
            temporal = read_alpha(
                Path(temporal_dir) / f"{frame:05d}" / f"{object_id}.png"
            )
            for name, prediction in (("final", final), ("temporal", temporal)):
                metrics = spatial_metrics(prediction, ground_truth)
                metrics["frame"] = int(frame)
                metrics["source_frame"] = source_frame
                variants[name]["frames"].append(metrics)
                if previous is not None:
                    variants[name]["pairs"].append(
                        {
                            "previous_frame": int(frame - 1),
                            "current_frame": int(frame),
                            "dtssd": dtssd_metric(
                                prediction,
                                previous[name],
                                ground_truth,
                                previous["ground_truth"],
                            ),
                        }
                    )
            previous = {
                "ground_truth": ground_truth,
                "final": final,
                "temporal": temporal,
            }

        final_aggregate = _aggregate_variant(
            variants["final"]["frames"], variants["final"]["pairs"]
        )
        temporal_aggregate = _aggregate_variant(
            variants["temporal"]["frames"], variants["temporal"]["pairs"]
        )
        lower_is_better = (
            "sad_mean",
            "mse_mean",
            "gradient_mean",
            "connectivity_mean",
            "dtssd_mean",
        )
        report_objects.append(
            {
                "object_id": int(object_id),
                "final": {
                    "aggregate": final_aggregate,
                    **variants["final"],
                },
                "temporal": {
                    "aggregate": temporal_aggregate,
                    **variants["temporal"],
                },
                "final_minus_temporal": {
                    key: float(final_aggregate[key] - temporal_aggregate[key])
                    for key in lower_is_better
                },
            }
        )

    return {
        "source": str(ground_truth_root.resolve()),
        "layout": (
            "<frame>/<object>.png or <object>/<frame>.png; flat <frame>.png "
            "is accepted for one object"
        ),
        "units": {
            "sad": "sum(abs(error)) / 1000",
            "mse": "mean(square(error))",
            "gradient": "sum(square(gaussian-gradient error)) / 1000",
            "connectivity": "sum(connectivity error) / 1000",
            "dtssd": "sqrt(mean(square(temporal derivative error))) * 100",
        },
        "lower_is_better": True,
        "objects": report_objects,
    }
