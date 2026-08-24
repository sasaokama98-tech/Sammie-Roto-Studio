"""Motion-aligned residual confidence for experimental Hybrid HQ Phase 4.2."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BidirectionalFlowAlignment:
    """Reduced-resolution flows and confidence for one adjacent frame pair."""

    flow_current_to_previous: np.ndarray
    confidence_current: np.ndarray
    flow_previous_to_current: np.ndarray
    confidence_previous: np.ndarray
    full_shape: tuple[int, int]


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 3:
        value = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
    if value.ndim != 2:
        raise ValueError("Hybrid motion input must be a grayscale or BGR image")
    if value.dtype == np.uint8:
        return value
    value = value.astype(np.float32)
    if value.size and float(np.nanmax(value)) <= 1.0:
        value *= 255.0
    return np.clip(np.nan_to_num(value), 0.0, 255.0).astype(np.uint8)


def _resize_for_flow(gray: np.ndarray, max_short_side: int) -> np.ndarray:
    if max_short_side <= 0:
        raise ValueError("Hybrid motion resolution must be positive")
    height, width = gray.shape
    short_side = min(height, width)
    if short_side <= max_short_side:
        return gray
    scale = max_short_side / float(short_side)
    target = (
        max(8, int(round(width * scale))),
        max(8, int(round(height * scale))),
    )
    return cv2.resize(gray, target, interpolation=cv2.INTER_AREA)


def _dis_flow(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Calculate source-to-target flow in source pixel coordinates."""

    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setUseSpatialPropagation(True)
    flow = estimator.calc(source, target, None).astype(np.float32)
    return np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)


def forward_backward_confidence(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    cycle_sigma: float = 1.5,
) -> np.ndarray:
    """Return confidence at forward-flow pixels using a round-trip cycle error."""

    forward = np.asarray(forward_flow, dtype=np.float32)
    backward = np.asarray(backward_flow, dtype=np.float32)
    if forward.shape != backward.shape or forward.ndim != 3 or forward.shape[2] != 2:
        raise ValueError("Hybrid motion flows must have matching HxWx2 dimensions")
    if cycle_sigma <= 0:
        raise ValueError("Hybrid motion cycle sigma must be positive")

    height, width = forward.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + forward[..., 0]
    map_y = grid_y + forward[..., 1]
    valid = (
        (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    reverse_x = cv2.remap(
        backward[..., 0], map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    reverse_y = cv2.remap(
        backward[..., 1], map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    cycle_error = np.sqrt(
        np.square(forward[..., 0] + reverse_x)
        + np.square(forward[..., 1] + reverse_y)
    )
    confidence = np.exp(-np.square(cycle_error / float(cycle_sigma)))
    confidence *= valid
    return np.clip(np.nan_to_num(confidence), 0.0, 1.0).astype(np.float32)


def calculate_bidirectional_alignment(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    max_short_side: int = 720,
) -> BidirectionalFlowAlignment:
    """Calculate one cached bidirectional DIS alignment for adjacent frames."""

    previous_full = _as_gray_u8(previous_image)
    current_full = _as_gray_u8(current_image)
    if previous_full.shape != current_full.shape:
        raise ValueError("Hybrid motion frames must have matching dimensions")
    previous = _resize_for_flow(previous_full, max_short_side)
    current = cv2.resize(
        current_full,
        (previous.shape[1], previous.shape[0]),
        interpolation=cv2.INTER_AREA,
    ) if current_full.shape != previous.shape else current_full

    previous_to_current = _dis_flow(previous, current)
    current_to_previous = _dis_flow(current, previous)
    confidence_previous = forward_backward_confidence(
        previous_to_current, current_to_previous
    )
    confidence_current = forward_backward_confidence(
        current_to_previous, previous_to_current
    )
    return BidirectionalFlowAlignment(
        flow_current_to_previous=current_to_previous,
        confidence_current=confidence_current,
        flow_previous_to_current=previous_to_current,
        confidence_previous=confidence_previous,
        full_shape=previous_full.shape,
    )


def warp_source_to_target(
    source: np.ndarray,
    target_to_source_flow: np.ndarray,
    target_confidence: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a full-resolution source array into target coordinates."""

    source = np.asarray(source, dtype=np.float32)
    flow = np.asarray(target_to_source_flow, dtype=np.float32)
    confidence = np.asarray(target_confidence, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2 or confidence.shape != flow.shape[:2]:
        raise ValueError("Hybrid motion flow/confidence dimensions are invalid")
    target_height, target_width = target_shape
    if source.shape != (target_height, target_width):
        raise ValueError("Hybrid motion source and target dimensions must match")

    small_height, small_width = flow.shape[:2]
    full_flow = cv2.resize(
        flow, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )
    full_flow[..., 0] *= target_width / float(small_width)
    full_flow[..., 1] *= target_height / float(small_height)
    full_confidence = cv2.resize(
        confidence, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(target_width, dtype=np.float32),
        np.arange(target_height, dtype=np.float32),
    )
    warped = cv2.remap(
        source,
        grid_x + full_flow[..., 0],
        grid_y + full_flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(np.float32), np.clip(full_confidence, 0.0, 1.0).astype(
        np.float32
    )


def motion_confidence_gate(
    current_residual: np.ndarray,
    previous_warped: np.ndarray,
    previous_confidence: np.ndarray,
    next_warped: np.ndarray,
    next_confidence: np.ndarray,
    agreement_sigma: float,
    previous_fallback: np.ndarray | None = None,
    next_fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gate an unstable residual and return confidence-aware aligned neighbors."""

    current = np.asarray(current_residual, dtype=np.float32)
    previous = np.asarray(previous_warped, dtype=np.float32)
    following = np.asarray(next_warped, dtype=np.float32)
    previous_conf = np.clip(previous_confidence, 0.0, 1.0).astype(np.float32)
    next_conf = np.clip(next_confidence, 0.0, 1.0).astype(np.float32)
    previous_base = previous if previous_fallback is None else np.asarray(
        previous_fallback, dtype=np.float32
    )
    next_base = following if next_fallback is None else np.asarray(
        next_fallback, dtype=np.float32
    )
    if not (
        current.shape == previous.shape == following.shape
        == previous_conf.shape == next_conf.shape == previous_base.shape
        == next_base.shape
    ):
        raise ValueError("Hybrid motion residual/confidence dimensions must match")
    if agreement_sigma <= 0:
        raise ValueError("Hybrid motion agreement sigma must be positive")

    confidence_sum = previous_conf + next_conf
    reference = (
        previous * previous_conf + following * next_conf
    ) / np.maximum(confidence_sum, 1e-6)
    support = np.clip(confidence_sum * 0.5, 0.0, 1.0)
    agreement = np.exp(
        -np.square(np.abs(current - reference) / float(agreement_sigma))
    ).astype(np.float32)
    motion_confidence = support * agreement
    # With unreliable flow support=0 and the Phase 4.1 residual passes through.
    gated = current * (1.0 - support + motion_confidence)
    # Low-confidence pixels use the original Phase 4.1 same-pixel neighbors.
    aligned_previous = previous * previous_conf + previous_base * (
        1.0 - previous_conf
    )
    aligned_next = following * next_conf + next_base * (1.0 - next_conf)
    return (
        gated.astype(np.float32),
        aligned_previous.astype(np.float32),
        aligned_next.astype(np.float32),
        np.clip(motion_confidence, 0.0, 1.0).astype(np.float32),
    )
