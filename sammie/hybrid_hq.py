"""Edge-band construction and temporal/spatial alpha integration for Hybrid HQ."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


PRESERVE_TEMPORAL = "Preserve Temporal"
BALANCED = "Balanced"
MAXIMUM_DETAIL = "Maximum Detail"


@dataclass(frozen=True)
class HybridMergePreset:
    """Parameters controlling how much frame-local MEMatte detail is accepted."""

    name: str
    residual_limit: float
    disagreement_start: float
    disagreement_end: float
    temporal_smoothing: float
    motion_agreement_sigma: float


HYBRID_MERGE_PRESETS = {
    PRESERVE_TEMPORAL: HybridMergePreset(
        PRESERVE_TEMPORAL,
        residual_limit=0.10,
        disagreement_start=0.08,
        disagreement_end=0.30,
        temporal_smoothing=0.75,
        motion_agreement_sigma=0.04,
    ),
    BALANCED: HybridMergePreset(
        BALANCED,
        residual_limit=0.20,
        disagreement_start=0.12,
        disagreement_end=0.50,
        temporal_smoothing=0.35,
        motion_agreement_sigma=0.08,
    ),
    MAXIMUM_DETAIL: HybridMergePreset(
        MAXIMUM_DETAIL,
        residual_limit=1.0,
        disagreement_start=1.0,
        disagreement_end=1.0,
        temporal_smoothing=0.0,
        motion_agreement_sigma=0.12,
    ),
}


def get_hybrid_merge_preset(name: str) -> HybridMergePreset:
    """Resolve a saved preset name, defaulting safely for old sessions."""

    return HYBRID_MERGE_PRESETS.get(name, HYBRID_MERGE_PRESETS[PRESERVE_TEMPORAL])


def alpha_to_float(alpha: np.ndarray) -> np.ndarray:
    """Normalize an 8/16-bit or floating-point alpha image to float32 0..1."""

    value = np.asarray(alpha)
    if value.ndim == 3:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError("Hybrid HQ alpha must be a 2D image")

    if np.issubdtype(value.dtype, np.integer):
        maximum = float(np.iinfo(value.dtype).max)
        normalized = value.astype(np.float32) / maximum
    else:
        normalized = value.astype(np.float32)
        finite_max = float(np.nanmax(normalized)) if normalized.size else 0.0
        if finite_max > 1.0:
            normalized /= 65535.0 if finite_max > 255.0 else 255.0
    return np.clip(np.nan_to_num(normalized), 0.0, 1.0)


def temporal_alpha_to_trimap(
    temporal_alpha: np.ndarray,
    edge_width: int = 12,
    background_threshold: float = 0.02,
    foreground_threshold: float = 0.98,
) -> np.ndarray:
    """Build a trimap from soft temporal alpha plus a hard-boundary safety band."""

    if edge_width < 0:
        raise ValueError("Hybrid HQ edge width cannot be negative")
    if not 0.0 <= background_threshold < foreground_threshold <= 1.0:
        raise ValueError("Hybrid HQ thresholds must satisfy 0 <= BG < FG <= 1")

    alpha = alpha_to_float(temporal_alpha)
    foreground = alpha >= 0.5
    unknown = (alpha > background_threshold) & (alpha < foreground_threshold)

    if edge_width > 0 and foreground.any() and (~foreground).any():
        radius = int(edge_width)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        hard_mask = foreground.astype(np.uint8)
        eroded = cv2.erode(hard_mask, kernel, borderType=cv2.BORDER_REPLICATE)
        dilated = cv2.dilate(hard_mask, kernel, borderType=cv2.BORDER_REPLICATE)
        unknown |= (dilated > 0) & (eroded == 0)

    trimap = np.zeros(alpha.shape, dtype=np.uint8)
    trimap[foreground] = 255
    trimap[unknown] = 128
    return trimap


def merge_edge_refinement(
    temporal_alpha: np.ndarray,
    spatial_alpha: np.ndarray,
    trimap: np.ndarray,
    feather_width: int = 4,
) -> np.ndarray:
    """Blend spatial detail only inside the unknown band, preserving temporal core."""

    if feather_width < 0:
        raise ValueError("Hybrid HQ feather width cannot be negative")
    temporal = alpha_to_float(temporal_alpha)
    spatial = alpha_to_float(spatial_alpha)
    trimap = np.asarray(trimap)
    if spatial.shape != temporal.shape or trimap.shape != temporal.shape:
        raise ValueError("Hybrid HQ alpha and trimap dimensions must match")

    unknown = trimap == 128
    if not unknown.any():
        return temporal.copy()

    if feather_width == 0:
        weight = unknown.astype(np.float32)
    else:
        distance = cv2.distanceTransform(unknown.astype(np.uint8), cv2.DIST_L2, 3)
        weight = np.clip(distance / float(feather_width), 0.0, 1.0)
        weight *= unknown

    merged = temporal * (1.0 - weight) + spatial * weight
    return np.clip(merged.astype(np.float32), 0.0, 1.0)


def edge_refinement_weight(trimap: np.ndarray, feather_width: int = 4) -> np.ndarray:
    """Return the legacy distance-feathered unknown-band weight."""

    if feather_width < 0:
        raise ValueError("Hybrid HQ feather width cannot be negative")
    unknown = np.asarray(trimap) == 128
    if not unknown.any():
        return np.zeros(unknown.shape, dtype=np.float32)
    if feather_width == 0:
        return unknown.astype(np.float32)
    distance = cv2.distanceTransform(unknown.astype(np.uint8), cv2.DIST_L2, 3)
    weight = np.clip(distance / float(feather_width), 0.0, 1.0)
    return (weight * unknown).astype(np.float32)


def build_edge_residual(
    temporal_alpha: np.ndarray,
    spatial_alpha: np.ndarray,
    trimap: np.ndarray,
    feather_width: int = 4,
    preset: str = PRESERVE_TEMPORAL,
) -> np.ndarray:
    """Build a bounded, confidence-gated MEMatte residual for one frame."""

    temporal = alpha_to_float(temporal_alpha)
    spatial = alpha_to_float(spatial_alpha)
    trimap = np.asarray(trimap)
    if spatial.shape != temporal.shape or trimap.shape != temporal.shape:
        raise ValueError("Hybrid HQ alpha and trimap dimensions must match")

    config = get_hybrid_merge_preset(preset)
    weight = edge_refinement_weight(trimap, feather_width)
    raw_residual = spatial - temporal
    if config.name == MAXIMUM_DETAIL:
        return (raw_residual * weight).astype(np.float32)

    magnitude = np.abs(raw_residual)
    span = config.disagreement_end - config.disagreement_start
    disagreement = np.clip(
        (magnitude - config.disagreement_start) / max(span, 1e-6), 0.0, 1.0
    )
    # Smoothstep avoids a visible confidence transition in the alpha edge.
    disagreement = disagreement * disagreement * (3.0 - 2.0 * disagreement)
    confidence = 1.0 - disagreement
    limited = np.clip(
        raw_residual, -config.residual_limit, config.residual_limit
    )
    return (limited * confidence * weight).astype(np.float32)


def stabilize_edge_residual(
    current_residual: np.ndarray,
    trimap: np.ndarray,
    preset: str = PRESERVE_TEMPORAL,
    previous_residual: np.ndarray | None = None,
    next_residual: np.ndarray | None = None,
) -> np.ndarray:
    """Suppress single-frame residual impulses with a conservative 3-frame median."""

    current = np.asarray(current_residual, dtype=np.float32)
    trimap = np.asarray(trimap)
    if current.shape != trimap.shape:
        raise ValueError("Hybrid HQ residual and trimap dimensions must match")
    config = get_hybrid_merge_preset(preset)
    if config.temporal_smoothing <= 0.0:
        return (current * (trimap == 128)).astype(np.float32)

    previous = current if previous_residual is None else np.asarray(
        previous_residual, dtype=np.float32
    )
    following = current if next_residual is None else np.asarray(
        next_residual, dtype=np.float32
    )
    if previous.shape != current.shape or following.shape != current.shape:
        raise ValueError("Hybrid HQ neighboring residual dimensions must match")

    lower = np.minimum(previous, current)
    upper = np.maximum(previous, current)
    median = np.maximum(lower, np.minimum(upper, following))
    strength = config.temporal_smoothing
    stable = current * (1.0 - strength) + median * strength
    # Neighboring residuals may never modify the current frame's known core/BG.
    return (stable * (trimap == 128)).astype(np.float32)


def apply_edge_residual(
    temporal_alpha: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    """Apply an accepted residual while preserving valid alpha range."""

    temporal = alpha_to_float(temporal_alpha)
    residual = np.asarray(residual, dtype=np.float32)
    if residual.shape != temporal.shape:
        raise ValueError("Hybrid HQ alpha and residual dimensions must match")
    return np.clip(temporal + residual, 0.0, 1.0).astype(np.float32)
