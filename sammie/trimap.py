"""Trimap generation utilities shared by image matting backends.

Sammie's segmentation masks are intentionally kept as hard masks.  This
module derives a three-class trimap without changing those source masks:

* 0   - definite background
* 128 - unknown / boundary region
* 255 - definite foreground
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class TrimapConfig:
    """Configuration for converting a hard mask into a trimap.

    ``None`` widths enable the resolution-aware automatic mode.  Widths are
    radii in pixels at the original mask resolution.
    """

    erode_width: int | None = None
    dilate_width: int | None = None
    auto_scale: float = 0.005
    min_width: int = 2
    max_width: int = 32
    foreground_threshold: float = 0.5

    def resolve_widths(self, shape: tuple[int, ...]) -> tuple[int, int]:
        if len(shape) < 2:
            raise ValueError("A trimap mask must have at least two dimensions")

        short_side = min(shape[:2])
        auto_width = int(round(short_side * self.auto_scale))
        auto_width = max(self.min_width, min(self.max_width, auto_width))

        erode = auto_width if self.erode_width is None else self.erode_width
        dilate = auto_width if self.dilate_width is None else self.dilate_width
        if erode < 0 or dilate < 0:
            raise ValueError("Trimap widths must be non-negative")
        return int(erode), int(dilate)


def _as_binary_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[..., 0]
        else:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
        cutoff = threshold
    else:
        cutoff = threshold * 255.0
    return (array > cutoff).astype(np.uint8)


def _morph(mask: np.ndarray, radius: int, operation: int) -> np.ndarray:
    if radius == 0:
        return mask.copy()
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask, operation, kernel)


def generate_trimap(
    mask: np.ndarray,
    config: TrimapConfig | None = None,
) -> np.ndarray:
    """Generate a uint8 three-class trimap from a hard segmentation mask."""

    config = config or TrimapConfig()
    binary = _as_binary_mask(mask, config.foreground_threshold)
    erode_width, dilate_width = config.resolve_widths(binary.shape)

    definite_foreground = _morph(binary, erode_width, cv2.MORPH_ERODE)
    possible_foreground = _morph(binary, dilate_width, cv2.MORPH_DILATE)

    trimap = np.zeros(binary.shape, dtype=np.uint8)
    trimap[possible_foreground > 0] = 128
    trimap[definite_foreground > 0] = 255
    return trimap


def save_trimap(
    mask: np.ndarray,
    output_path: str | Path,
    config: TrimapConfig | None = None,
) -> np.ndarray:
    """Generate and save a trimap, returning the generated array."""

    trimap = generate_trimap(mask, config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), trimap):
        raise OSError(f"Failed to write trimap: {path}")
    return trimap


def render_trimap_preview(image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    """Render RGB trimap classes over an RGB source image for UI preview."""

    if image.shape[:2] != trimap.shape[:2]:
        raise ValueError("Image and trimap dimensions must match")
    preview = (image.astype(np.float32) * 0.45).astype(np.uint8)
    class_colors = np.zeros_like(image, dtype=np.uint8)
    class_colors[trimap == 0] = (32, 64, 160)
    class_colors[trimap == 128] = (255, 176, 0)
    class_colors[trimap == 255] = (48, 208, 96)
    return cv2.addWeighted(preview, 0.55, class_colors, 0.45, 0)
