"""ViTMatte inference adapter and ROI integration utilities.

The adapter deliberately consumes original-resolution RGB frames and three-
class trimaps.  Only unknown trimap pixels are replaced by the predicted alpha;
known foreground/background pixels remain exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


class VitMatteUnavailableError(RuntimeError):
    """Raised when the optional ViTMatte runtime or checkpoint cannot load."""


@dataclass(frozen=True)
class Roi:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


def unknown_roi(trimap: np.ndarray, margin: int = 64) -> Roi | None:
    """Return a padded bounding box containing every unknown trimap pixel."""

    if margin < 0:
        raise ValueError("ROI margin must be non-negative")
    array = np.asarray(trimap)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D trimap, got shape {array.shape}")

    ys, xs = np.nonzero((array > 0) & (array < 255))
    if xs.size == 0:
        return None
    height, width = array.shape
    return Roi(
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(width, int(xs.max()) + margin + 1),
        min(height, int(ys.max()) + margin + 1),
    )


def _rois_touch(a: Roi, b: Roi) -> bool:
    return not (a.x1 < b.x0 or b.x1 < a.x0 or a.y1 < b.y0 or b.y1 < a.y0)


def _merge_rois(rois: list[Roi]) -> list[Roi]:
    merged: list[Roi] = []
    for candidate in rois:
        index = 0
        while index < len(merged):
            current = merged[index]
            if _rois_touch(candidate, current):
                candidate = Roi(
                    min(candidate.x0, current.x0),
                    min(candidate.y0, current.y0),
                    max(candidate.x1, current.x1),
                    max(candidate.y1, current.y1),
                )
                merged.pop(index)
                index = 0
            else:
                index += 1
        merged.append(candidate)
    return sorted(merged, key=lambda roi: (roi.y0, roi.x0))


def unknown_rois(trimap: np.ndarray, margin: int = 64) -> list[Roi]:
    """Return merged padded ROIs for disconnected unknown components."""

    if margin < 0:
        raise ValueError("ROI margin must be non-negative")
    array = np.asarray(trimap)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D trimap, got shape {array.shape}")
    unknown = ((array > 0) & (array < 255)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(unknown, connectivity=8)
    height, width = array.shape
    rois = []
    for label in range(1, count):
        x, y, component_width, component_height, _ = stats[label]
        rois.append(
            Roi(
                max(0, int(x) - margin),
                max(0, int(y) - margin),
                min(width, int(x + component_width) + margin),
                min(height, int(y + component_height) + margin),
            )
        )
    return _merge_rois(rois)


def split_roi(roi: Roi, max_tile_size: int, overlap: int = 128) -> list[Roi]:
    """Split one ROI into overlapping tiles that completely cover it."""

    if max_tile_size <= 0:
        return [roi]
    if overlap < 0 or overlap >= max_tile_size:
        raise ValueError("Tile overlap must be non-negative and smaller than tile size")

    def starts(begin: int, end: int) -> list[int]:
        length = end - begin
        if length <= max_tile_size:
            return [begin]
        stride = max_tile_size - overlap
        values = list(range(begin, end - max_tile_size + 1, stride))
        last = end - max_tile_size
        if values[-1] != last:
            values.append(last)
        return values

    xs = starts(roi.x0, roi.x1)
    ys = starts(roi.y0, roi.y1)
    return [
        Roi(x, y, min(x + max_tile_size, roi.x1), min(y + max_tile_size, roi.y1))
        for y in ys
        for x in xs
    ]


def _tile_weight(height: int, width: int, feather: int) -> np.ndarray:
    if feather <= 0:
        return np.ones((height, width), dtype=np.float32)
    y = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1))
    x = np.minimum(np.arange(width) + 1, np.arange(width, 0, -1))
    weight = np.minimum(y[:, None], x[None, :]).astype(np.float32)
    return np.clip(weight / min(feather, max(1, min(height, width) // 2)), 1e-3, 1.0)


def integrate_unknown_alpha(
    trimap: np.ndarray,
    predicted_alpha: np.ndarray,
    roi: Roi | None = None,
) -> np.ndarray:
    """Merge a prediction while preserving definite trimap regions exactly."""

    alpha = integrate_unknown_alpha_float(trimap, predicted_alpha, roi)
    return np.round(alpha * 255.0).astype(np.uint8)


def integrate_unknown_alpha_float(
    trimap: np.ndarray,
    predicted_alpha: np.ndarray,
    roi: Roi | None = None,
) -> np.ndarray:
    """Float32 variant used for high-precision intermediate alpha storage."""

    trimap = np.asarray(trimap, dtype=np.uint8)
    output = np.where(trimap == 255, 1.0, 0.0).astype(np.float32)
    target = trimap if roi is None else trimap[roi.slices]
    prediction = np.asarray(predicted_alpha)
    if prediction.shape != target.shape:
        prediction = cv2.resize(
            prediction.astype(np.float32),
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if not np.issubdtype(prediction.dtype, np.floating) or (
        prediction.size and float(prediction.max()) > 1.0
    ):
        scale = 65535.0 if prediction.dtype == np.uint16 else 255.0
        prediction = prediction.astype(np.float32) / scale
    prediction = np.clip(prediction, 0.0, 1.0).astype(np.float32)

    destination = output if roi is None else output[roi.slices]
    unknown = (target > 0) & (target < 255)
    destination[unknown] = prediction[unknown]
    return output


def run_tiled_unknown_inference(
    predictor,
    rgb: np.ndarray,
    trimap: np.ndarray,
    margin: int = 64,
    max_tile_size: int = 1024,
    tile_overlap: int = 128,
) -> np.ndarray:
    """Run an image-matting predictor over feathered unknown-band tiles."""

    rois = unknown_rois(trimap, margin)
    if not rois:
        return integrate_unknown_alpha_float(trimap, np.zeros_like(trimap))
    tiles = [
        tile
        for roi in rois
        for tile in split_roi(roi, max_tile_size, tile_overlap)
    ]
    accumulated = np.zeros(trimap.shape, dtype=np.float32)
    weights = np.zeros(trimap.shape, dtype=np.float32)
    feather = tile_overlap // 2
    for tile in tiles:
        rgb_crop = rgb[tile.slices]
        trimap_crop = trimap[tile.slices]
        prediction = predictor(rgb_crop, trimap_crop).astype(np.float32)
        weight = _tile_weight(*trimap_crop.shape, feather)
        unknown = (trimap_crop > 0) & (trimap_crop < 255)
        target_alpha = accumulated[tile.slices]
        target_weight = weights[tile.slices]
        target_alpha[unknown] += prediction[unknown] * weight[unknown]
        target_weight[unknown] += weight[unknown]

    prediction = np.zeros(trimap.shape, dtype=np.float32)
    valid = weights > 0
    prediction[valid] = accumulated[valid] / weights[valid]
    return integrate_unknown_alpha_float(trimap, prediction)


class VitMatteBackend:
    """Hugging Face Transformers-backed ViTMatte inference adapter."""

    MODEL_ID = "hustvl/vitmatte-small-composition-1k"

    def __init__(self, device: torch.device, model_id: str | None = None):
        self.device = device
        self.model_id = model_id or self.MODEL_ID
        self.image_processor = None
        self.model = None

    def load(self):
        try:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        except ImportError as exc:
            raise VitMatteUnavailableError(
                "ViTMatte is optional. Install it with `uv sync --extra vitmatte`."
            ) from exc

        try:
            self.image_processor = VitMatteImageProcessor.from_pretrained(self.model_id)
            self.model = VitMatteForImageMatting.from_pretrained(self.model_id)
            self.model.to(self.device).eval()
        except Exception as exc:
            raise VitMatteUnavailableError(
                f"Unable to load ViTMatte checkpoint {self.model_id!r}."
            ) from exc
        return self.model

    def predict(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        if self.model is None or self.image_processor is None:
            raise RuntimeError("ViTMatte model is not loaded")
        height, width = trimap.shape
        inputs = self.image_processor(
            images=rgb,
            trimaps=trimap,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            alphas = self.model(pixel_values=pixel_values).alphas
        alpha = alphas[0, 0, :height, :width].float().cpu().numpy()
        return np.clip(alpha, 0.0, 1.0)

    def predict_roi(
        self,
        rgb: np.ndarray,
        trimap: np.ndarray,
        margin: int = 64,
    ) -> np.ndarray:
        roi = unknown_roi(trimap, margin)
        if roi is None:
            return integrate_unknown_alpha(trimap, np.zeros_like(trimap))
        rgb_crop = rgb[roi.slices]
        trimap_crop = trimap[roi.slices]
        prediction = self.predict(rgb_crop, trimap_crop)
        return integrate_unknown_alpha(trimap, prediction, roi)

    def predict_multi_roi(
        self,
        rgb: np.ndarray,
        trimap: np.ndarray,
        margin: int = 64,
        max_tile_size: int = 1024,
        tile_overlap: int = 128,
    ) -> np.ndarray:
        """Infer disconnected/tiled unknown regions and feather tile overlaps."""

        alpha = self.predict_multi_roi_float(
            rgb, trimap, margin, max_tile_size, tile_overlap
        )
        return np.round(alpha * 255.0).astype(np.uint8)

    def predict_multi_roi_float(
        self,
        rgb: np.ndarray,
        trimap: np.ndarray,
        margin: int = 64,
        max_tile_size: int = 1024,
        tile_overlap: int = 128,
    ) -> np.ndarray:
        """High-precision multi-ROI result in the normalized 0..1 range."""

        return run_tiled_unknown_inference(
            self.predict,
            rgb,
            trimap,
            margin=margin,
            max_tile_size=max_tile_size,
            tile_overlap=tile_overlap,
        )

    def unload(self):
        self.model = None
        self.image_processor = None
