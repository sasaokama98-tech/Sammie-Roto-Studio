"""Shot-wide, inference-only pregrade for segmentation and matting.

The source footage is never modified. A bounded, shared adjustment is estimated
from a few frames rather than re-estimating each frame (which would flicker).
"""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import os

import cv2
import numpy as np


@dataclass(frozen=True)
class Pregrade:
    gain: float = 1.0
    contrast: float = 1.0
    pivot: float = 0.5


def estimate_pregrade(frame_paths: list[str]) -> Pregrade:
    """Estimate one conservative luma adjustment for an entire sequence."""
    if not frame_paths:
        raise ValueError("No frames available for auto pregrade")
    sample_indices = np.linspace(0, len(frame_paths) - 1, min(7, len(frame_paths)), dtype=int)
    values = []
    for index in sorted(set(sample_indices.tolist())):
        frame = cv2.imread(frame_paths[index], cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"Cannot read pregrade sample: {frame_paths[index]}")
        height, width = frame.shape[:2]
        scale = min(1.0, 512.0 / max(height, width))
        if scale < 1.0:
            frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        active = luma[luma > 4]  # Ignore letterbox and pure-black pixels.
        if active.size:
            values.append(active[::max(1, active.size // 30000)])
    if not values:
        return Pregrade()
    p10, p50, p90, p98 = np.percentile(np.concatenate(values), [10, 50, 90, 98])
    if p50 < 8:
        return Pregrade()
    gain = float(np.clip(110.0 / p50, 0.7, 1.65))
    gain = min(gain, float(245.0 / max(p98, 1.0)))
    contrast = float(np.clip(150.0 / max(p90 - p10, 25.0), 0.9, 1.18))
    return Pregrade(gain=gain, contrast=contrast, pivot=float(p50 / 255.0))


def apply_pregrade(frame: np.ndarray, grade: Pregrade) -> np.ndarray:
    """Apply the same neutral transform to every BGR frame."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Auto pregrade expects an 8-bit BGR frame")
    pixels = frame.astype(np.float32) / 255.0
    pixels = (pixels - grade.pivot) * grade.contrast + grade.pivot
    return np.clip(pixels * grade.gain * 255.0, 0, 255).astype(np.uint8)


@lru_cache(maxsize=8)
def _estimate_cached(source_dir: str, extension: str, total_frames: int,
                     first_mtime: int, last_mtime: int) -> Pregrade:
    frame_paths = [str(Path(source_dir) / f"{index:05d}.{extension}") for index in range(total_frames)]
    return estimate_pregrade(frame_paths)


def _stage_cached(source_dir: str, target_root: str, extension: str, total_frames: int,
                  first_mtime: int, last_mtime: int) -> str:
    frame_paths = [str(Path(source_dir) / f"{index:05d}.{extension}") for index in range(total_frames)]
    missing = [path for path in frame_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Auto pregrade source frame is missing: {missing[0]}")
    grade = _estimate_cached(source_dir, extension, total_frames, first_mtime, last_mtime)
    signature = sha256(repr((source_dir, extension, total_frames, first_mtime,
                            last_mtime, grade)).encode("utf-8")).hexdigest()[:16]
    target = Path(target_root) / signature
    target.mkdir(parents=True, exist_ok=True)
    pending = target / "_pending"
    pending.mkdir(exist_ok=True)
    staged = 0
    for index, path in enumerate(frame_paths):
        destination = target / f"{index:05d}.{extension}"
        if destination.is_file():
            continue
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"Cannot read pregrade frame: {path}")
        options = [cv2.IMWRITE_JPEG_QUALITY, 95] if extension in {"jpg", "jpeg"} else []
        temporary = pending / destination.name
        if not cv2.imwrite(str(temporary), apply_pregrade(frame, grade), options):
            raise OSError(f"Cannot write pregrade frame: {destination}")
        os.replace(temporary, destination)
        staged += 1
    if staged:
        print(f"Auto pregrade staged {staged}/{total_frames} inference-only frames (gain={grade.gain:.2f}, contrast={grade.contrast:.2f})")
    return str(target)


def inference_frames_dir(source_dir: str, temp_dir: str, extension: str,
                         total_frames: int, stage: str, enabled: bool) -> str:
    """Return source frames or a shot-consistent pregraded staging directory."""
    if not enabled or total_frames <= 0:
        return source_dir
    if stage not in {"segmentation", "matting"}:
        raise ValueError(f"Unsupported auto pregrade stage: {stage}")
    source = Path(source_dir).resolve()
    first = source / f"{0:05d}.{extension}"
    last = source / f"{total_frames - 1:05d}.{extension}"
    # Both model stages share identical frames instead of duplicating a 4K
    # sequence on disk.
    return _stage_cached(str(source), str(Path(temp_dir).resolve() / "auto_pregrade_frames"),
                         extension, total_frames, first.stat().st_mtime_ns,
                         last.stat().st_mtime_ns)
