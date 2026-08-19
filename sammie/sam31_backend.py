"""Adapter between Sammie's segmentation interface and Meta SAM 3.1.

SAM 3.1 uses a request/session API rather than SAM2's inference-state API.
Keeping that difference in this module lets the existing UI and mask/export
pipeline remain backend-agnostic.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class Sam31UnavailableError(RuntimeError):
    """Raised when the optional SAM 3.1 runtime cannot be loaded."""


class Sam31Backend:
    MODEL_NAME = "SAM 3.1"

    def __init__(self, device, frame_extension: str):
        self.device = device
        self.frame_extension = frame_extension.lower()
        self.predictor = None
        self.session_id: str | None = None
        self.resource_path: str | None = None

    def load(self):
        if getattr(self.device, "type", str(self.device)) != "cuda":
            raise Sam31UnavailableError(
                "SAM 3.1 Object Multiplex currently requires a CUDA device. "
                "Select SAM2/EfficientTAM on CPU, MPS, or XPU."
            )
        if os.name == "nt":
            import torch

            capability = torch.cuda.get_device_capability()
            if capability[0] >= 12:
                print(
                    "Warning: SAM 3.1 multiplex propagation has an unresolved "
                    "upstream kernel issue on Windows with RTX 50-series GPUs. "
                    "Linux or WSL2 is recommended for production use."
                )

        try:
            from sam3.model_builder import build_sam3_multiplex_video_predictor
        except ImportError as exc:
            raise Sam31UnavailableError(
                "SAM 3.1 is an optional dependency. Install it with "
                "`uv sync --extra sam31` from the project directory."
            ) from exc

        try:
            # The official builder downloads the gated SAM 3.1 checkpoint from
            # Hugging Face when no explicit checkpoint is supplied.
            self.predictor = build_sam3_multiplex_video_predictor(use_fa3=False)
        except Exception as exc:
            raise Sam31UnavailableError(
                "Unable to load the SAM 3.1 checkpoint. Request access to "
                "facebook/sam3 on Hugging Face and run `hf auth login`, then "
                "try again."
            ) from exc
        return self.predictor

    def _prepare_resource(self, frames_dir: str) -> str:
        """Return a JPEG frame directory accepted by the official predictor."""

        if self.frame_extension in {"jpg", "jpeg"}:
            return frames_dir

        source = Path(frames_dir)
        target = source.parent / "sam31_frames"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        frame_paths = sorted(source.glob(f"*.{self.frame_extension}"))
        if not frame_paths:
            raise FileNotFoundError(f"No {self.frame_extension} frames found in {source}")
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise OSError(f"Failed to read frame for SAM 3.1: {frame_path}")
            output_path = target / f"{frame_path.stem}.jpg"
            if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Failed to stage frame for SAM 3.1: {output_path}")
        return str(target)

    def start_session(self, frames_dir: str):
        if self.predictor is None:
            raise RuntimeError("SAM 3.1 model is not loaded")
        self.close_session()
        self.resource_path = self._prepare_resource(frames_dir)
        response = self.predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": self.resource_path,
                "offload_video_to_cpu": True,
            }
        )
        self.session_id = response["session_id"]

    @staticmethod
    def masks_from_outputs(outputs: dict) -> list[tuple[int, np.ndarray]]:
        object_ids = np.asarray(outputs.get("out_obj_ids", []))
        masks = np.asarray(outputs.get("out_binary_masks", []))
        if masks.ndim == 2:
            masks = masks[None, ...]
        result = []
        for object_id, mask in zip(object_ids.tolist(), masks):
            mask = np.squeeze(np.asarray(mask, dtype=bool))
            if mask.ndim != 2:
                raise ValueError(
                    f"Unexpected SAM 3.1 mask shape for object {object_id}: {mask.shape}"
                )
            result.append((int(object_id), mask))
        return result

    def add_points(
        self,
        frame_number: int,
        object_id: int,
        points,
        labels,
        clear_old_points: bool = True,
    ) -> list[tuple[int, np.ndarray]]:
        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self.session_id,
                "frame_index": frame_number,
                "points": np.asarray(points, dtype=np.float32),
                "point_labels": np.asarray(labels, dtype=np.int32),
                "obj_id": object_id,
                "clear_old_points": clear_old_points,
                "rel_coordinates": False,
            }
        )
        return self.masks_from_outputs(response["outputs"])

    def propagate(
        self,
        start_frame_idx: int,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> Iterable[tuple[int, list[tuple[int, np.ndarray]]]]:
        direction = "backward" if reverse else "forward"
        request = {
            "type": "propagate_in_video",
            "session_id": self.session_id,
            "propagation_direction": direction,
            "start_frame_index": start_frame_idx,
            "max_frame_num_to_track": max_frame_num_to_track,
        }
        for response in self.predictor.handle_stream_request(request):
            yield response["frame_index"], self.masks_from_outputs(response["outputs"])

    def reset(self):
        if self.predictor is not None and self.session_id is not None:
            self.predictor.handle_request(
                {"type": "reset_session", "session_id": self.session_id}
            )

    def remove_object(self, object_id: int, frame_number: int = 0):
        if self.predictor is not None and self.session_id is not None:
            return self.predictor.handle_request(
                {
                    "type": "remove_object",
                    "session_id": self.session_id,
                    "frame_index": frame_number,
                    "obj_id": object_id,
                }
            )
        return None

    def close_session(self):
        if self.predictor is not None and self.session_id is not None:
            try:
                self.predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": self.session_id,
                        "run_gc_collect": False,
                    }
                )
            finally:
                self.session_id = None

    def unload(self):
        self.close_session()
        self.predictor = None
        if self.resource_path and os.path.basename(self.resource_path) == "sam31_frames":
            shutil.rmtree(self.resource_path, ignore_errors=True)
        self.resource_path = None
