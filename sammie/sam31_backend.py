"""Adapter between Sammie's segmentation interface and Meta SAM 3.1.

SAM 3.1 uses a request/session API rather than SAM2's inference-state API.
Keeping that difference in this module lets the existing UI and mask/export
pipeline remain backend-agnostic.
"""

from __future__ import annotations

import inspect
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

    def __init__(
        self,
        device,
        frame_extension: str,
        checkpoint_path: str | Path | None = None,
    ):
        self.device = device
        self.frame_extension = frame_extension.lower()
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self.session_id: str | None = None
        self.resource_path: str | None = None
        self.sdpa_backend_mode: str | None = None

    def _resolve_checkpoint_path(self) -> Path | None:
        """Resolve an explicit, environment, or project-local checkpoint."""

        configured = self.checkpoint_path or os.environ.get("SAM31_CHECKPOINT_PATH")
        if configured:
            path = Path(configured).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            path = path.resolve()
            if not path.is_file():
                raise Sam31UnavailableError(
                    f"Configured SAM 3.1 checkpoint was not found: {path}"
                )
            return path

        project_checkpoint = (
            Path(__file__).resolve().parent.parent
            / "checkpoints"
            / "sam31"
            / "sam3.1_multiplex.pt"
        )
        return project_checkpoint if project_checkpoint.is_file() else None

    def load(self):
        if getattr(self.device, "type", str(self.device)) != "cuda":
            raise Sam31UnavailableError(
                "SAM 3.1 Object Multiplex currently requires a CUDA device. "
                "Select SAM2/EfficientTAM on CPU, MPS, or XPU."
            )
        try:
            from sam3.model_builder import build_sam3_multiplex_video_predictor
        except ImportError as exc:
            raise Sam31UnavailableError(
                "SAM 3.1 is an optional dependency. Install it with "
                "`uv sync --extra sam31` from the project directory."
            ) from exc

        self._configure_sdpa_fallback()

        checkpoint_path = self._resolve_checkpoint_path()
        try:
            # The released multiplex checkpoint stores complex-valued RoPE
            # buffers (``freqs_cis``). Real-valued RoPE is intended for the
            # compiled path and otherwise reports those checkpoint keys as
            # missing/unexpected.
            builder_args = {"use_fa3": False, "use_rope_real": False}
            if checkpoint_path is not None:
                builder_args["checkpoint_path"] = str(checkpoint_path)
                print(f"Loading local SAM 3.1 checkpoint: {checkpoint_path}")
            self.predictor = build_sam3_multiplex_video_predictor(**builder_args)
        except Exception as exc:
            if checkpoint_path is not None:
                message = (
                    f"Unable to load local SAM 3.1 checkpoint {checkpoint_path}: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                message = (
                    "Unable to download the SAM 3.1 checkpoint. Request access "
                    "to facebook/sam3.1 on Hugging Face and run `hf auth login`, "
                    f"then try again. Root cause: {type(exc).__name__}: {exc}"
                )
            raise Sam31UnavailableError(message) from exc
        return self.predictor

    def _configure_sdpa_fallback(self):
        """Avoid SAM3's flash-only SDPA context when Flash is unavailable."""
        import torch

        flash_available = getattr(
            torch.backends.cuda, "is_flash_attention_available", lambda: False
        )()
        if flash_available:
            self.sdpa_backend_mode = "flash"
            return

        import sam3.model.decoder as decoder
        from torch.nn.attention import SDPBackend

        if getattr(decoder, "_sammie_sdpa_fallback_enabled", False):
            self.sdpa_backend_mode = getattr(
                decoder, "_sammie_sdpa_backend_mode", "fallback"
            )
            return

        original_sdpa_kernel = decoder.sdpa_kernel
        fallback_backends = [
            backend
            for backend in (
                getattr(SDPBackend, "CUDNN_ATTENTION", None),
                getattr(SDPBackend, "EFFICIENT_ATTENTION", None),
                getattr(SDPBackend, "MATH", None),
            )
            if backend is not None
        ]
        if not fallback_backends:
            raise Sam31UnavailableError(
                "No compatible PyTorch SDPA backend is available for SAM 3.1."
            )

        def compatible_sdpa_kernel(_requested_backend, *args, **kwargs):
            try:
                return original_sdpa_kernel(fallback_backends, set_priority=True)
            except TypeError:
                # Compatibility with PyTorch versions before set_priority.
                return original_sdpa_kernel(fallback_backends)

        backend_names = "/".join(
            str(backend).rsplit(".", 1)[-1] for backend in fallback_backends
        )
        decoder.sdpa_kernel = compatible_sdpa_kernel
        decoder._sammie_sdpa_fallback_enabled = True
        decoder._sammie_sdpa_backend_mode = backend_names
        self.sdpa_backend_mode = backend_names
        print(
            "SAM 3.1 Flash SDPA is unavailable; using "
            f"{backend_names} fallback."
        )

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
        request = {
            "type": "start_session",
            "resource_path": self.resource_path,
            "offload_video_to_cpu": True,
        }
        response = self._handle_start_session(request)
        self.session_id = response["session_id"]

    def _handle_start_session(self, request):
        """Bridge SAM3 predictor/model init_state signature differences."""
        model = getattr(self.predictor, "model", None)
        init_state = getattr(model, "init_state", None)
        if init_state is None:
            return self.predictor.handle_request(request)

        try:
            parameters = inspect.signature(init_state).parameters
        except (TypeError, ValueError):
            return self.predictor.handle_request(request)

        accepts_extra_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_extra_kwargs or "offload_state_to_cpu" in parameters:
            return self.predictor.handle_request(request)

        supported_names = set(parameters)
        original_init_state = init_state

        def compatible_init_state(*args, **kwargs):
            supported_kwargs = {
                name: value
                for name, value in kwargs.items()
                if name in supported_names
            }
            return original_init_state(*args, **supported_kwargs)

        # Sam3BasePredictor currently always supplies offload_state_to_cpu,
        # while the released SAM 3.1 multiplex init_state does not accept it.
        # Patch only for the duration of session creation and restore it even
        # if initialization fails.
        model.init_state = compatible_init_state
        try:
            return self.predictor.handle_request(request)
        finally:
            model.init_state = original_init_state

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
