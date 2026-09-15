"""Adapter between Sammie's segmentation interface and Meta SAM 3.1.

SAM 3.1 uses a request/session API rather than SAM2's inference-state API.
Keeping that difference in this module lets the existing UI and mask/export
pipeline remain backend-agnostic.
"""

from __future__ import annotations

import ast
import io
import inspect
import os
import re
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class Sam31UnavailableError(RuntimeError):
    """Raised when the optional SAM 3.1 runtime cannot be loaded."""


class _CheckpointKeyOutputFilter(io.TextIOBase):
    """Suppress verbose non-strict key lists while preserving normal output."""

    _KEY_LINE = re.compile(
        r"^(Missing|Unexpected) keys(?: \((\d+)\))?:\s*(.*)$"
    )

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.buffer = ""
        self.missing_count = 0
        self.unexpected_count = 0
        self.report_count = 0

    @staticmethod
    def _payload_count(payload: str) -> int:
        value = payload.strip()
        if value.endswith("..."):
            value = value[:-3]
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return 0
        return len(parsed) if isinstance(parsed, (list, tuple)) else 0

    def _process_line(self, line: str, newline: bool = True):
        match = self._KEY_LINE.match(line.strip())
        if match is None:
            self.target.write(line + ("\n" if newline else ""))
            return
        kind, explicit_count, payload = match.groups()
        count = int(explicit_count) if explicit_count else self._payload_count(payload)
        if kind == "Missing":
            self.missing_count += count
        else:
            self.unexpected_count += count
        self.report_count += 1

    def write(self, value):
        if not isinstance(value, str):
            value = str(value)
        self.buffer += value
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._process_line(line)
        return len(value)

    def flush(self):
        self.target.flush()

    def finish(self):
        if self.buffer:
            self._process_line(self.buffer, newline=False)
            self.buffer = ""
        self.target.flush()

    def summary(self) -> str | None:
        if not self.report_count:
            return None
        return (
            "SAM 3.1 checkpoint compatibility: "
            f"{self.report_count} non-strict key lists suppressed "
            "(set "
            "SAM31_VERBOSE_CHECKPOINT_KEYS=1 to show them)."
        )


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
        self.frame_size: tuple[int, int] | None = None
        self.sdpa_backend_mode: str | None = None
        self.prompt_preview_session_id: str | None = None
        self.prompt_preview_text: str = ""
        self.prompt_preview_frame: int | None = None
        self.prompt_candidates: list[dict] = []
        self.prompt_seed: dict | None = None
        self.prompt_seed_active = False
        self.internal_to_studio: dict[int, int] = {}
        self.studio_to_internal: dict[int, int] = {}

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
            verbose_keys = os.environ.get(
                "SAM31_VERBOSE_CHECKPOINT_KEYS", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            if verbose_keys:
                self.predictor = build_sam3_multiplex_video_predictor(
                    **builder_args
                )
            else:
                output_filter = _CheckpointKeyOutputFilter(sys.stdout)
                try:
                    with redirect_stdout(output_filter):
                        self.predictor = build_sam3_multiplex_video_predictor(
                            **builder_args
                        )
                finally:
                    output_filter.finish()
                    summary = output_filter.summary()
                    if summary:
                        print(summary)
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
        self.discard_prompt_preview()
        self.close_session()
        self.resource_path = self._prepare_resource(frames_dir)
        self.frame_size = self._read_frame_size(self.resource_path)
        request = {
            "type": "start_session",
            "resource_path": self.resource_path,
            "offload_video_to_cpu": True,
        }
        response = self._handle_start_session(request)
        self.session_id = response["session_id"]
        self.prompt_seed_active = False
        self._reset_object_maps()

    @staticmethod
    def _read_frame_size(resource_path: str) -> tuple[int, int]:
        """Read the original UI coordinate space from the staged frame set."""

        resource = Path(resource_path)
        frame_paths = sorted(
            path
            for path in resource.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not frame_paths:
            raise FileNotFoundError(f"No readable frames found in {resource}")
        frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"Failed to read SAM 3.1 frame dimensions: {frame_paths[0]}")
        height, width = frame.shape[:2]
        return width, height

    @staticmethod
    def normalize_points(points, frame_size: tuple[int, int]) -> np.ndarray:
        """Map original-frame pixel coordinates to SAM 3.1 relative coordinates."""

        coordinates = np.asarray(points, dtype=np.float32)
        if coordinates.ndim == 1 and coordinates.size == 2:
            coordinates = coordinates.reshape(1, 2)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("SAM 3.1 points must have shape Nx2")
        if not np.isfinite(coordinates).all():
            raise ValueError("SAM 3.1 points must contain finite coordinates")

        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid SAM 3.1 frame size: {frame_size}")
        if (
            (coordinates[:, 0] < 0).any()
            or (coordinates[:, 0] > width).any()
            or (coordinates[:, 1] < 0).any()
            or (coordinates[:, 1] > height).any()
        ):
            raise ValueError(
                f"SAM 3.1 point lies outside the {width}x{height} source frame"
            )

        scale = np.asarray([width, height], dtype=np.float32)
        return np.clip(coordinates / scale, 0.0, 1.0)

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

    @staticmethod
    def _candidates_from_outputs(outputs: dict) -> list[dict]:
        masks = Sam31Backend.masks_from_outputs(outputs)
        probabilities = np.asarray(outputs.get("out_probs", []), dtype=np.float32)
        boxes = np.asarray(outputs.get("out_boxes_xywh", []), dtype=np.float32)
        candidates = []
        for index, (object_id, mask) in enumerate(masks):
            score = float(probabilities[index]) if index < len(probabilities) else None
            box = boxes[index].tolist() if index < len(boxes) else None
            candidates.append(
                {
                    "candidate_index": index,
                    "candidate_id": int(object_id),
                    "score": score,
                    "box_xywh": box,
                    "area": int(np.count_nonzero(mask)),
                    "mask": mask,
                }
            )
        return candidates

    def _close_session_id(self, session_id: str | None):
        if self.predictor is None or session_id is None:
            return
        self.predictor.handle_request(
            {
                "type": "close_session",
                "session_id": session_id,
                "run_gc_collect": False,
            }
        )

    def discard_prompt_preview(self):
        preview_session_id = self.prompt_preview_session_id
        self.prompt_preview_session_id = None
        self.prompt_preview_text = ""
        self.prompt_preview_frame = None
        self.prompt_candidates = []
        if preview_session_id is not None and preview_session_id != self.session_id:
            self._close_session_id(preview_session_id)

    def preview_text_prompt(self, frame_number: int, text: str) -> list[dict]:
        """Evaluate a semantic prompt without mutating the active point session."""
        if self.predictor is None or self.session_id is None or not self.resource_path:
            raise RuntimeError("SAM 3.1 session is not initialized")
        prompt = text.strip()
        if not prompt:
            raise ValueError("SAM 3.1 prompt text cannot be empty")

        self.discard_prompt_preview()
        response = self._handle_start_session(
            {
                "type": "start_session",
                "resource_path": self.resource_path,
                "offload_video_to_cpu": True,
            }
        )
        preview_session_id = response["session_id"]
        try:
            response = self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": preview_session_id,
                    "frame_index": int(frame_number),
                    "text": prompt,
                }
            )
            candidates = self._candidates_from_outputs(response["outputs"])
        except Exception:
            self._close_session_id(preview_session_id)
            raise

        self.prompt_preview_session_id = preview_session_id
        self.prompt_preview_text = prompt
        self.prompt_preview_frame = int(frame_number)
        self.prompt_candidates = candidates
        return candidates

    def _reset_object_maps(self):
        self.internal_to_studio = {}
        self.studio_to_internal = {}
        if not self.prompt_seed:
            return
        for mapping in self.prompt_seed.get("mappings", []):
            internal_id = int(mapping["candidate_id"])
            studio_id = int(mapping["object_id"])
            self.internal_to_studio[internal_id] = studio_id
            self.studio_to_internal[studio_id] = internal_id

    def configure_prompt_seed(self, seed: dict | None):
        """Configure a committed semantic seed loaded from session metadata."""
        if not seed or not seed.get("text") or seed.get("frame") is None:
            self.prompt_seed = None
        else:
            mappings = [dict(mapping) for mapping in seed.get("mappings", [])]
            self.prompt_seed = {
                "text": str(seed["text"]),
                "frame": int(seed["frame"]),
                "mappings": mappings,
            }
        self.prompt_seed_active = False
        self._reset_object_maps()

    def prompt_seed_metadata(self) -> dict | None:
        if not self.prompt_seed:
            return None
        return {
            "text": self.prompt_seed["text"],
            "frame": int(self.prompt_seed["frame"]),
            "mappings": [dict(mapping) for mapping in self.prompt_seed["mappings"]],
        }

    def clear_prompt_seed(self):
        self.discard_prompt_preview()
        self.prompt_seed = None
        self.prompt_seed_active = False
        self.internal_to_studio = {}
        self.studio_to_internal = {}

    def _remove_internal_object(self, internal_id: int, frame_number: int):
        return self.predictor.handle_request(
            {
                "type": "remove_object",
                "session_id": self.session_id,
                "frame_index": int(frame_number),
                "obj_id": int(internal_id),
            }
        )

    def commit_prompt_candidates(
        self, candidate_ids: list[int], studio_object_ids: list[int]
    ) -> list[tuple[int, np.ndarray]]:
        """Adopt the isolated prompt session and map accepted candidates to Studio IDs."""
        if self.prompt_preview_session_id is None:
            raise RuntimeError("No SAM 3.1 prompt preview is available")
        if not candidate_ids or len(candidate_ids) != len(studio_object_ids):
            raise ValueError("Select at least one prompt candidate and matching object ID")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Prompt candidates must be unique")
        if len(set(studio_object_ids)) != len(studio_object_ids):
            raise ValueError("Studio object IDs must be unique")

        candidates_by_id = {
            int(candidate["candidate_id"]): candidate
            for candidate in self.prompt_candidates
        }
        missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidates_by_id]
        if missing:
            raise ValueError(f"Unknown SAM 3.1 prompt candidate IDs: {missing}")

        old_session_id = self.session_id
        self.session_id = self.prompt_preview_session_id
        self.prompt_preview_session_id = None
        if old_session_id is not None and old_session_id != self.session_id:
            self._close_session_id(old_session_id)

        selected = set(int(candidate_id) for candidate_id in candidate_ids)
        for candidate in self.prompt_candidates:
            candidate_id = int(candidate["candidate_id"])
            if candidate_id not in selected:
                self._remove_internal_object(candidate_id, self.prompt_preview_frame or 0)

        mappings = []
        masks = []
        for candidate_id, studio_id in zip(candidate_ids, studio_object_ids):
            candidate = candidates_by_id[int(candidate_id)]
            mappings.append(
                {
                    "candidate_index": int(candidate["candidate_index"]),
                    "candidate_id": int(candidate_id),
                    "object_id": int(studio_id),
                    "score": candidate["score"],
                }
            )
            masks.append((int(studio_id), candidate["mask"]))

        self.prompt_seed = {
            "text": self.prompt_preview_text,
            "frame": int(self.prompt_preview_frame),
            "mappings": mappings,
        }
        self.prompt_seed_active = True
        self._reset_object_maps()
        self.prompt_preview_text = ""
        self.prompt_preview_frame = None
        self.prompt_candidates = []
        return masks

    def _restore_prompt_seed(self) -> list[tuple[int, np.ndarray]]:
        if not self.prompt_seed:
            return []
        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self.session_id,
                "frame_index": int(self.prompt_seed["frame"]),
                "text": self.prompt_seed["text"],
            }
        )
        candidates = self._candidates_from_outputs(response["outputs"])
        by_id = {int(candidate["candidate_id"]): candidate for candidate in candidates}
        selected_candidates = []
        for mapping in self.prompt_seed["mappings"]:
            candidate = by_id.get(int(mapping["candidate_id"]))
            if candidate is None:
                index = int(mapping.get("candidate_index", -1))
                if index < 0 or index >= len(candidates):
                    raise RuntimeError(
                        "SAM 3.1 prompt candidates changed and the saved object mapping "
                        "cannot be restored deterministically"
                    )
                candidate = candidates[index]
                mapping["candidate_id"] = int(candidate["candidate_id"])
            selected_candidates.append(candidate)

        selected_ids = {int(candidate["candidate_id"]) for candidate in selected_candidates}
        for candidate in candidates:
            candidate_id = int(candidate["candidate_id"])
            if candidate_id not in selected_ids:
                self._remove_internal_object(candidate_id, int(self.prompt_seed["frame"]))

        self.prompt_seed_active = True
        self._reset_object_maps()
        return [
            (int(mapping["object_id"]), candidate["mask"])
            for mapping, candidate in zip(self.prompt_seed["mappings"], selected_candidates)
        ]

    def ensure_prompt_seed(self):
        if self.prompt_seed and not self.prompt_seed_active:
            return self._restore_prompt_seed()
        return []

    def retain_prompt_objects(self, studio_object_ids: set[int]):
        """Drop committed prompt mappings whose Studio object has no points."""
        if not self.prompt_seed:
            return
        retained = []
        for mapping in self.prompt_seed["mappings"]:
            studio_id = int(mapping["object_id"])
            if studio_id in studio_object_ids:
                retained.append(mapping)
            elif self.prompt_seed_active:
                self._remove_internal_object(
                    int(mapping["candidate_id"]), int(self.prompt_seed["frame"])
                )
        self.prompt_seed["mappings"] = retained
        if not retained:
            self.clear_prompt_seed()
        else:
            self._reset_object_maps()

    def _internal_object_id(self, studio_object_id: int) -> int:
        studio_object_id = int(studio_object_id)
        if studio_object_id in self.studio_to_internal:
            return self.studio_to_internal[studio_object_id]
        used = set(self.internal_to_studio)
        internal_id = studio_object_id
        if internal_id in used:
            internal_id = 0
            while internal_id in used:
                internal_id += 1
        self.internal_to_studio[internal_id] = studio_object_id
        self.studio_to_internal[studio_object_id] = internal_id
        return internal_id

    def _mapped_masks_from_outputs(self, outputs: dict) -> list[tuple[int, np.ndarray]]:
        mapped = []
        for internal_id, mask in self.masks_from_outputs(outputs):
            if internal_id in self.internal_to_studio:
                mapped.append((self.internal_to_studio[internal_id], mask))
            elif not self.prompt_seed:
                mapped.append((internal_id, mask))
        return mapped

    @staticmethod
    def _has_nonempty_object_mask(
        masks: list[tuple[int, np.ndarray]], object_id: int
    ) -> bool:
        return any(
            int(output_id) == int(object_id) and np.asarray(mask).any()
            for output_id, mask in masks
        )

    def _submit_point_prompt(
        self,
        frame_number: int,
        internal_object_id: int,
        normalized_points: np.ndarray,
        point_labels: np.ndarray,
        clear_old_points: bool,
    ) -> list[tuple[int, np.ndarray]]:
        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self.session_id,
                "frame_index": frame_number,
                "points": normalized_points,
                "point_labels": point_labels,
                "obj_id": internal_object_id,
                "clear_old_points": clear_old_points,
                "rel_coordinates": True,
            }
        )
        return self._mapped_masks_from_outputs(response["outputs"])

    def _retry_single_positive_point(
        self,
        frame_number: int,
        object_id: int,
        internal_object_id: int,
        normalized_points: np.ndarray,
        point_labels: np.ndarray,
        clear_old_points: bool,
    ) -> list[tuple[int, np.ndarray]]:
        """Recover when the first-click multimask candidate is empty.

        The official tracker intentionally enables multimask only for the first
        click. Some checkpoint/runtime combinations can rank an empty candidate
        highest. Retry the same authored point through the single-mask token,
        then use an adjacent-pixel positive point only if that still fails.
        """

        model = getattr(self.predictor, "model", None)
        tracker = getattr(model, "tracker", None)
        if tracker is not None and hasattr(tracker, "multimask_output_in_sam"):
            original_multimask = tracker.multimask_output_in_sam
            try:
                tracker.multimask_output_in_sam = False
                masks = self._submit_point_prompt(
                    frame_number,
                    internal_object_id,
                    normalized_points,
                    point_labels,
                    clear_old_points,
                )
            finally:
                tracker.multimask_output_in_sam = original_multimask
            if self._has_nonempty_object_mask(masks, object_id):
                print(
                    "SAM 3.1 single-point recovery: used the single-mask "
                    f"decoder for frame {frame_number}, object {object_id}."
                )
                return masks

        width, height = self.frame_size
        one_pixel = np.asarray(
            [1.0 / max(width, 1), 1.0 / max(height, 1)], dtype=np.float32
        )
        direction = np.where(normalized_points[0] >= 0.5, -1.0, 1.0)
        adjacent_point = np.clip(
            normalized_points[0] + one_pixel * direction, 0.0, 1.0
        )
        retry_points = np.vstack((normalized_points, adjacent_point)).astype(
            np.float32
        )
        retry_labels = np.asarray([1, 1], dtype=np.int32)
        masks = self._submit_point_prompt(
            frame_number,
            internal_object_id,
            retry_points,
            retry_labels,
            clear_old_points,
        )
        if self._has_nonempty_object_mask(masks, object_id):
            print(
                "SAM 3.1 single-point recovery: used an adjacent-pixel "
                f"positive seed for frame {frame_number}, object {object_id}."
            )
        else:
            print(
                "SAM 3.1 single positive point produced an empty mask after "
                f"recovery attempts on frame {frame_number}, object {object_id}."
            )
            raise RuntimeError(
                "SAM 3.1 could not create a mask from the positive point. "
                "Move the point farther inside the object or add another "
                "positive point."
            )
        return masks

    def add_points(
        self,
        frame_number: int,
        object_id: int,
        points,
        labels,
        clear_old_points: bool = True,
    ) -> list[tuple[int, np.ndarray]]:
        if self.frame_size is None:
            raise RuntimeError("SAM 3.1 session frame size is not initialized")
        normalized_points = self.normalize_points(points, self.frame_size)
        point_labels = np.asarray(labels, dtype=np.int32)
        if point_labels.ndim != 1 or len(point_labels) != len(normalized_points):
            raise ValueError("SAM 3.1 point labels must match the point count")
        self.ensure_prompt_seed()
        internal_object_id = self._internal_object_id(object_id)
        masks = self._submit_point_prompt(
            frame_number,
            internal_object_id,
            normalized_points,
            point_labels,
            clear_old_points,
        )
        is_single_positive = (
            len(normalized_points) == 1
            and len(point_labels) == 1
            and int(point_labels[0]) == 1
        )
        if is_single_positive and not self._has_nonempty_object_mask(
            masks, object_id
        ):
            masks = self._retry_single_positive_point(
                frame_number,
                object_id,
                internal_object_id,
                normalized_points,
                point_labels,
                clear_old_points,
            )
        return masks

    def propagate(
        self,
        start_frame_idx: int,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> Iterable[tuple[int, list[tuple[int, np.ndarray]]]]:
        self.ensure_prompt_seed()
        direction = "backward" if reverse else "forward"
        request = {
            "type": "propagate_in_video",
            "session_id": self.session_id,
            "propagation_direction": direction,
            "start_frame_index": start_frame_idx,
            "max_frame_num_to_track": max_frame_num_to_track,
        }
        for response in self.predictor.handle_stream_request(request):
            yield response["frame_index"], self._mapped_masks_from_outputs(response["outputs"])

    def reset(self):
        if self.predictor is not None and self.session_id is not None:
            self.predictor.handle_request(
                {"type": "reset_session", "session_id": self.session_id}
            )
        self.prompt_seed_active = False
        self._reset_object_maps()

    def remove_object(self, object_id: int, frame_number: int = 0):
        if self.predictor is not None and self.session_id is not None:
            internal_id = self.studio_to_internal.get(int(object_id), int(object_id))
            response = self._remove_internal_object(internal_id, frame_number)
            self.internal_to_studio.pop(internal_id, None)
            self.studio_to_internal.pop(int(object_id), None)
            if self.prompt_seed:
                self.prompt_seed["mappings"] = [
                    mapping
                    for mapping in self.prompt_seed["mappings"]
                    if int(mapping["object_id"]) != int(object_id)
                ]
                if not self.prompt_seed["mappings"]:
                    self.clear_prompt_seed()
            return response
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
                self.prompt_seed_active = False

    def unload(self):
        self.discard_prompt_preview()
        self.close_session()
        self.predictor = None
        if self.resource_path and os.path.basename(self.resource_path) == "sam31_frames":
            shutil.rmtree(self.resource_path, ignore_errors=True)
        self.resource_path = None
        self.frame_size = None
