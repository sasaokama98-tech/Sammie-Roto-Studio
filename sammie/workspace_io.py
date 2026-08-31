"""Transactional workspace preparation and frame-sequence validation."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import cv2

from sammie import core
from sammie.media_input import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class ValidatedFrameWorkspace:
    frame_count: int
    width: int
    height: int
    frame_format: str


def workspace_paths(workspace_dir: str) -> dict[str, str]:
    return {
        "root": workspace_dir,
        "frames": os.path.join(workspace_dir, "frames"),
        "masks": os.path.join(workspace_dir, "masks"),
        "trimaps": os.path.join(workspace_dir, "trimaps"),
        "matting": os.path.join(workspace_dir, "matting"),
        "removal": os.path.join(workspace_dir, "removal"),
    }


def create_load_workspace(live_workspace: str) -> str:
    """Create an empty staging workspace beside the live workspace."""
    live_path = os.path.abspath(live_workspace)
    parent = os.path.dirname(live_path)
    name = os.path.basename(live_path)
    staging = os.path.join(parent, f".{name}.load-{uuid.uuid4().hex}")
    paths = workspace_paths(staging)
    for key in ("frames", "masks", "trimaps", "matting", "removal"):
        os.makedirs(paths[key], exist_ok=False)
    return staging


def discard_workspace(workspace_dir: str | None) -> None:
    if workspace_dir and os.path.exists(workspace_dir):
        core.remove_tree(workspace_dir)


def validate_frame_workspace(
    workspace_dir: str, expected_count: int
) -> ValidatedFrameWorkspace:
    """Require exactly one readable, consistently sized file per frame index."""
    if expected_count <= 0:
        raise ValueError("A loaded workspace must contain at least one frame")

    frames_dir = workspace_paths(workspace_dir)["frames"]
    if not os.path.isdir(frames_dir):
        raise RuntimeError("Loaded workspace has no frames directory")

    indexed_files = {}
    for entry in os.scandir(frames_dir):
        if not entry.is_file():
            continue
        stem, extension = os.path.splitext(entry.name)
        if extension.lower() not in IMAGE_EXTENSIONS or not stem.isdigit():
            continue
        index = int(stem)
        if index in indexed_files:
            raise RuntimeError(f"Duplicate loaded frame index: {index}")
        indexed_files[index] = entry.path

    expected_indexes = set(range(expected_count))
    actual_indexes = set(indexed_files)
    if actual_indexes != expected_indexes:
        missing = sorted(expected_indexes - actual_indexes)
        extra = sorted(actual_indexes - expected_indexes)
        details = []
        if missing:
            details.append(f"missing {missing[:5]}")
        if extra:
            details.append(f"unexpected {extra[:5]}")
        raise RuntimeError(
            "Loaded frame sequence is incomplete: " + ", ".join(details)
        )

    width = height = None
    extensions = set()
    for index in range(expected_count):
        path = indexed_files[index]
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError(f"Loaded frame is unreadable: {path}")
        frame_height, frame_width = frame.shape[:2]
        if width is None:
            width, height = frame_width, frame_height
        elif (frame_width, frame_height) != (width, height):
            raise RuntimeError(
                f"Loaded frame {index} has size {frame_width}x{frame_height}; "
                f"expected {width}x{height}"
            )
        extensions.add(os.path.splitext(path)[1].lower().lstrip("."))

    if len(extensions) != 1:
        raise RuntimeError(
            f"Loaded workspace uses mixed frame formats: {sorted(extensions)}"
        )

    return ValidatedFrameWorkspace(
        frame_count=expected_count,
        width=int(width),
        height=int(height),
        frame_format=next(iter(extensions)),
    )


def swap_workspace(staging_workspace: str, live_workspace: str) -> str | None:
    """Swap a validated staging workspace into place, retaining a rollback copy."""
    staging = os.path.abspath(staging_workspace)
    live = os.path.abspath(live_workspace)
    if os.path.dirname(staging) != os.path.dirname(live):
        raise ValueError("Staging and live workspaces must be siblings")
    if not os.path.isdir(staging):
        raise FileNotFoundError(staging)

    backup = None
    if os.path.exists(live):
        backup = f"{live}.previous-{uuid.uuid4().hex}"
        os.replace(live, backup)
    try:
        os.replace(staging, live)
    except Exception:
        if backup and os.path.exists(backup) and not os.path.exists(live):
            os.replace(backup, live)
        raise
    return backup


def finalize_workspace_swap(backup_workspace: str | None) -> None:
    discard_workspace(backup_workspace)


def rollback_workspace_swap(
    live_workspace: str, backup_workspace: str | None
) -> None:
    discard_workspace(live_workspace)
    if backup_workspace and os.path.exists(backup_workspace):
        os.replace(backup_workspace, os.path.abspath(live_workspace))
