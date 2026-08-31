"""Media path classification and direct-folder image-sequence discovery."""

import os
import re
from collections import defaultdict


IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp"}
)
VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".m4v", ".mkv", ".mov", ".avi", ".webm"}
)
SUPPORTED_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def is_supported_drop_path(path: str) -> bool:
    """Return whether a dropped file or direct sequence folder is eligible."""
    if not path:
        return False
    if os.path.isdir(path):
        return True
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in (
        SUPPORTED_MEDIA_EXTENSIONS
    )


def discover_image_sequences(directory: str) -> list[list[str]]:
    """Find numbered image sequences directly inside *directory*.

    The search is intentionally non-recursive. A one-digit suffix is accepted
    only when separated by ``.``, ``_`` or ``-``; otherwise ordinary names such
    as ``version1.png`` and ``version2.png`` are not treated as frame sequences.
    """
    if not os.path.isdir(directory):
        return []

    groups = defaultdict(list)
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return []

    for entry in entries:
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue

        stem, extension = os.path.splitext(entry.name)
        extension = extension.lower()
        if extension not in IMAGE_EXTENSIONS:
            continue

        match = re.match(r"^(.+?)(\d+)$", stem)
        if match is None:
            continue
        prefix, number_text = match.groups()
        if len(number_text) < 2 and not prefix.endswith((".", "_", "-")):
            continue

        key = (prefix.casefold(), extension)
        groups[key].append((int(number_text), entry.name.casefold(), entry.path))

    sequences = []
    for frames in groups.values():
        if len(frames) < 2:
            continue
        frames.sort(key=lambda item: (item[0], item[1]))
        sequences.append([item[2] for item in frames])

    sequences.sort(key=lambda paths: os.path.basename(paths[0]).casefold())
    return sequences


def sequence_display_name(paths: list[str]) -> str:
    """Create a compact human-readable label for a discovered sequence."""
    if not paths:
        return "Empty sequence"
    first = os.path.basename(paths[0])
    last = os.path.basename(paths[-1])
    return f"{first}  —  {last}  ({len(paths)} frames)"
