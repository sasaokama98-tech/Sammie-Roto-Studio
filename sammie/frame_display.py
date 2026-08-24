"""Frame-number display helpers shared by the UI and sequence loader."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence


FRAME_INDEX_MODE = "Frame Index"
SOURCE_FRAME_MODE = "Source Frame"
TIMECODE_MODE = "Timecode"


def extract_source_frame_number(path: str) -> tuple[int, int] | None:
    """Return the final numeric filename token and its zero-padding width."""

    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)$", stem)
    if match is None:
        return None
    token = match.group(1)
    return int(token), len(token)


def sequence_frame_metadata(paths: Sequence[str]) -> tuple[list[int], int]:
    """Extract an exact source-frame mapping for a detected image sequence."""

    parsed = [extract_source_frame_number(path) for path in paths]
    if not parsed or any(value is None for value in parsed):
        return [], 0
    values = [value for value in parsed if value is not None]
    return [number for number, _padding in values], max(
        padding for _number, padding in values
    )


def infer_contiguous_sequence_metadata(
    selected_path: str, total_frames: int
) -> tuple[list[int], int]:
    """Rebuild legacy sequence numbering when only the selected path remains."""

    parsed = extract_source_frame_number(selected_path)
    if parsed is None or total_frames <= 1:
        return [], 0
    start_frame, padding = parsed
    return list(range(start_frame, start_frame + total_frames)), padding


def available_frame_display_modes(
    total_frames: int,
    source_frame_numbers: Sequence[int] | None = None,
    source_timecodes: Sequence[str] | None = None,
) -> list[str]:
    """Return only modes backed by complete frame metadata."""

    modes = [FRAME_INDEX_MODE]
    if source_frame_numbers and len(source_frame_numbers) == total_frames:
        modes.append(SOURCE_FRAME_MODE)
    if source_timecodes and len(source_timecodes) == total_frames:
        modes.append(TIMECODE_MODE)
    return modes


def format_frame_value(
    frame_index: int,
    mode: str,
    source_frame_numbers: Sequence[int] | None = None,
    source_frame_padding: int = 0,
    source_timecodes: Sequence[str] | None = None,
) -> str:
    """Format one internal frame index in the selected display coordinate system."""

    if mode == SOURCE_FRAME_MODE and source_frame_numbers:
        if 0 <= frame_index < len(source_frame_numbers):
            number = int(source_frame_numbers[frame_index])
            return f"{number:0{source_frame_padding}d}" if source_frame_padding else str(number)
    if mode == TIMECODE_MODE and source_timecodes:
        if 0 <= frame_index < len(source_timecodes):
            return str(source_timecodes[frame_index])
    return str(frame_index)


def format_in_out_range(
    in_point: int | None,
    out_point: int | None,
    formatter,
) -> str:
    """Format explicit In/Out endpoints and their inclusive frame count."""

    if in_point is None and out_point is None:
        return ""
    in_text = formatter(in_point) if in_point is not None else "—"
    out_text = formatter(out_point) if out_point is not None else "—"
    if in_point is not None and out_point is not None and out_point >= in_point:
        range_text = f"{out_point - in_point + 1} frames"
    else:
        range_text = "—"
    return f"In: {in_text}   Out: {out_text}   Range: {range_text}"
