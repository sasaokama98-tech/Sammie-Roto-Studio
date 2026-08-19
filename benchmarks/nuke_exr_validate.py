"""Run with ``Nuke -t`` to verify the Phase 2 float EXR fixture."""

from __future__ import annotations

import json
import sys

import nuke


def main():
    path = sys.argv[-1]
    read = nuke.nodes.Read(file=path)
    channel = "Object_0.Y"
    if channel not in read.channels():
        raise RuntimeError(f"Missing channel {channel}: {read.channels()}")
    actual = sorted(
        float(nuke.sample(read, channel, x + 0.5, 0.5)) for x in range(4)
    )
    expected = sorted([0.0, 1.0 / 65535.0, 32768.0 / 65535.0, 1.0])
    max_error = max(abs(a - b) for a, b in zip(actual, expected))
    result = {
        "channels": read.channels(),
        "samples": actual,
        "max_abs_error": max_error,
        "passed": max_error <= 1e-7,
    }
    print("PHASE2_NUKE_RESULT=" + json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise RuntimeError(f"EXR precision validation failed: {result}")


main()
