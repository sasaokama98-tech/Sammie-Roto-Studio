"""Create a deterministic 16-bit-alpha-to-float-EXR validation fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sammie.export_workers import SequenceExportWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    values = np.array([[0, 1, 32768, 65535]], dtype=np.float32) / 65535.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    SequenceExportWorker._write_exr_file(
        str(args.output), {"Object_0.Y": values}
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
