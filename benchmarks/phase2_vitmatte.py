"""Reproducible 2K/4K ViTMatte ROI benchmark.

Runs the real Hugging Face checkpoint on deterministic synthetic composites so
timing, peak memory, unknown-band MAE, and tile planning can be compared across
machines without redistributing footage.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

from sammie.trimap import TrimapConfig, generate_trimap
from sammie.vitmatte_backend import VitMatteBackend, unknown_rois


def synthetic_case(width: int, height: int):
    y, x = np.mgrid[:height, :width]
    background = np.stack(
        [40 + 80 * x / width, 70 + 60 * y / height, 110 - 40 * x / width], axis=-1
    ).astype(np.uint8)
    alpha = np.zeros((height, width), dtype=np.float32)
    centers = [(width // 4, height // 2), (3 * width // 4, height // 2)]
    for cx, cy in centers:
        distance = ((x - cx) / 150.0) ** 2 + ((y - cy) / 220.0) ** 2
        alpha = np.maximum(alpha, np.clip((1.12 - distance) * 5.0, 0.0, 1.0))
        for offset in range(-120, 121, 20):
            strand = np.exp(-((x - (cx + offset + (y - cy) * 0.08)) ** 2) / 3.0)
            strand *= np.exp(-((y - (cy - 210)) ** 2) / 18000.0)
            alpha = np.maximum(alpha, strand.astype(np.float32))
    foreground = np.zeros_like(background)
    foreground[..., 0] = 210
    foreground[..., 1] = 155
    foreground[..., 2] = 95
    rgb = np.round(
        foreground * alpha[..., None] + background * (1.0 - alpha[..., None])
    ).astype(np.uint8)
    coarse = (alpha >= 0.5).astype(np.uint8) * 255
    trimap = generate_trimap(
        coarse, TrimapConfig(erode_width=16, dilate_width=24)
    )
    return rgb, trimap, alpha


class MemorySampler:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join()


def run_case(backend, width, height, margin, tile_size, overlap):
    rgb, trimap, truth = synthetic_case(width, height)
    roi_count = len(unknown_rois(trimap, margin))
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with MemorySampler() as memory:
        alpha = backend.predict_multi_roi_float(
            rgb, trimap, margin, tile_size, overlap
        )
    elapsed = time.perf_counter() - started
    unknown = trimap == 128
    mae = float(np.mean(np.abs(alpha[unknown] - truth[unknown])))
    result = {
        "resolution": f"{width}x{height}",
        "seconds": round(elapsed, 3),
        "fps": round(1.0 / elapsed, 4),
        "unknown_mae": round(mae, 6),
        "process_peak_mib": round(memory.peak / 1024**2, 1),
        "roi_count": roi_count,
        "known_bg_exact": bool(np.all(alpha[trimap == 0] == 0.0)),
        "known_fg_exact": bool(np.all(alpha[trimap == 255] == 1.0)),
    }
    if backend.device.type == "cuda":
        result["cuda_peak_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--margin", type=int, default=64)
    args = parser.parse_args()

    backend = VitMatteBackend(torch.device(args.device))
    backend.load()
    try:
        results = [
            run_case(backend, 2048, 1080, args.margin, args.tile_size, args.overlap),
            run_case(backend, 3840, 2160, args.margin, args.tile_size, args.overlap),
        ]
    finally:
        backend.unload()
    report = {
        "device": args.device,
        "model": backend.model_id,
        "tile_size": args.tile_size,
        "tile_overlap": args.overlap,
        "roi_margin": args.margin,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
