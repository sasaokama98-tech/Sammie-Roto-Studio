# Phase 5: Memory and performance profiles

Phase 5 groups the existing temporal-resolution, batch, ROI/tile, token, and
flow controls into production-friendly profiles. Selecting a profile updates
the underlying session settings; manually editing any managed control switches
the session to `Custom` without discarding the edited values.

## Profiles

| Setting | Memory Safe | Balanced | Fast |
|---|---:|---:|---:|
| Temporal short side | 480 | 720 | 1080 |
| VideoMaMa overlap | 2 | 2 | 2 |
| VideoMaMa frames/batch | 16 | 16 | 32 |
| ViTMatte ROI margin | 48 | 64 | 96 |
| ViTMatte tile / overlap | 768 / 96 | 1024 / 128 | 2048 / 128 |
| MEMatte ROI margin | 64 | 96 | 128 |
| MEMatte tile / overlap | 1024 / 96 | 2048 / 128 | 3072 / 128 |
| MEMatte max global tokens | 6144 | 12000 | 18000 |
| MEMatte precision | Float16 | Float16 | Float16 |
| Hybrid flow short side | 480 | 720 | 1080 |

`Balanced` is the default for new sessions. Sessions created before Phase 5
open as `Custom`, preserving their existing numerical settings rather than
silently replacing them.

## Model residency invariant

Profiles do not change the staged lifecycle:

1. SAM 3.1 is unloaded before matting.
2. A Hybrid temporal model completes and is unloaded.
3. CUDA cache is cleared.
4. MEMatte loads for edge refinement and is unloaded after completion.
5. SAM 3.1 is restored when it was active before matting.

`Fast` therefore means larger inference batches, tiles, token budget, and
resolution. It never opts into simultaneous residency of the large models.
This keeps the original project requirement that inactive models are unloaded.

## Scope and validation

The profile values are applied to MatAnyone/MatAnyone2, VideoMaMa, ViTMatte,
MEMatte, and both Hybrid HQ stages through their existing session settings.
Unit tests cover the profile ordering, grouped UI updates, `Custom` switching,
and fixed-row/scrollable layout. Peak VRAM and throughput still need to be
measured on representative 2K/4K production sequences before tuning these
initial values.

## Phase 5.1 performance telemetry

Enable **Record Performance Metrics** to measure each Hybrid HQ run without
changing inference. Reports are written to:

```text
temp/hybrid_performance/<timestamp>_<temporal>_<profile>.json
temp/hybrid_performance/summary.csv
```

The detailed report separates `temporal`, `edge`, and optional `evaluation`
stages. It records elapsed seconds, seconds per frame-equivalent, and—on
CUDA—peak allocated/reserved bytes plus the increase from the start of each
stage. The comparison CSV stores one row per run with the active profile and
the numerical settings needed to reproduce it.

CUDA is synchronized at stage boundaries so GPU timings and peak readings are
meaningful. This adds a small measurement overhead and is therefore optional,
although it is enabled by default for new sessions. Report failures never
invalidate or delete completed mattes.
