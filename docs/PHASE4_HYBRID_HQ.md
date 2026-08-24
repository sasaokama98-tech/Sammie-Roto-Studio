# Phase 4: Hybrid HQ

Temporal-stability findings and the proposed conservative VideoMaMa merge are
tracked in [HYBRID_HQ_TEMPORAL_STABILITY_PLAN.md](HYBRID_HQ_TEMPORAL_STABILITY_PLAN.md).

Hybrid HQ combines a temporally stable video matte with MEMatte spatial detail
without replacing the full alpha on every frame.

## Pipeline

```text
MatAnyone2 or VideoMaMa
    -> temporal alpha
    -> unload temporal model / clear CUDA
    -> soft uncertainty + boundary safety band
    -> MEMatte on unknown ROIs only
    -> distance-feathered edge merge
    -> 16-bit PNG / existing EXR export
```

Known pixels outside the generated unknown band retain their temporal alpha
exactly. Inside the band, MEMatte gains influence with distance from the band
boundary. This protects the stable foreground core and background while
allowing fine hair, fur, defocus, and motion-blurred edges to be reconstructed
from the original-resolution RGB frame.

Phase 4.1 adds a `Hybrid Stability` preset. `Preserve Temporal` is the default:
it bounds and confidence-gates the MEMatte residual, then suppresses isolated
one-frame residuals with a three-frame median blend. `Maximum Detail` preserves
the original Phase 4 merge for direct comparison.

## Controls

- **Hybrid Temporal Base**: MatAnyone2 is the memory-safe default. VideoMaMa
  can provide a stronger temporal base but uses more VRAM and its overlap/chunk
  settings remain active.
- **Hybrid Stability**: Preserve Temporal, Balanced, or Maximum Detail.
- **Motion Confidence (Experimental)**: Uses bidirectional DIS optical flow to
  align residual detail. It is off by default and falls back pixel-by-pixel to
  Phase 4.1 when flow is inconsistent, occluded, or outside the image.
- **Flow Resolution**: CPU flow short-side resolution (`480`, `720`, or
  `1080`). Start with `720`.
- **Save Evaluation Run**: After both model stages are unloaded, archive the
  temporal/final comparison and calculate Phase 4.3 diagnostics. It is off by
  default because a run duplicates the 16-bit mattes on disk.
- **Evaluation Label**: Optional comparison name. A model/preset-based name is
  generated when blank, and a numeric suffix is added instead of overwriting
  an existing run.
- **Hybrid Edge Width**: Adds a pixel safety band around the temporal 0.5
  contour. Soft temporal pixels are always part of the unknown region.
- **Hybrid Edge Feather**: Blends between temporal and MEMatte alpha at the
  unknown-band boundary. Zero uses the MEMatte result throughout the band.
- **MEMatte ROI/Tile/Max Global Tokens/Precision**: The same controls used by
  standalone MEMatte apply to the spatial stage.
- **Internal Resolution**: Controls the temporal stage only. MEMatte continues
  to use original-resolution RGB and ROI tiles.

## Stage artifacts

Final mattes continue to use the existing contract:

```text
temp/matting/<frame>/<object>.png
```

Hybrid HQ also keeps diagnostic 16-bit temporal bases and generated trimaps:

```text
temp/hybrid_temporal/<frame>/<object>.png
temp/hybrid_trimaps/<frame>/<object>.png
```

These diagnostics make it possible to distinguish temporal-stage errors from
edge-refinement or merge errors. They are temporary project data and are not
included in source control.

## Phase 4.3 evaluation runs

An enabled evaluation follows Sammie's established frame/object order:

```text
temp/hybrid_evaluation/<run>/final/<frame>/<object>.png
temp/hybrid_evaluation/<run>/temporal/<frame>/<object>.png
temp/hybrid_evaluation/<run>/confidence/<frame>/<object>.png
temp/hybrid_evaluation/<run>/report.json
temp/hybrid_evaluation/summary.csv
```

The report records unknown-band residual magnitude, exact known-region drift,
boundary-gradient change, unwarped frame delta, confidence-gated flow-warped
temporal error, and residual flow-warped error. Flow is calculated once per
adjacent source-frame pair and reused for every object. The summary CSV appends
one comparison row per object, which makes runs easy to compare externally.

These are no-reference production diagnostics. They measure changes relative
to the run's own temporal base and cannot prove absolute matte accuracy.
Ground-truth SAD, MSE, gradient, connectivity, and formal dtSSD remain separate
fixture-validation work.

## Current validation boundary

Edge-band construction, 8/16-bit normalization, known-region preservation,
feathering, combined-object mapping, manager selection, GUI visibility,
evaluation archive safety, metric generation, and summary output are covered
by automated tests. Full fixture comparison still requires matched MatAnyone2
and VideoMaMa runs plus production review.
