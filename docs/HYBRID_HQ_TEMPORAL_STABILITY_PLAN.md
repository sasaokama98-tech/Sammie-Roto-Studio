# Hybrid HQ temporal-stability investigation

## Finding

The current Hybrid HQ merge is spatial-only. For each frame it:

1. derives an unknown band independently from the temporal alpha;
2. runs MEMatte independently on that frame; and
3. blends the MEMatte alpha at up to 100% weight inside the band.

It does not compare the refinement with adjacent frames, limit the MEMatte
residual, or reject an unstable refinement. This can replace a stable
VideoMaMa edge with frame-dependent image-matting detail.

The retained 37-frame diagnostic result (`139-175`) confirms the mechanism.
Inside the Hybrid unknown band:

- mean absolute MEMatte-vs-VideoMaMa change: `0.0692` alpha;
- mean 95th-percentile change: `0.3100` alpha;
- maximum change: `0.9944` alpha;
- mean frame-to-frame delta increased from `0.1571` to `0.1630` (`1.038x`).

The frame-difference measurement is not motion-compensated, so it is a
diagnostic rather than a benchmark metric. The very large spatial residuals
and the added residual variation nevertheless show that the merge can make
large, temporally independent changes to the VideoMaMa base.

## Existing techniques relevant to the fix

- [VideoMaMa](https://arxiv.org/abs/2601.14255) separates high-resolution
  spatial learning from video temporal learning. Its temporal stage exists
  because spatial-only detail does not guarantee temporal consistency.
- [MatAnyone 2](https://arxiv.org/abs/2512.11782) uses a pixel-wise Matting
  Quality Evaluator to identify reliable and erroneous alpha regions. Its
  offline pixel-wise selection is the closest published analogue to selecting
  between VideoMaMa and MEMatte instead of blindly blending them.
- [Robust Video Matting](https://openaccess.thecvf.com/content/WACV2022/html/Lin_Robust_High-Resolution_Video_Matting_With_Temporal_Guidance_WACV_2022_paper.html)
  uses recurrent temporal guidance and evaluates temporal coherence with
  `dtSSD`.
- [Deep Video Matting via Spatio-Temporal Alignment and Aggregation](https://arxiv.org/abs/2104.11208)
  aligns and aggregates neighboring-frame information. It also cautions that
  optical flow can be unreliable in complex matting regions, so flow should
  have an occlusion/confidence gate rather than being trusted everywhere.
- [Video Matting via Consistency-Regularized Graph Neural Networks](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Video_Matting_via_Consistency-Regularized_Graph_Neural_Networks_ICCV_2021_paper.html)
  explicitly regularizes alpha/foreground consistency across video.

## Recommended implementation sequence

### Phase 4.1 — Conservative residual merge

Status: implemented.

Keep VideoMaMa as the authority and express MEMatte only as a residual:

```text
residual = MEMatte alpha - VideoMaMa alpha
final    = VideoMaMa alpha + confidence * limited(residual)
```

Add all of the following before introducing optical flow:

- cap residual magnitude (starting test values: `0.05`, `0.10`, `0.20`);
- reduce refinement weight when MEMatte and VideoMaMa strongly disagree;
- apply a three-frame median or EMA to the residual, only in the unknown band;
- stabilize the merge weight/band across adjacent frames;
- bypass MEMatte when its accepted residual is too small to provide a visible
  detail gain.

Expose presets rather than many low-level controls:

- `Preserve Temporal` (default): residual limit `0.10`, disagreement fade
  `0.08-0.30`, three-frame residual median strength `75%`;
- `Balanced`: residual limit `0.20`, disagreement fade `0.12-0.50`,
  three-frame residual median strength `35%`;
- `Maximum Detail`: the original spatial-first, frame-independent merge.

Only the signed MEMatte residual is temporally filtered. The VideoMaMa or
MatAnyone2 temporal alpha is never filtered or replaced outside the current
frame's unknown band. Processing keeps only a three-frame CPU-side rolling
buffer and releases full alpha/trimap arrays after each finalized frame.

### Phase 4.2 — Motion-compensated confidence

Status: implemented as an opt-in experimental mode; disabled by default.

Warp the previous accepted residual or alpha into the current frame and lower
MEMatte confidence where the current proposal disagrees with it. Use
forward/backward consistency and an occlusion mask. Never apply flow smoothing
where flow confidence is low.

The implementation uses CPU OpenCV DIS flow at a configurable short-side
resolution (`480`, `720`, or `1080`). Each adjacent pair is calculated in both
directions. Round-trip cycle error and out-of-frame sampling produce a
per-pixel flow confidence map. Reliable pixels use motion-warped neighboring
residuals; unreliable pixels and occlusions use the original Phase 4.1
same-pixel residual neighbors.

The current MEMatte residual is additionally gated by its agreement with the
motion-aligned neighbor reference. Confidence diagnostics are written as
8-bit images under `temp/hybrid_motion_confidence/<frame>/<object>.png`,
matching the established masks/matting directory contract.
Each run also writes a preset-specific JSON summary under
`temp/hybrid_diagnostics/`, including the frame range, accepted residual
mean/max, and mean motion confidence per object.

### Phase 4.3 — Objective comparison

Status: no-reference comparison/archive implemented; ground-truth benchmark
metrics remain pending.

Retain standalone temporal alpha and all Hybrid variants, then compare:

- spatial: SAD/MAD, MSE, Grad, Conn where ground truth exists;
- temporal: `dtSSD` and a flow-warped alpha error;
- production review: hair/fur detail, motion-blur preservation, edge chatter,
  holes, contamination, and holdout on already-clean VideoMaMa edges.

The acceptance rule should be: Hybrid HQ must improve boundary detail without
regressing temporal metrics or visible stability relative to its temporal base.

The implemented production evaluator archives each run under
`temp/hybrid_evaluation/<run>/` and appends one row per object to
`summary.csv`. It reports residual size, known-region drift, boundary-gradient
ratio, and confidence-gated flow-warped errors for the temporal base, final
alpha, and accepted edge residual. Existing run names are never overwritten;
cancelled or failed partial archives are removed without touching other runs.

The first retained VideoMaMa / Preserve Temporal / Motion 720 run (`139-175`)
produced 37 temporal/final frames, 35 center-frame confidence maps, and 36
adjacent-flow comparisons. Known-region drift was exactly `0`; final
flow-warped error was `1.0099x` the temporal base and boundary-gradient ratio
was `1.00009x`. This is a healthy comparison baseline, not a claim of absolute
accuracy.
