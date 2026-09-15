# VFX HQ matting extension plan

## Current architecture (upstream main, reviewed 2026-08-19)

- `sammie/sammie.py::SamManager` owns segmentation model load/unload,
  point prompting, replay, and video propagation. SAM2 Large/Base and
  EfficientTAM share the SAM2 predictor API.
- `sammie/matting.py::MattingManager` is already an abstract base with a
  factory for MatAnyone/MatAnyone2 and VideoMaMa.
- `sammie_main.py` owns the PySide6 model selectors and starts each manager.
- `sammie/core.py` owns mask directories, post-processing, ROI helpers, and
  device cache clearing.
- `sammie/export_formats.py` and `sammie/export_workers.py` preserve PNG and
  multilayer EXR output. The HQ work should continue to write alpha into the
  existing `temp/matting/<frame>/<object>.png` contract.

## Source integration decision

Keep Sammie-Roto-2 as the application source and track it with an `upstream`
Git remote. Do not vendor the SAM 3.1 repository: install it as an optional
dependency and keep its request/session API behind `Sam31Backend`. This avoids
mixing the SAM License into Sammie's GPL-covered source tree and makes upstream
Sammie updates easier to merge.

ViTMatte and MEMatte should likewise be integrated behind matting adapters.
Their original repositories pin Torch 2.0.0, timm 0.5.4, and older OpenCV, so
copying their environments into Sammie would conflict with Sammie's current
Torch 2.11/OpenCV 4.12 stack. Prefer a compatible packaged inference API where
available (for example, the Transformers ViTMatte implementation), or isolate
legacy runtimes until compatibility tests pass.

## Phase 1 (implemented)

- Add `SAM 3.1` to the existing segmentation model selector.
- Add an optional official SAM3 dependency (`uv sync --extra sam31`).
- Adapt point prompts, object IDs, forward/backward propagation, reset,
  remove-object, and session close without changing the existing mask layout.
- Convert UI points from original-frame pixels to normalized 0..1 coordinates
  before using SAM 3.1's relative-coordinate point API. Passing source pixels
  directly as model-space coordinates breaks selection on non-1008 footage.
- Recover an empty first-click multimask by retrying the authored positive point
  through the single-mask decoder, then an adjacent-pixel positive seed. The
  synthetic seed is not added to the Studio point table, and an unrecoverable
  empty mask now reports an explicit error instead of failing silently.
- Stage PNG/TIFF frames as high-quality JPEG only for SAM 3.1 model input.
  Original footage remains the source for display, matting, and export.
- Generate automatic 0/128/255 trimaps next to each coarse mask under
  `temp/trimaps/<frame>/<object>.png`.
- Keep model unload and CUDA cache clearing in the existing lifecycle.

SAM 3.1 first uses a local checkpoint at
`checkpoints/sam31/sam3.1_multiplex.pt`. A different location can be selected
with `SAM31_CHECKPOINT_PATH`. If neither exists, the backend falls back to the
gated `facebook/sam3.1` Hugging Face repository, which requires accepted access
and `hf auth login`. The official SAM 3.1 multiplex path is CUDA-only in this
integration; SAM2/EfficientTAM remain available elsewhere.

Dependency note: official SAM3 metadata currently requires `numpy<2`, while
Sammie's OpenCV 4.12+ requires `numpy>=2`. The `uv` configuration retains the
current OpenCV stack and overrides NumPy to `>=2,<2.3` for this PoC. This
avoids a known Sammie colorspace dependency regression, but the override must
pass the real SAM 3.1 CUDA smoke test before release packaging.

On native Windows, the extra installs `triton-windows` 3.6.x, matching
PyTorch 2.11. The Windows PyTorch build can omit Flash SDPA even though SAM
3.1 multiplex forces its Flash-only context. The backend detects that case
and limits the SAM 3.1 decoder to cuDNN, Efficient Attention, then Math SDPA.
This avoids the `No available kernel` propagation failure without changing
global PyTorch backend selection.

The official multiplex builder performs multiple non-strict checkpoint loads
and prints every missing/unexpected key, even when the assembled predictor
loads successfully. Sammie Roto Studio suppresses only those verbose key-list
lines and emits one compatibility summary so the console remains readable.
Set `SAM31_VERBOSE_CHECKPOINT_KEYS=1` before launch to restore the complete
official diagnostic output when debugging a checkpoint mismatch.

### SAM 3.1 follow-up — prompt selection mode (implemented)

- Prompt controls appear only while `SAM 3.1` is selected; SAM2 and
  EfficientTAM point workflows are unchanged.
- Semantic `add_prompt` runs in an isolated preview session. One or more masks
  can be previewed and selected without modifying the active point session.
- Accepting candidates is restricted to the In or Out frame and explicitly
  confirms before clearing existing points, masks, mattes, or removal results.
- Accepted SAM candidate IDs are mapped to Studio object IDs. A stable positive
  seed is placed inside each accepted mask so normal point refinement and
  forward/backward tracking continue to work.
- Prompt text, anchor frame, and candidate-to-object mappings are stored in
  session/project settings and restored before point replay.

See `SAM31_PROMPT_SELECTION.md` for the workflow and persistence semantics.


## Phase 2 — ViTMatte image matting (complete)

Implemented:

- `ImageMattingManager` consumes original-resolution RGB plus generated
  trimaps and writes the existing `temp/matting` alpha contract.
- `VitMatteBackend` uses the official Transformers implementation and
  `hustvl/vitmatte-small-composition-1k` checkpoint.
- Unknown-band ROI inference with a configurable margin; known foreground,
  known background, and all pixels outside the ROI remain exact.
- Matting UI model selection and Auto/manual trimap controls for FG erosion,
  BG dilation, and ViTMatte ROI margin.
- A `Trimap-Preview` viewer mode with definite-background, unknown, and
  definite-foreground color classes.
- Connected-component multi-ROI planning and overlapping 1024px tiles with
  feathered reintegration for large/disconnected unknown regions.
- ViTMatte intermediate alpha is stored as 16-bit PNG. EXR export reads that
  data directly as float32, avoiding an 8-bit viewer conversion.
- SAM 3.1 is fully unloaded before a matting model is loaded, then restored
  after matting, because SAM 3.1 cannot be CPU-offloaded.
- Unit coverage for trimap preview, ROI extraction/reintegration, tiling,
  16-bit intermediate precision, and float EXR round-trips.
- Real-checkpoint 2K/4K CPU and CUDA benchmarks on deterministic composites.
- Nuke 17.0v3 readback of `Object_0.Y`, including the smallest non-zero 16-bit
  alpha value, with maximum absolute error below `1.2e-10`.

See `PHASE2_VALIDATION.md` and the JSON benchmark outputs for measurements.

## Phase 3 — MEMatte high-resolution backend (adapter implemented)

Implemented:

- External-source adapter for the official `linyiheng123/MEMatte` ViTS model;
  no MEMatte source or checkpoint is redistributed.
- Conventional local paths plus `MEMATTE_REPO_PATH` and
  `MEMATTE_CHECKPOINT_PATH` overrides.
- Minimal inference-only compatibility layer for the detectron2/fvcore/
  fairscale symbols used by the official model, avoiding a legacy detectron2
  install in the current Windows/Torch 2.11 environment.
- Configurable `max-number-token`, ROI margin, tile size, tile overlap, and
  Float16/BFloat16/Float32 CUDA inference settings.
- Connected unknown-band/object ROIs and feathered overlapping tiles, so 2K/4K
  frames do not require unconditional full-frame inference.
- The same 16-bit PNG intermediate and float32 EXR path validated in Phase 2.
- Unit coverage for path resolution, checkpoint normalization, trimap input,
  token controls, tiled reintegration, and known-region preservation.

Pending validation:

- Run the official ViTS AIM-500 and DIM checkpoints on the 2K/4K fixture and
  record peak VRAM, time, edge scores, and Float16/BFloat16/Float32 deltas.
- Obtain an explicit standalone upstream license file or maintainer
  confirmation before any redistribution of code or checkpoint files.

See `PHASE3_MEMATTE.md` for local runtime setup and current limitations.

## Phase 4 — Hybrid HQ (Phase 4.4 implemented)

Implemented:

- Selectable MatAnyone2 or VideoMaMa temporal base.
- Strict staged lifecycle: finish and unload the temporal model, clear CUDA,
  then load MEMatte. The two large models never share VRAM residency.
- Temporal-alpha trimap generation combining soft-alpha uncertainty with a
  configurable morphological safety band.
- MEMatte inference only over connected unknown-band ROIs and tiles.
- Distance-weighted edge feathering. Known foreground/background retain the
  temporal alpha exactly; the full alpha is never replaced unconditionally.
- 16-bit Hybrid output plus diagnostic temporal mattes and trimaps under
  `temp/hybrid_temporal` and `temp/hybrid_trimaps`.
- GUI controls for temporal base, edge width, edge feather, and the existing
  MEMatte ROI/tile/token/precision settings.
- Preserve Temporal/Balanced/Maximum Detail residual policies and opt-in
  bidirectional motion confidence.
- Named Phase 4.3 evaluation archives with temporal/final/confidence mattes,
  per-frame JSON diagnostics, and an append-only comparison CSV.
- Optional Phase 4.4 ground-truth evaluation of both the temporal base and
  final Hybrid result with SAD/MSE/Gradient/Connectivity/dtSSD, strict frame
  and object completeness, archived GT alpha, and a separate comparison CSV.

Pending validation and follow-up:

- Compare MatAnyone2 and VideoMaMa bases on the full VFX fixture and record
  edge detail, flicker, VRAM, and time.
- Tune edge width/feather defaults for hair, motion blur, and defocus.
- Run the new Phase 4.4 metrics on the full ground-truth fixture and establish
  acceptance thresholds for each shot class.
See `PHASE4_HYBRID_HQ.md` for processing semantics and controls.

## Phase 5 — Memory/performance profiles (implemented)

Implemented:

- `Memory Safe`, `Balanced`, and `Fast` profiles covering temporal resolution,
  VideoMaMa batch/overlap, ViTMatte and MEMatte ROI/tile settings, MEMatte
  token/precision settings, and Hybrid flow resolution.
- `Custom` is selected automatically when a managed setting is edited.
- Existing sessions without profile metadata remain `Custom`; new sessions
  use the configurable `Balanced` default.
- Strict large-model staged unload remains mandatory in every profile,
  including `Fast`.

Pending validation:

- Record peak VRAM and frame time for all three profiles on matched 2K/4K
  sequences, then adjust the initial thresholds if necessary.

See `PHASE5_MEMORY_PROFILES.md` for the exact profile matrix.

### Phase 5.1 — Performance telemetry (implemented)

- Hybrid HQ records temporal, MEMatte edge, and optional evaluation stages
  independently.
- JSON reports include elapsed time, seconds per frame-equivalent, and CUDA
  peak allocated/reserved memory.
- `temp/hybrid_performance/summary.csv` provides one reproducible comparison
  row per run, including the active profile and core numerical settings.
- Telemetry is optional and cannot fail or alter an otherwise successful matte.

### Phase 5.2 — I/O reliability (implemented)

- Media is decoded/copied into a sibling staging workspace and validated before
  replacing the live `temp` session. Cancellation, unreadable frames,
  inconsistent dimensions, missing indexes, or write failures leave the
  previous session intact.
- Loaded frame workspaces require exactly one readable frame for every internal
  index in the detected sequence or decoded movie range.
- PNG and EXR sequence export writes every requested frame atomically and
  verifies the complete output range. Missing segmentation/matting masks become
  zero-value frames/layers instead of sequence gaps.
- Multilayer EXR uses a stable object-layer set across the complete sequence.
- Video export rejects skipped renders, writes through a temporary container,
  then decodes the completed file to verify its frame count before replacing an
  existing output.
- Sequence overwrite detection checks the complete requested range rather than
  sampling only its first five frames.

## Validation matrix

Use short, redistributable 2K and 4K sequences covering hair, beard, fur,
motion blur, defocus, translucency, fast motion, similar foreground/background
colors, and occlusion. Record edge detail, flicker/chatter, holes,
contamination, blur preservation, peak VRAM, and frame time. Verify PNG and EXR
round-trips in Nuke before enabling a backend by default.

## License notes (not legal advice)

- Sammie-Roto-2 source is GPL-3.0.
- SAM 3/3.1 uses Meta's custom SAM License, not an OSI permissive license.
  Redistribution must retain that agreement; use is also subject to its trade,
  privacy, reverse-engineering, and prohibited-end-use clauses.
- ViTMatte code declares MIT. Check the separately downloaded checkpoint and
  training-dataset terms before commercial redistribution.
- MEMatte's README states MIT, but upstream commit
  `3c887e2a517b38f936b97f27d933c5897c6d47a1` reviewed on 2026-08-20 still did
  not contain a standalone LICENSE file. Obtain an explicit license file or
  maintainer confirmation before vendoring or shipping its code/checkpoints.

Primary references:

- https://github.com/Zarxrax/Sammie-Roto-2
- https://github.com/facebookresearch/sam3
- https://github.com/hustvl/ViTMatte
- https://github.com/linyiheng123/MEMatte
