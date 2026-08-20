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

## Phase 3 — MEMatte high-resolution backend

1. Resolve code/checkpoint licensing before distribution.
2. Add `max-number-token`, ROI margin, and tile-overlap settings.
3. Avoid full-frame 4K inference by deriving a padded object bounding box from
   the coarse mask and unknown trimap band.
4. Add seam-free ROI reintegration and half/float precision validation for EXR.

## Phase 4 — Hybrid HQ

1. Use MatAnyone2 or VideoMaMa alpha as the temporally stable base.
2. Derive foreground core, definite background, and uncertain edge bands.
3. Run MEMatte only on uncertain edge ROIs.
4. Merge with confidence-weighted feathering and optional short-window temporal
   stabilization; never replace the full temporal alpha unconditionally.
5. Expose Memory Safe/Balanced/Fast policies controlling model residency,
   image size, ROI batching, and token limits.

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
- MEMatte's README states MIT, but the repository tree reviewed on 2026-08-19
  did not contain a standalone LICENSE file. Obtain an explicit license file or
  maintainer confirmation before vendoring or shipping its code/checkpoints.

Primary references:

- https://github.com/Zarxrax/Sammie-Roto-2
- https://github.com/facebookresearch/sam3
- https://github.com/hustvl/ViTMatte
- https://github.com/linyiheng123/MEMatte
