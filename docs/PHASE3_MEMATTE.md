# Phase 3: MEMatte local runtime

Sammie Roto Studio integrates MEMatte as an optional high-resolution image
matting backend. The official source and checkpoints are not redistributed by
this repository.

## Local files

Clone the official source into the conventional location:

```powershell
git clone https://github.com/linyiheng123/MEMatte.git external/MEMatte
```

Download one of the official ViTS checkpoints linked from the upstream README
and place it at one of these paths:

```text
checkpoints/mematte/MEMatte_ViTS_AIM500.pth
checkpoints/mematte/MEMatte_ViTS_DIM.pth
```

If the directory contains exactly one `.pth` file, Studio also discovers it
without requiring a rename. With multiple files, use one of the conventional
names or set `MEMATTE_CHECKPOINT_PATH` explicitly.

The AIM-500 checkpoint is the first candidate because it is trained for robust
real-world images. The DIM/Composition-1K checkpoint remains available for
controlled comparison. Custom locations can be selected without modifying the
project:

```powershell
$env:MEMATTE_REPO_PATH = "D:\models\MEMatte"
$env:MEMATTE_CHECKPOINT_PATH = "D:\models\MEMatte_ViTS_AIM500.pth"
```

The adapter uses the current PyTorch/torchvision environment and `timm`. It
provides the limited detectron2/fvcore/fairscale API surface imported by the
official inference model, instead of installing the upstream repository's
legacy Torch 2.0/OpenCV 4.5 environment.

Install the optional runtime together with the compute backend used by Studio.
For the current Windows CUDA 13.0 setup:

```powershell
uv sync --extra cu130 --extra mematte
```

Use `cu126`, `cpu`, or another existing compute extra in place of `cu130` when
appropriate.

## Processing controls

- **ROI Margin** adds original-image context around each connected unknown
  trimap boundary.
- **Tile Size** limits the largest spatial crop sent to the model.
- **Tile Overlap** creates feathered overlap for seam-free reintegration.
- **Max Global Tokens** caps tokens routed through global attention. Lower
  values use less VRAM; upstream demonstrates `18000`, while Studio defaults to
  `12000` for a safer starting point.
- **Precision** controls CUDA autocast. CPU inference always uses Float32.

Known foreground/background pixels remain exact. Only unknown trimap pixels
are replaced, and alpha is saved as 16-bit PNG before the existing EXR export.

## Validation status

The adapter architecture, model construction, ROI integration, checkpoint
state handling, and precision plumbing are covered by automated tests. A real
official checkpoint is still required for 2K/4K CUDA quality, VRAM, and timing
benchmarks.

## License boundary

The official MEMatte README states that its code is MIT-licensed, but the
reviewed repository does not include a standalone LICENSE file. Until upstream
provides one or confirms redistribution terms, Studio loads a user-supplied
checkout and checkpoint and does not vendor or package either artifact.

Official sources:

- https://github.com/linyiheng123/MEMatte
- https://arxiv.org/abs/2412.10702
