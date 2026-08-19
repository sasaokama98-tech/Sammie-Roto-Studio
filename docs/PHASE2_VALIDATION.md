# Phase 2 validation

Validated on 2026-08-19 with the official
`hustvl/vitmatte-small-composition-1k` checkpoint.

## Automated coverage

- Three-class automatic/manual trimap generation and color preview.
- Disconnected-component ROI extraction, padded ROI merging, tile coverage,
  overlap feathering, and exact known foreground/background preservation.
- Float alpha integration with detail below one 8-bit code value.
- 16-bit PNG intermediate readback and float32 multilayer EXR round-trip.
- Application/module import and dependency lock checks.

## 2K / 4K benchmark

The reproducible benchmark uses two disconnected soft-edge subjects with thin
strand details. It runs the real model rather than a mock. The scene is
deterministic and has a known alpha, allowing unknown-band MAE to be reported.

| Device | Resolution | Time | Throughput | Peak VRAM | Process peak | Unknown MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5060 Ti / CUDA 13.0 | 2048x1080 | 0.485 s | 2.062 fps | 523.4 MiB | 1450.2 MiB | 0.071024 |
| RTX 5060 Ti / CUDA 13.0 | 3840x2160 | 0.405 s | 2.467 fps | 523.4 MiB | 1656.9 MiB | 0.065544 |
| CPU | 2048x1080 | 1.022 s | 0.978 fps | n/a | 1051.1 MiB | 0.071007 |
| CPU | 3840x2160 | 0.916 s | 1.092 fps | n/a | 1331.9 MiB | 0.065546 |

Both resolutions produced two independent ROIs. Known background remained
exactly 0.0 and known foreground exactly 1.0. Similar 2K/4K timings are
expected because inference follows the unknown-region ROIs rather than the
full frame. Machine-readable results are in `PHASE2_BENCHMARK.json` and
`PHASE2_BENCHMARK_CUDA.json`.

## Nuke / EXR

Nuke 17.0v3 successfully read the float channel `Object_0.Y`. Values derived
from 16-bit alpha codes `[0, 1, 32768, 65535]` were recovered with a maximum
absolute error of `1.1641709818377421e-10`.

The commands are reproducible with:

```powershell
python -m benchmarks.phase2_export_fixture precision.exr
& 'C:\Program Files\Nuke17.0v3\Nuke17.0.exe' -t `
  benchmarks/nuke_exr_validate.py precision.exr
```

The deterministic benchmark measures regression and resource behavior. Final
artistic acceptance on production footage remains part of the broader VFX
validation matrix, not a blocker for the Phase 2 backend implementation.
