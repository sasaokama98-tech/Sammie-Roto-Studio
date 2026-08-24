"""Adapter for an external checkout of the official MEMatte implementation.

MEMatte is intentionally not vendored.  The upstream repository currently
states MIT terms in its README but does not contain a standalone LICENSE file.
Users supply an official checkout and checkpoint locally; this module only
adapts its public model architecture to Sammie Roto Studio's image pipeline.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import importlib
import os
import sys
import types

import numpy as np
import torch
from torch import nn

from sammie.vitmatte_backend import run_tiled_unknown_inference


class MematteUnavailableError(RuntimeError):
    """Raised when the external MEMatte runtime or checkpoint cannot load."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_PATH = PROJECT_ROOT / "external" / "MEMatte"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "mematte"
DEFAULT_CHECKPOINT_NAMES = (
    "MEMatte_ViTS_AIM500.pth",
    "MEMatte_ViTS_DIM.pth",
)


@dataclass(frozen=True)
class MematteRuntimePaths:
    repo: Path
    checkpoint: Path


def resolve_mematte_paths(
    repo_path: str | os.PathLike | None = None,
    checkpoint_path: str | os.PathLike | None = None,
) -> MematteRuntimePaths:
    """Resolve explicit, environment, then conventional local paths."""

    repo_value = repo_path or os.environ.get("MEMATTE_REPO_PATH") or DEFAULT_REPO_PATH
    repo = Path(repo_value).expanduser().resolve()

    checkpoint_value = checkpoint_path or os.environ.get("MEMATTE_CHECKPOINT_PATH")
    if checkpoint_value:
        checkpoint = Path(checkpoint_value).expanduser().resolve()
    else:
        candidates = [DEFAULT_CHECKPOINT_DIR / name for name in DEFAULT_CHECKPOINT_NAMES]
        checkpoint = next((path for path in candidates if path.is_file()), None)
        if checkpoint is None:
            discovered = sorted(DEFAULT_CHECKPOINT_DIR.glob("*.pth"))
            checkpoint = discovered[0] if len(discovered) == 1 else candidates[0]
    return MematteRuntimePaths(repo=repo, checkpoint=checkpoint)


class _CNNBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride


class _LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, value):
        mean = value.mean(1, keepdim=True)
        variance = (value - mean).pow(2).mean(1, keepdim=True)
        normalized = (value - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight[:, None, None] + self.bias[:, None, None]


@dataclass
class _ShapeSpec:
    channels: int | None = None
    height: int | None = None
    width: int | None = None
    stride: int | None = None


def _get_norm(name, channels):
    if name in (None, ""):
        return None
    if name == "LN":
        return _LayerNorm2d(channels)
    if name == "BN":
        return nn.BatchNorm2d(channels)
    raise MematteUnavailableError(f"Unsupported MEMatte normalization: {name}")


def _c2_msra_fill(layer):
    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


@contextmanager
def _official_import_environment(repo: Path):
    """Temporarily provide the small detectron2/fvcore surface MEMatte imports."""

    installed = {}

    def add_module(name, module):
        if name not in sys.modules:
            sys.modules[name] = module
            installed[name] = module

    try:
        import detectron2  # noqa: F401
    except ImportError:
        detectron2_module = types.ModuleType("detectron2")
        layers_module = types.ModuleType("detectron2.layers")
        layers_module.CNNBlockBase = _CNNBlockBase
        layers_module.Conv2d = nn.Conv2d
        layers_module.ShapeSpec = _ShapeSpec
        layers_module.get_norm = _get_norm
        structures_module = types.ModuleType("detectron2.structures")
        structures_module.ImageList = object
        modeling_module = types.ModuleType("detectron2.modeling")
        backbone_module = types.ModuleType("detectron2.modeling.backbone")
        fpn_module = types.ModuleType("detectron2.modeling.backbone.fpn")
        fpn_module._assert_strides_are_log2_contiguous = lambda _strides: None
        add_module("detectron2", detectron2_module)
        add_module("detectron2.layers", layers_module)
        add_module("detectron2.structures", structures_module)
        add_module("detectron2.modeling", modeling_module)
        add_module("detectron2.modeling.backbone", backbone_module)
        add_module("detectron2.modeling.backbone.fpn", fpn_module)

    try:
        import fvcore  # noqa: F401
    except ImportError:
        fvcore_module = types.ModuleType("fvcore")
        fvcore_nn_module = types.ModuleType("fvcore.nn")
        weight_init_module = types.ModuleType("fvcore.nn.weight_init")
        weight_init_module.c2_msra_fill = _c2_msra_fill
        add_module("fvcore", fvcore_module)
        add_module("fvcore.nn", fvcore_nn_module)
        add_module("fvcore.nn.weight_init", weight_init_module)

    try:
        import fairscale  # noqa: F401
    except ImportError:
        fairscale_module = types.ModuleType("fairscale")
        fairscale_nn_module = types.ModuleType("fairscale.nn")
        checkpoint_module = types.ModuleType("fairscale.nn.checkpoint")
        checkpoint_module.checkpoint_wrapper = lambda module: module
        add_module("fairscale", fairscale_module)
        add_module("fairscale.nn", fairscale_nn_module)
        add_module("fairscale.nn.checkpoint", checkpoint_module)

    repo_text = str(repo)
    sys.path.insert(0, repo_text)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == repo_text:
            sys.path.pop(0)
        else:
            try:
                sys.path.remove(repo_text)
            except ValueError:
                pass
        for name, module in reversed(list(installed.items())):
            if sys.modules.get(name) is module:
                sys.modules.pop(name, None)


class MematteBackend:
    """Original-resolution MEMatte inference with ROI and token controls."""

    VALID_PRECISIONS = {"Float32", "Float16", "BFloat16"}

    def __init__(
        self,
        device: torch.device,
        repo_path: str | os.PathLike | None = None,
        checkpoint_path: str | os.PathLike | None = None,
        max_number_token: int = 12000,
        precision: str = "Float16",
    ):
        if max_number_token <= 0:
            raise ValueError("MEMatte max_number_token must be positive")
        if precision not in self.VALID_PRECISIONS:
            raise ValueError(f"Unsupported MEMatte precision: {precision}")
        self.device = torch.device(device)
        self.paths = resolve_mematte_paths(repo_path, checkpoint_path)
        self.max_number_token = int(max_number_token)
        self.precision = precision
        self.model = None

    def _validate_paths(self):
        expected = self.paths.repo / "modeling" / "__init__.py"
        if not expected.is_file():
            raise MematteUnavailableError(
                "MEMatte source checkout was not found. Clone the official "
                "linyiheng123/MEMatte repository to external/MEMatte or set "
                "MEMATTE_REPO_PATH."
            )
        if not self.paths.checkpoint.is_file():
            raise MematteUnavailableError(
                "MEMatte checkpoint was not found. Place an official ViTS "
                "checkpoint in checkpoints/mematte or set MEMATTE_CHECKPOINT_PATH."
            )

    def _build_model(self):
        try:
            import timm  # noqa: F401
        except ImportError as exc:
            raise MematteUnavailableError(
                "MEMatte requires timm. Install the SAM 3.1 dependencies or timm."
            ) from exc

        existing = sys.modules.get("modeling")
        if existing is not None:
            source = Path(getattr(existing, "__file__", "")).resolve()
            if self.paths.repo not in source.parents:
                raise MematteUnavailableError(
                    f"A conflicting Python module named 'modeling' is loaded from {source}."
                )

        try:
            with _official_import_environment(self.paths.repo):
                official = importlib.import_module("modeling")
                from functools import partial

                backbone = official.ViT(
                    in_chans=4,
                    img_size=512,
                    patch_size=16,
                    embed_dim=384,
                    depth=12,
                    num_heads=6,
                    drop_path_rate=0,
                    window_size=14,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
                    residual_block_indexes=[2, 5, 8, 11],
                    use_rel_pos=True,
                    out_feature="last_feat",
                    topk=0.25,
                    multi_score=True,
                    max_number_token=self.max_number_token,
                )
                model = official.MEMatte(
                    teacher_backbone=None,
                    backbone=backbone,
                    criterion=official.MattingCriterion(
                        losses=[
                            "unknown_l1_loss",
                            "known_l1_loss",
                            "loss_pha_laplacian",
                            "loss_gradient_penalty",
                        ]
                    ),
                    pixel_mean=[123.675 / 255.0, 116.280 / 255.0, 103.530 / 255.0],
                    pixel_std=[58.395 / 255.0, 57.120 / 255.0, 57.375 / 255.0],
                    input_format="RGB",
                    size_divisibility=32,
                    decoder=official.Detail_Capture(),
                    distill=True,
                    distill_loss_ratio=1,
                    token_loss_ratio=1,
                )
        except MematteUnavailableError:
            raise
        except Exception as exc:
            raise MematteUnavailableError(
                "Unable to construct the official MEMatte ViTS model with the current runtime."
            ) from exc
        return model

    @staticmethod
    def _checkpoint_state(payload):
        if not isinstance(payload, dict):
            raise MematteUnavailableError("Unsupported MEMatte checkpoint format")
        state = payload.get("model", payload.get("state_dict", payload))
        if not isinstance(state, dict):
            raise MematteUnavailableError("MEMatte checkpoint has no model state dictionary")
        normalized = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[7:]
            if not key.startswith("teacher_backbone."):
                normalized[key] = value
        if not any(key.startswith("backbone.") for key in normalized):
            raise MematteUnavailableError("MEMatte checkpoint does not contain backbone weights")
        return normalized

    def load(self):
        self._validate_paths()
        model = self._build_model()
        try:
            payload = torch.load(
                self.paths.checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            state = self._checkpoint_state(payload)
            missing, unexpected = model.load_state_dict(state, strict=False)
        except MematteUnavailableError:
            raise
        except Exception as exc:
            raise MematteUnavailableError(
                f"Unable to load MEMatte checkpoint {str(self.paths.checkpoint)!r}."
            ) from exc

        missing_required = [
            key for key in missing if key.startswith(("backbone.", "decoder."))
        ]
        unexpected_required = [
            key for key in unexpected if not key.startswith("teacher_backbone.")
        ]
        if missing_required or unexpected_required:
            details = (missing_required + unexpected_required)[:5]
            raise MematteUnavailableError(
                "MEMatte checkpoint is incompatible with the ViTS architecture: "
                + ", ".join(details)
            )

        self.model = model.to(self.device).eval()
        self.model.backbone.max_number_token = self.max_number_token
        return self.model

    def _autocast(self):
        if self.device.type != "cuda" or self.precision == "Float32":
            return nullcontext()
        dtype = torch.float16 if self.precision == "Float16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def predict(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MEMatte model is not loaded")
        rgb = np.asarray(rgb)
        trimap = np.asarray(trimap)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or trimap.shape != rgb.shape[:2]:
            raise ValueError("MEMatte expects an HxWx3 RGB image and matching 2D trimap")

        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        image = image.float().div_(255.0).unsqueeze(0)
        normalized_trimap = np.where(
            trimap >= 170, 1.0, np.where(trimap >= 85, 0.5, 0.0)
        ).astype(np.float32)
        trimap_tensor = torch.from_numpy(normalized_trimap).unsqueeze(0).unsqueeze(0)
        batch = {"image": image, "trimap": trimap_tensor}

        self.model.backbone.max_number_token = self.max_number_token
        with torch.inference_mode(), self._autocast():
            outputs, _, _ = self.model(batch, patch_decoder=True)
        alpha = outputs["phas"][0, 0, : trimap.shape[0], : trimap.shape[1]]
        return np.clip(alpha.float().cpu().numpy(), 0.0, 1.0)

    def predict_multi_roi_float(
        self,
        rgb: np.ndarray,
        trimap: np.ndarray,
        margin: int = 96,
        max_tile_size: int = 2048,
        tile_overlap: int = 128,
    ) -> np.ndarray:
        return run_tiled_unknown_inference(
            self.predict,
            rgb,
            trimap,
            margin=margin,
            max_tile_size=max_tile_size,
            tile_overlap=tile_overlap,
        )

    def unload(self):
        self.model = None
