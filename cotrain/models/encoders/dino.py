"""Frozen DINOv2 ViT-B/14 visual encoder (PROJECT_PLAN_1.md §2.2.1).

We load the model via the `jimmy` library (`clementpoiret/jimmy`), which
ships a native Flax NNX implementation. **Production pipeline:** dataset
RGB is 224×224 uint8; the encoder bilinear-upscales to the training
resolution 518×518, normalizes with ImageNet mean/std, then runs the
DINOv2 ViT-B/14 forward and returns the mean-pooled patch tokens (B, 768).

We feed at native 518 (rather than letting jimmy's dynamic_img_size shrink
the position embedding to 16×16) because PyTorch DINOv2 and jimmy use
slightly different position-embed interpolation schemes (PyTorch's
`interpolate_offset=0.1` historical kludge vs jimmy's plain
`jax.image.resize`); upscaling the input avoids that mismatch entirely
and lets us hit the §2.2.1 parity gate (max|Δ| < 1e-3 vs the PyTorch
reference). Verified at fp64: max|Δ|=1.3e-12.

Two GELU formulas exist in the wild and they don't agree to fp32: PyTorch
DINOv2 uses `nn.GELU(approximate='none')` (exact, erf-based) while jimmy's
default is `nnx.gelu` (approximate, tanh-based). We patch every block's
`mlp.act` at construction to use the exact form so the rest of the network
is bit-equal.

Frozen forward: this module owns nnx.Param leaves but the transformer
training step uses `train_step_with_encoder(...)` which keeps the encoder
*outside* the optimizer's gradient tree (§3.2 / §8.10). It exposes a
`forward_image_batch(rgb)` helper to make the flattening of
(B, T, H, W, C) → (B*T, H, W, C) explicit rather than buried inside
`__call__`.

**Pretrained weight loading:** the published `.jim` checkpoint was made with
flax<=0.8.x where nnx.Param leaves serialized with a `raw_value` key. In
flax >=0.10 those became `value`. We bypass jimmy's `load_model(pretrained=
True)` path and adapt the saved tree ourselves to keep the encoder
loadable on the current stack.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import jax
import jax.image
import jax.numpy as jnp
import py7zr
from flax import nnx

# ImageNet RGB stats (DINOv2 was trained with these).
_DINO_MEAN = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
_DINO_STD  = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)

# Hosted by jimmy's author per PROJECT_PLAN_1.md §2.2.1. The agent should
# verify periodically; if the URL moves, point this at any orbax checkpoint
# whose state-tree shape matches DINOV2_VITB14.
DEFAULT_WEIGHT_URL = (
    "https://huggingface.co/poiretclement/dinov2_jax/resolve/main/dinov2_vitb14.jim"
)

# Output dim of ViT-B/14. 768 wires through to the VIS projection head's
# input dim (cotrain.models.heads.projection.DINO_FEATURE_DIM).
EMBED_DIM = 768

# DINOv2 was trained at 518×518; we upscale dataset RGB (224×224) to this
# size at the encoder boundary. Validated at fp64: feeding native 518
# directly gives max|Δ|=1.3e-12 vs PyTorch DINOv2.
ENCODER_INPUT_SIZE = 518


class DinoV2Encoder(nnx.Module):
    """NNX wrapper around jimmy's pretrained DinoV2 ViT-B/14.

    Construct once per process; reuse for every batch. The pretrained download
    is cached by jimmy under `~/.cache/jimmy/`, so first instantiation pays
    the download cost and subsequent instantiations are fast.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        pretrained: bool = True,
        url: str | None = DEFAULT_WEIGHT_URL,
        cache_dir: str | Path | None = None,
    ) -> None:
        # Local import so the cotrain._compat shim runs first via package
        # __init__ before jimmy is touched. Direct top-level import would
        # bypass the shim if a downstream caller imports this module without
        # importing `cotrain` itself.
        from jimmy.models import DINOV2_VITB14
        from jimmy.models.vit import DinoV2

        cfg = dict(DINOV2_VITB14["config"])
        self.backbone = DinoV2(
            **cfg,
            # Allow 224 input vs 518 training: jimmy interpolates the
            # learned 37x37 position-embedding grid down to 16x16 at
            # forward time. Same mechanism PyTorch DINOv2 uses.
            dynamic_img_size=True,
            rngs=rngs,
        )
        self.embed_dim = EMBED_DIM

        if pretrained:
            if url is None:
                raise ValueError("pretrained=True requires a weight URL")
            ckpt_dir = _ensure_checkpoint(name=DINOV2_VITB14["name"], url=url, cache_dir=cache_dir)
            _restore_pretrained(self.backbone, ckpt_dir)

        # PyTorch DINOv2 uses nn.GELU(approximate='none'); jimmy uses the
        # tanh approximation by default. Override per-block to match — see
        # the §2.2.1 parity discussion in the module docstring.
        self._patch_blocks_to_exact_gelu(depth=DINOV2_VITB14["config"]["depth"])

    def _patch_blocks_to_exact_gelu(self, *, depth: int) -> None:
        def _exact_gelu(x: jnp.ndarray) -> jnp.ndarray:
            return jax.nn.gelu(x, approximate=False)
        for i in range(depth):
            block = getattr(self.backbone, f"blocks.{i}")
            block.mlp.act = _exact_gelu

    def normalize(self, rgb_uint8: jnp.ndarray) -> jnp.ndarray:
        """uint8 (..., H, W, 3) -> float32 (..., H, W, 3) with DINO mean/std.

        Pure normalization — no resizing. Use `preprocess` for the full
        encoder-input-prep pipeline."""
        x = rgb_uint8.astype(jnp.float32) / 255.0
        return (x - _DINO_MEAN) / _DINO_STD

    def _resize_to_input(self, x: jnp.ndarray) -> jnp.ndarray:
        """Bilinear-upscale (..., H, W, 3) to (..., 518, 518, 3). No-op when
        the input is already at the encoder's expected resolution.

        Bilinear matches torch.nn.functional.interpolate(mode='bilinear',
        align_corners=False) to ~1e-6 at fp32 (verified empirically), so
        feeding the same uint8 RGB through both backends gives equivalent
        encoder inputs."""
        if x.shape[-3:-1] == (ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE):
            return x
        leading = x.shape[:-3]
        flat = x.reshape((-1,) + x.shape[-3:])
        flat = jax.image.resize(
            flat,
            (flat.shape[0], ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE, 3),
            method="bilinear",
        )
        return flat.reshape(leading + (ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE, 3))

    def preprocess(self, rgb: jnp.ndarray) -> jnp.ndarray:
        """Full encoder-input prep: optional uint8→float32, normalize, upscale
        to 518. Idempotent — passing already-preprocessed float32 (..., 518,
        518, 3) returns the input unchanged."""
        x = self.normalize(rgb) if rgb.dtype == jnp.uint8 else rgb
        return self._resize_to_input(x)

    def __call__(self, rgb: jnp.ndarray) -> jnp.ndarray:
        """Forward over a batch of images.

        rgb: (B, H, W, 3) uint8 (any spatial size) or float32 already
        preprocessed at 518. Returns: (B, 768) mean-pooled patch tokens.
        """
        x = self.preprocess(rgb) if rgb.dtype == jnp.uint8 else rgb
        tokens = self.backbone(x)            # (B, 1 + N_patch, 768)
        return tokens[:, 1:].mean(axis=1)    # drop [CLS], mean over patches

    def forward_image_batch(self, rgb: jnp.ndarray) -> jnp.ndarray:
        """Forward over a (B, T, H, W, 3) clip — flattens, encodes, reshapes.

        This is the path the trainer uses: a window of T frames per sample
        gets encoded as a single (B*T, ...) batch through the DINO forward,
        then reshaped back to (B, T, 768) before the VIS projection head."""
        if rgb.ndim != 5:
            raise ValueError(f"expected (B, T, H, W, 3); got shape {rgb.shape}")
        B, T, H, W, C = rgb.shape
        flat = rgb.reshape(B * T, H, W, C)
        pooled = self(flat)                  # (B*T, 768)
        return pooled.reshape(B, T, EMBED_DIM)


# --- Pretrained weight loading --------------------------------------------

_DEFAULT_CACHE_DIR = Path.home() / ".jimmy" / "hub" / "checkpoints"


def _ensure_checkpoint(*, name: str, url: str, cache_dir: str | Path | None) -> Path:
    """Download (if missing) and decompress the .jim checkpoint.

    Mirrors jimmy's loader cache layout so we don't re-download what jimmy
    has already pulled."""
    cache_root = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cache_root / name
    compressed = cache_root / f"{name}.jim"

    if not ckpt_dir.exists() and not compressed.exists():
        logging.getLogger(__name__).info("downloading %s -> %s", url, compressed)
        urlretrieve(url, compressed)
    if not ckpt_dir.exists() and compressed.exists():
        with py7zr.SevenZipFile(compressed, mode="r") as archive:
            archive.extractall(cache_root)
    return ckpt_dir


def _unwrap_leaf_key(tree: Any, leaf_key: str) -> Any:
    """Recursively replace `{leaf_key: array}` with `array` in a nested dict.

    Older Flax/NNX wrapped each nnx.Param value in a singleton dict like
    `{"raw_value": array}` when serializing. Current NNX expects raw arrays
    at those positions — `nnx.update` then assigns them straight into each
    Param's `.value` attribute. Without this unwrap, `nnx.update` assigns
    the whole `{value: array}` dict to `Param.value`, and the forward pass
    crashes downstream when JAX tries to coerce a dict to a tensor."""
    if isinstance(tree, dict):
        if len(tree) == 1 and leaf_key in tree:
            return tree[leaf_key]
        return {k: _unwrap_leaf_key(v, leaf_key) for k, v in tree.items()}
    return tree


def _restore_pretrained(model: nnx.Module, ckpt_dir: Path) -> None:
    """Load the saved params into `model` in place."""
    import orbax.checkpoint as ocp
    raw = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
    if not isinstance(raw, dict) or "params" not in raw:
        raise RuntimeError(f"checkpoint at {ckpt_dir} missing 'params' key; got {type(raw)}")
    params = _unwrap_leaf_key(raw["params"], "raw_value")
    nnx.update(model, params)
