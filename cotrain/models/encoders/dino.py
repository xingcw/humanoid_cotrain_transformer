"""Frozen DINOv2 ViT-B/14 visual encoder (PROJECT_PLAN_1.md §2.2.1).

We load the model via the `jimmy` library (`clementpoiret/jimmy`), which
ships a native Flax NNX implementation. The pretrained weights were trained
at 518×518 resolution; we feed 224×224 (the dataset's native resolution) and
let jimmy's `dynamic_img_size=True` interpolate the trained 37×37 position
embedding grid down to 16×16 at runtime. PyTorch DINOv2 has the same
mechanism so the parity check is apples-to-apples.

Output: (B, T, 768) — mean-pooled patch tokens. We drop the [CLS] token
because the VIS slot wants a holistic image descriptor, and DINOv2 patch
mean has been shown to outperform the [CLS] alone for dense downstream
tasks (this is also what the plan calls for in §2.2 -- "mean-pool patch
tokens").

Frozen forward: this module owns nnx.Param leaves but the transformer
training step uses `nnx.split(model, nnx.Param, ...)` and will exclude this
encoder's params via the wrapper logic in cotrain.training.trainer (§8.5).
This module exposes a `forward_image_batch(rgb)` helper to make the
flattening of (B, T, H, W, C) → (B*T, H, W, C) and the pooling explicit
rather than buried inside `__call__`.

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

    def normalize(self, rgb_uint8: jnp.ndarray) -> jnp.ndarray:
        """uint8 (..., H, W, 3) -> float32 (..., H, W, 3) with DINO mean/std."""
        x = rgb_uint8.astype(jnp.float32) / 255.0
        return (x - _DINO_MEAN) / _DINO_STD

    def __call__(self, rgb: jnp.ndarray) -> jnp.ndarray:
        """Forward over a batch of images.

        rgb: (B, H, W, 3) uint8 OR float32 already-normalized.
        returns: (B, 768) mean-pooled patch tokens.
        """
        x = self.normalize(rgb) if rgb.dtype == jnp.uint8 else rgb
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
