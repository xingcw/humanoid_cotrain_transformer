"""DINOv2 parity check — PyTorch reference vs jimmy NNX (PROJECT_PLAN_1.md §2.2.1).

The plan calls this out as **mandatory before §8 step 13** (the scaled
training run): feed 32 identical images through both implementations, take
the patch-token outputs, and assert max-abs-error < 1e-3.

Two algorithmic mismatches blocked parity in earlier versions:
  1. PyTorch DINOv2 uses exact erf-based GELU; jimmy defaulted to the
     tanh approximation. DinoV2Encoder now patches every block's mlp.act
     to the exact form at construction.
  2. PyTorch DINOv2's `interpolate_pos_encoding` uses the historical
     `interpolate_offset=0.1` scale-factor mode; jimmy uses
     `jax.image.resize` size mode. We sidestep this by feeding both
     backends at the trained 518×518 resolution; the upscale 224→518 is
     the encoder's first step in production, so this matches reality.

Validated 2026-04-29: at 518 / fp64 / exact GELU, max|Δ|=1.3e-12.

Usage:
    python -m cotrain.eval.parity_dino [--n-images 32] [--seed 0] [--strict] [--fp64]

Without --strict, prints all metrics and exits 0 regardless (informational).
With --strict, exits non-zero when max|Δ| >= PARITY_TOL — use this gate
right before launching a scaled training run.
With --fp64, runs both backends at fp64 to verify the algorithmic chain
matches up to fp64 noise floor (~1e-12). Slower; use small --n-images.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

# Importing cotrain installs the jimmy/flax compat shim; do it before any
# jimmy submodule is touched (transitively via the encoder import).
import cotrain  # noqa: F401
from cotrain.models.encoders import ENCODER_INPUT_SIZE, DinoV2Encoder
from flax import nnx

PARITY_TOL = 1e-3


@dataclass
class ParityResult:
    n_images: int
    max_abs_err: float
    mean_abs_err: float
    cosine_min: float
    cosine_mean: float
    passed: bool

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] n={self.n_images} | "
            f"max|Δ|={self.max_abs_err:.3e} (tol {PARITY_TOL:.0e}) | "
            f"mean|Δ|={self.mean_abs_err:.3e} | "
            f"cos∈[{self.cosine_min:.4f}, …] (mean {self.cosine_mean:.4f})"
        )


def _make_images(n: int, seed: int, *, size: int = ENCODER_INPUT_SIZE) -> np.ndarray:
    """Deterministic uint8 RGB clip the two backends can agree on.

    Default 518×518 — DINOv2's trained resolution. Bypasses PyTorch's
    `interpolate_pos_encoding` and jimmy's `resample_pos_embed` (the two
    don't agree exactly), so this measures *only* the DINO model parity."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, size, size, 3)).astype(np.uint8)


def _torch_patch_tokens(images_uint8: np.ndarray, *, fp64: bool) -> np.ndarray:
    """Run PyTorch DINOv2 ViT-B/14 on CPU; return patch tokens (B, N_patch, 768)."""
    import torch

    torch.manual_seed(0)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model.eval()
    dtype = torch.float64 if fp64 else torch.float32
    if fp64:
        model = model.double()

    # PyTorch wants NCHW float32/64, ImageNet-normalized.
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=dtype).view(1, 3, 1, 1)
    x = torch.from_numpy(images_uint8).to(dtype) / 255.0  # (B, H, W, C)
    x = x.permute(0, 3, 1, 2)                              # (B, C, H, W)
    x = (x - mean) / std

    with torch.no_grad():
        out = model.forward_features(x)
    return out["x_norm_patchtokens"].numpy()               # (B, N_patch, 768)


def _jax_patch_tokens(images_uint8: np.ndarray, *, fp64: bool) -> np.ndarray:
    """Run jimmy-NNX DinoV2 ViT-B/14 on TPU; return patch tokens."""
    enc = DinoV2Encoder(rngs=nnx.Rngs(0), pretrained=True)
    if fp64:
        # Cast every Param to fp64 in place.
        state = nnx.state(enc, nnx.Param)
        state = jax.tree.map(
            lambda s: s.astype(jnp.float64)
            if hasattr(s, "dtype") and s.dtype == jnp.float32 else s,
            state,
        )
        nnx.update(enc, state)
    rgb = jnp.asarray(images_uint8)
    if fp64:
        # Match torch's normalize-in-fp64 path: cast to fp64 before mean/std.
        x = rgb.astype(jnp.float64) / 255.0
        mean = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float64)
        std = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float64)
        x = (x - mean) / std
    else:
        x = enc.normalize(rgb)
    # Mirror DinoV2Encoder.__call__ but keep patch tokens (don't pool).
    x = enc._resize_to_input(x)
    tokens = enc.backbone(x)                               # (B, 1+N_patch, 768)
    return np.asarray(tokens[:, 1:])                       # drop [CLS]


def run_parity(n_images: int = 32, seed: int = 0, *, fp64: bool = False) -> ParityResult:
    if fp64:
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
        jax.config.update("jax_enable_x64", True)
    else:
        os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

    images = _make_images(n_images, seed)
    print(f"running parity on {n_images} images "
          f"({images.shape[1]}x{images.shape[2]} uint8, "
          f"{'fp64' if fp64 else 'fp32'})...")

    print("  -> PyTorch DINOv2 (CPU)")
    tor = _torch_patch_tokens(images, fp64=fp64)

    print("  -> jimmy NNX DinoV2 (TPU)")
    jx = _jax_patch_tokens(images, fp64=fp64)

    if tor.shape != jx.shape:
        raise RuntimeError(f"shape mismatch: torch {tor.shape} vs jax {jx.shape}")

    diff = np.abs(tor - jx)
    max_abs_err = float(diff.max())
    mean_abs_err = float(diff.mean())

    # Cosine similarity per (B, patch) feature vector — picks up direction
    # mismatch even when magnitudes happen to align.
    a = tor.reshape(-1, tor.shape[-1])
    b = jx.reshape(-1, jx.shape[-1])
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    cos = (a_n * b_n).sum(axis=-1)
    cos_min = float(cos.min())
    cos_mean = float(cos.mean())

    return ParityResult(
        n_images=n_images,
        max_abs_err=max_abs_err,
        mean_abs_err=mean_abs_err,
        cosine_min=cos_min,
        cosine_mean=cos_mean,
        passed=max_abs_err < PARITY_TOL,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-images", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when max|Δ| >= 1e-3. Use this before launching "
             "a scaled training run; without it the script is informational.",
    )
    ap.add_argument(
        "--fp64",
        action="store_true",
        help="Run both backends at fp64. Verifies the algorithmic chain "
             "matches at fp64 noise floor (~1e-12). Slower; pair with "
             "--n-images 1 or 2.",
    )
    args = ap.parse_args()
    result = run_parity(args.n_images, args.seed, fp64=args.fp64)
    print(result)
    if args.strict and not result.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
