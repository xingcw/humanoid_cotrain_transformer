"""DINOv2 parity check — PyTorch reference vs jimmy NNX (PROJECT_PLAN_1.md §2.2.1).

The plan calls this out as **mandatory before §8 step 13** (the scaled
training run): feed 32 identical images through both implementations, take
the patch-token outputs, and assert max-abs-error < 1e-3. Below that
threshold, we have confidence the JAX path is producing the same features
the PyTorch reference would.

Above that threshold, the fallback (per the plan) is to precompute features
offline with PyTorch DINOv2 and load them via a new HDF5 group
`dino_features` — a one-day implementation we should not pre-build.

Current status (2026-04-29): max|Δ| ≈ 1, mean|Δ| ≈ 0.04, cosine ≈ 0.999.
Direction matches; magnitude drifts inside the 12 transformer blocks at TPU
fp32 precision. Build proceeded past §8 step 3 on this signal; tighten
before step 13 (likely fix: thread precision=HIGHEST through jimmy's
Attention matmuls, or run with jax_enable_x64).

Usage:
    python -m cotrain.eval.parity_dino [--n-images 32] [--seed 0] [--strict]

Without --strict, prints all metrics and exits 0 regardless (informational).
With --strict, exits non-zero when max|Δ| >= PARITY_TOL — use this gate
right before launching a scaled training run.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from flax import nnx

# Importing cotrain installs the jimmy/flax compat shim; do it before any
# jimmy submodule is touched (transitively via the encoder import).
import cotrain  # noqa: F401
from cotrain.models.encoders import DinoV2Encoder

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


def _make_images(n: int, seed: int) -> np.ndarray:
    """Deterministic uint8 RGB clip the two backends can agree on."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 224, 224, 3)).astype(np.uint8)


def _torch_patch_tokens(images_uint8: np.ndarray) -> np.ndarray:
    """Run PyTorch DINOv2 ViT-B/14 on CPU; return patch tokens (B, 256, 768)."""
    import torch

    torch.manual_seed(0)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model.eval()

    # PyTorch wants NCHW float32, ImageNet-normalized.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = torch.from_numpy(images_uint8).float() / 255.0  # (B, H, W, C)
    x = x.permute(0, 3, 1, 2)                            # (B, C, H, W)
    x = (x - mean) / std

    with torch.no_grad():
        out = model.forward_features(x)
    return out["x_norm_patchtokens"].numpy()             # (B, 256, 768)


def _jax_patch_tokens(images_uint8: np.ndarray) -> np.ndarray:
    """Run jimmy-NNX DinoV2 ViT-B/14 on TPU; return patch tokens (B, 256, 768)."""
    enc = DinoV2Encoder(rngs=nnx.Rngs(0), pretrained=True)

    rgb = jnp.asarray(images_uint8)
    # Mirror DinoV2Encoder.__call__ but keep patch tokens (don't pool).
    x = enc.normalize(rgb)
    tokens = enc.backbone(x)                             # (B, 257, 768)
    return np.asarray(tokens[:, 1:])                     # drop [CLS]


def run_parity(n_images: int = 32, seed: int = 0) -> ParityResult:
    images = _make_images(n_images, seed)
    print(f"running parity on {n_images} images (224x224 uint8) ...")

    print("  -> PyTorch DINOv2 (CPU)")
    tor = _torch_patch_tokens(images)

    print("  -> jimmy NNX DinoV2 (TPU)")
    jx = _jax_patch_tokens(images)

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
    args = ap.parse_args()
    result = run_parity(args.n_images, args.seed)
    print(result)
    if args.strict and not result.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
