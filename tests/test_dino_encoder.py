"""Smoke tests for the §2.2.1 DINOv2 encoder wrapper.

These run with `pretrained=False` so they don't depend on network access or
the ~330 MB checkpoint download. The full pretrained-weights path is
exercised by `cotrain.eval.parity_dino` and run on demand by the user.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import cotrain  # noqa: F401  (installs the jimmy/flax compat shim)
from cotrain.models.encoders import EMBED_DIM, DinoV2Encoder
from cotrain.models.heads.projection import DINO_FEATURE_DIM


def _make_encoder() -> DinoV2Encoder:
    return DinoV2Encoder(rngs=nnx.Rngs(0), pretrained=False)


def test_embed_dim_matches_projection_head_input() -> None:
    """Wire-check: encoder output dim must equal what the VIS head expects."""
    assert EMBED_DIM == DINO_FEATURE_DIM == 768


def test_image_batch_shape_and_dtype() -> None:
    """(B, H, W, 3) uint8 -> (B, 768) float32."""
    enc = _make_encoder()
    rgb = jnp.asarray(np.random.default_rng(0).integers(0, 256,
                       size=(2, 224, 224, 3)).astype(np.uint8))
    out = enc(rgb)
    assert out.shape == (2, EMBED_DIM)
    assert out.dtype == jnp.float32


def test_clip_batch_shape() -> None:
    """(B, T, H, W, 3) uint8 -> (B, T, 768) — the trainer's call site."""
    enc = _make_encoder()
    rgb = jnp.asarray(np.random.default_rng(1).integers(0, 256,
                       size=(2, 4, 224, 224, 3)).astype(np.uint8))
    out = enc.forward_image_batch(rgb)
    assert out.shape == (2, 4, EMBED_DIM)
    assert out.dtype == jnp.float32


def test_rejects_wrong_rank() -> None:
    enc = _make_encoder()
    with pytest.raises(ValueError, match="expected"):
        enc.forward_image_batch(jnp.zeros((2, 224, 224, 3), dtype=jnp.uint8))


def test_runs_under_nnx_jit() -> None:
    """Same trap as the projection heads (§3.4): silent breakage on Python
    branching is the failure mode. A single nnx.jit call is the cheap check."""
    enc = _make_encoder()

    @nnx.jit
    def fwd(model: DinoV2Encoder, rgb: jnp.ndarray) -> jnp.ndarray:
        return model(rgb)

    rgb = jnp.asarray(np.random.default_rng(2).integers(0, 256,
                       size=(2, 224, 224, 3)).astype(np.uint8))
    out = fwd(enc, rgb)
    assert out.shape == (2, EMBED_DIM)


def test_normalize_uint8_to_normalized_float32() -> None:
    """Mid-gray (128) -> per-channel normalized values, channel-aware."""
    enc = _make_encoder()
    rgb = jnp.full((1, 224, 224, 3), 128, dtype=jnp.uint8)
    out = enc.normalize(rgb)
    assert out.dtype == jnp.float32
    # ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
    # 128/255 = 0.502 -> per-channel: (~0.074, ~0.205, ~0.427).
    per_channel = np.asarray(out).mean(axis=(0, 1, 2))
    expected = np.array([
        (128/255 - 0.485) / 0.229,
        (128/255 - 0.456) / 0.224,
        (128/255 - 0.406) / 0.225,
    ])
    np.testing.assert_allclose(per_channel, expected, atol=1e-5)


def test_param_tree_includes_encoder_when_split() -> None:
    """Until §8.5 wires the freeze-isolation, encoder Params are visible. The
    test pins this so we'll notice when the wiring changes."""
    enc = _make_encoder()
    state = nnx.state(enc, nnx.Param)
    leaves = jax.tree_util.tree_leaves(state)
    # ViT-B/14: depth=12 blocks, each with multiple Params; expect well over 100.
    assert len(leaves) >= 100, f"expected many Param leaves, got {len(leaves)}"


def test_preprocess_upscales_224_to_518() -> None:
    """Encoder upscales 224×224 uint8 inputs to the trained 518×518 size."""
    from cotrain.models.encoders import ENCODER_INPUT_SIZE
    enc = _make_encoder()
    rgb = jnp.asarray(np.random.default_rng(0).integers(
        0, 256, size=(2, 224, 224, 3)).astype(np.uint8))
    out = enc.preprocess(rgb)
    assert out.shape == (2, ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE, 3)
    assert out.dtype == jnp.float32


def test_preprocess_is_idempotent_at_target_size() -> None:
    """Already-preprocessed inputs at 518×518 pass through unchanged."""
    from cotrain.models.encoders import ENCODER_INPUT_SIZE
    enc = _make_encoder()
    pre = jnp.asarray(np.random.default_rng(1).normal(
        size=(1, ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE, 3)).astype(np.float32))
    # Preprocess on already-fp32-and-correct-size input is just the resize
    # no-op (no normalize since it's not uint8).
    out = enc._resize_to_input(pre)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(pre))


def test_block_mlp_act_patched_to_exact_gelu() -> None:
    """Every block's MLP must use the erf-based exact GELU to match the
    PyTorch DINOv2 reference (PROJECT_PLAN_1.md §2.2.1)."""
    enc = _make_encoder()
    # Probe a known-large input where exact and approximate GELU disagree.
    x_test = jnp.array([16.0, -16.0], dtype=jnp.float32)
    expected = jax.nn.gelu(x_test, approximate=False)
    for i in range(12):
        block = getattr(enc.backbone, f"blocks.{i}")
        np.testing.assert_allclose(
            np.asarray(block.mlp.act(x_test)), np.asarray(expected),
            rtol=0, atol=1e-6,
            err_msg=f"block {i} mlp.act not exact GELU",
        )
