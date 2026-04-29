"""Tests for §3.2 token interleaving and slot/time embeddings.

The plan §3.3 invariant we're protecting: 'slot identity is preserved by
position alone — no operation may scramble the slot order'. The tests
encode (slot_id, time_id) into the token *values* and verify that the
interleaved sequence has the right value at every position.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from cotrain.models.transformer import (
    NUM_SLOTS,
    SLOT_ORDER,
    SlotTimeEmbeds,
    interleave_slot_tokens,
)


def _tagged_tokens(B: int, T: int, d_model: int) -> dict[str, jnp.ndarray]:
    """Build per-slot tokens whose first two dims encode (slot_id, time_id).

    Token at slot s, time t, batch b has value vector [s, t, 0, 0, ...] —
    so we can read (slot, time) back out of the interleaved sequence.
    """
    out = {}
    for s, name in enumerate(SLOT_ORDER):
        # tags[t] = [s, t, 0, ...]
        tags = np.zeros((T, d_model), dtype=np.float32)
        tags[:, 0] = s
        tags[:, 1] = np.arange(T, dtype=np.float32)
        # Broadcast across batch.
        out[name] = jnp.asarray(np.broadcast_to(tags, (B, T, d_model)))
    return out


def test_interleave_preserves_slot_and_time_order() -> None:
    """For every flat position p, the token must encode (p % S, p // S)."""
    B, T, D = 2, 4, 8
    tokens = _tagged_tokens(B, T, D)
    seq = np.asarray(interleave_slot_tokens(tokens))
    assert seq.shape == (B, T * NUM_SLOTS, D)
    for p in range(T * NUM_SLOTS):
        slot_id = p % NUM_SLOTS
        time_id = p // NUM_SLOTS
        # Same value across the batch, by construction.
        np.testing.assert_array_equal(seq[:, p, 0], slot_id)
        np.testing.assert_array_equal(seq[:, p, 1], time_id)


def test_interleave_missing_slot_raises() -> None:
    B, T, D = 1, 2, 4
    tokens = _tagged_tokens(B, T, D)
    del tokens["box"]
    with pytest.raises(KeyError, match="box"):
        interleave_slot_tokens(tokens)


def test_slot_time_embeds_shape_and_addition() -> None:
    """Output shape (B, 6T, d_model); embeddings are additive (zero token in
    -> the output equals slot_emb + time_emb at every position)."""
    B, T, D = 3, 5, 16
    embeds = SlotTimeEmbeds(T=T, d_model=D, rngs=nnx.Rngs(0))
    zeros = {s: jnp.zeros((B, T, D), dtype=jnp.float32) for s in SLOT_ORDER}
    out = embeds(zeros)
    assert out.shape == (B, T * NUM_SLOTS, D)
    out_np = np.asarray(out)
    # Every batch row should be identical (no token data, only embeddings).
    np.testing.assert_allclose(out_np[0], out_np[1], atol=1e-6)
    np.testing.assert_allclose(out_np[1], out_np[2], atol=1e-6)
    # Position 0 should equal slot_emb[0] + time_emb[0].
    expected_pos0 = (
        np.asarray(embeds.slot_emb(jnp.array(0)))
        + np.asarray(embeds.time_emb(jnp.array(0)))
    )
    np.testing.assert_allclose(out_np[0, 0], expected_pos0, atol=1e-6)


def test_slot_time_embeds_shape_mismatch_raises() -> None:
    B, T, D = 2, 4, 8
    embeds = SlotTimeEmbeds(T=T, d_model=D, rngs=nnx.Rngs(0))
    # Wrong T:
    bad = {s: jnp.zeros((B, T + 1, D), dtype=jnp.float32) for s in SLOT_ORDER}
    with pytest.raises(ValueError, match="sequence length"):
        embeds(bad)
    # Wrong d_model:
    bad = {s: jnp.zeros((B, T, D + 1), dtype=jnp.float32) for s in SLOT_ORDER}
    with pytest.raises(ValueError, match="d_model"):
        embeds(bad)


def test_runs_under_nnx_jit() -> None:
    B, T, D = 2, 4, 16
    embeds = SlotTimeEmbeds(T=T, d_model=D, rngs=nnx.Rngs(0))

    @nnx.jit
    def fwd(model: SlotTimeEmbeds, tokens) -> jnp.ndarray:
        return model(tokens)

    tokens = _tagged_tokens(B, T, D)
    out = fwd(embeds, tokens)
    assert out.shape == (B, T * NUM_SLOTS, D)


def test_end_to_end_with_projection_heads() -> None:
    """Round-trip: read a window's per-slot data, project through the §2.2
    heads, interleave with embeds, and verify shape (B, 6T, d_model)."""
    from cotrain.data.schemas import DEFAULT_D_A, DEFAULT_D_H, DEFAULT_D_P
    from cotrain.models.heads import ProjectionHeads
    from cotrain.models.heads.projection import DINO_FEATURE_DIM

    B, T, D = 2, 4, 32
    rng = np.random.default_rng(0)
    # Simulate post-encoder DINO features for vis (no real DINO call).
    batch = {
        "vis":         jnp.asarray(rng.normal(size=(B, T, DINO_FEATURE_DIM)).astype(np.float32)),
        "state_robot": jnp.asarray(rng.normal(size=(B, T, DEFAULT_D_P)).astype(np.float32)),
        "state_human": jnp.asarray(rng.normal(size=(B, T, DEFAULT_D_H)).astype(np.float32)),
        "box":         jnp.asarray(rng.normal(size=(B, T, 7)).astype(np.float32)),
        "phase":       jnp.asarray(rng.integers(0, 5, size=(B, T, 1)).astype(np.int8)),
        "contact":     jnp.asarray(rng.integers(0, 2, size=(B, T, 3)).astype(np.float32)),
        "action":      jnp.asarray(rng.normal(size=(B, T, DEFAULT_D_A)).astype(np.float32)),
        "source_mask": jnp.asarray([True, False]),
    }
    heads = ProjectionHeads(
        d_model=D, D_p=DEFAULT_D_P, D_h=DEFAULT_D_H, D_a=DEFAULT_D_A,
        rngs=nnx.Rngs(0),
    )
    embeds = SlotTimeEmbeds(T=T, d_model=D, rngs=nnx.Rngs(1))

    projected = heads(**batch)                    # dict slot -> (B, T, D)
    seq = embeds(projected)                       # (B, 6T, D)
    assert seq.shape == (B, T * NUM_SLOTS, D)
    assert seq.dtype == jnp.float32
