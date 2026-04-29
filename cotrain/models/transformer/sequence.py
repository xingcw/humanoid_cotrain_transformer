"""Token sequence assembly for the §3.2 transformer (PROJECT_PLAN_1.md §2.1, §3.2).

Per timestep, the model consumes a fixed-order 6-token sequence:

    [VIS_t] [STATE_t] [BOX_t] [PHASE_t] [CONTACT_t] [ACTION_t]

Across a window of T timesteps, the full input is the **time-outer,
slot-inner** flattening (§2.1):

    [VIS_0, STATE_0, BOX_0, PHASE_0, CONTACT_0, ACTION_0,
     VIS_1, STATE_1, ..., ACTION_{T-1}]

so that flat position `p` decodes to `slot_id = p % 6, time_id = p // 6`.
We pin this convention here in `SLOT_ORDER` and use it everywhere — the
output heads in §2.3 read from these same slot positions, so any reordering
silently breaks training (§3.3).

This module provides:
- `SLOT_ORDER`: the canonical slot order. Don't reorder.
- `interleave_slot_tokens`: pure-jax helper that takes the per-slot
  projected tokens and produces (B, 6T, d_model) without any embeddings —
  used by the masking module's tests for slot identity.
- `SlotTimeEmbeds`: NNX module that wraps `interleave_slot_tokens` plus the
  learned slot (6) and time (T) embeddings (§3.2 #3).
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

# Canonical slot order. Used by the projection heads, the interleaver, and
# the output heads. Frozen tuple so it can't be mutated at import time.
SLOT_ORDER: tuple[str, ...] = ("vis", "state", "box", "phase", "contact", "action")
NUM_SLOTS: int = len(SLOT_ORDER)


def interleave_slot_tokens(tokens_per_slot: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Interleave 6 per-slot tensors of shape (B, T, d_model) into (B, 6T, d_model).

    The result is **time-outer, slot-inner**: position `t*6 + s` holds the
    slot-`s` token at time `t`. Uses the canonical `SLOT_ORDER`."""
    missing = [s for s in SLOT_ORDER if s not in tokens_per_slot]
    if missing:
        raise KeyError(f"missing slot(s): {missing}")
    # Stack along a new "slot" axis right after T, then merge T and slot.
    stacked = jnp.stack([tokens_per_slot[s] for s in SLOT_ORDER], axis=2)
    B, T, S, D = stacked.shape
    return stacked.reshape(B, T * S, D)


def _embed_init():
    """Small-stddev init for slot/time embeddings — they're additive on top
    of projected tokens, so they shouldn't dominate."""
    return nnx.with_partitioning(nnx.initializers.normal(stddev=0.02), (None, "model"))


class SlotTimeEmbeds(nnx.Module):
    """Add learned slot-id and time-id embeddings to an interleaved sequence.

    Takes a `tokens_per_slot` dict (slot -> (B, T, d_model)), interleaves to
    (B, 6T, d_model), and adds slot + time embeds. The shapes are pinned at
    construction; passing tokens with a different T or slot count is a hard
    error rather than a silent broadcast.
    """

    def __init__(
        self,
        *,
        T: int,
        d_model: int,
        num_slots: int = NUM_SLOTS,
        rngs: nnx.Rngs,
    ) -> None:
        self.T = T
        self.d_model = d_model
        self.num_slots = num_slots
        self.slot_emb = nnx.Embed(num_slots, d_model, rngs=rngs,
                                   embedding_init=_embed_init())
        self.time_emb = nnx.Embed(T, d_model, rngs=rngs,
                                   embedding_init=_embed_init())

    def __call__(self, tokens_per_slot: dict[str, jnp.ndarray]) -> jnp.ndarray:
        seq = interleave_slot_tokens(tokens_per_slot)         # (B, 6T, d_model)
        B, L, D = seq.shape
        if D != self.d_model:
            raise ValueError(f"d_model mismatch: tokens {D} vs embeds {self.d_model}")
        if L != self.T * self.num_slots:
            raise ValueError(
                f"sequence length {L} != T*num_slots = {self.T * self.num_slots}"
            )
        positions = jnp.arange(L)
        slot_ids = positions % self.num_slots                 # [0..S-1] repeating
        time_ids = positions // self.num_slots                # [0..T-1] each repeated S
        slot_vecs = self.slot_emb(slot_ids)                   # (L, d_model)
        time_vecs = self.time_emb(time_ids)                   # (L, d_model)
        return seq + slot_vecs[None, :, :] + time_vecs[None, :, :]
