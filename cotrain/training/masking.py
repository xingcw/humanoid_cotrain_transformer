"""Modality masking — the heart of the shared bridge (PROJECT_PLAN_1.md §3.4).

Two responsibilities:

1. **Input-side swap.** Replace projected VIS tokens with `vis_mask_token`
   for robot samples; replace projected ACTION tokens with
   `action_mask_token` for human samples. Both swaps use jnp.where keyed on
   `source_mask` — never Python-level `if source == "robot"`, which silently
   breaks under nnx.jit (the plan flags this as the #1 bug-prone path).

2. **Loss-side mask.** Per §3.4 / §3.5:
     - VIS: loss mask = 0 everywhere (next-frame DINO prediction is skipped).
     - STATE / BOX / PHASE / CONTACT: loss mask = 1 everywhere (bridge slots
       are supervised in *both* sources).
     - ACTION: loss mask = 1 for robot rows, 0 for human rows.

   The trainer multiplies the per-token loss by these masks. Zero
   contribution from human ACTION predictions and from any VIS prediction
   is what closes the loop on the masked-modality scheme of Radosavovic et
   al. — the transformer learns that when slot 0 is `[VIS_MASK]` the
   action prediction must rely on slots 1–4 (state + bridge), and when
   slot 5 is `[MASK_ACTION]` the box / phase / contact predictions must
   explain themselves through slots 0–4.

The function is jit-clean (no host-side branching, no in-place mutation).
"""
from __future__ import annotations

import jax.numpy as jnp


def apply_modality_masks(
    tokens: dict[str, jnp.ndarray],
    source_mask: jnp.ndarray,
    vis_mask_token: jnp.ndarray,
    action_mask_token: jnp.ndarray,
) -> tuple[dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
    """Apply the §3.4 input swap and produce per-slot loss masks.

    Args:
      tokens: dict of slot_name -> (B, T, d_model) projected tokens. Must
        contain at least ``vis`` and ``action``; other slots are passed
        through unchanged.
      source_mask: (B,) bool. True for robot samples, False for human.
      vis_mask_token: (d_model,) — usually an nnx.Param value, broadcast
        in for robot rows.
      action_mask_token: (d_model,) — broadcast in for human rows.

    Returns:
      (masked_tokens, loss_masks).
      - ``masked_tokens`` is a new dict with the same keys as ``tokens``;
        only ``vis`` and ``action`` differ.
      - ``loss_masks`` maps each *predicted* slot name to a (B, T) float32
        tensor in {0, 1} that the trainer multiplies into per-token losses.
    """
    if source_mask.dtype != jnp.bool_:
        source_mask = source_mask.astype(jnp.bool_)
    if "vis" not in tokens or "action" not in tokens:
        raise KeyError("tokens dict must contain 'vis' and 'action' slots")

    B = source_mask.shape[0]
    T = tokens["vis"].shape[1]
    is_robot = source_mask[:, None, None]                                # (B, 1, 1)

    vis_replaced = jnp.broadcast_to(vis_mask_token, tokens["vis"].shape)
    act_replaced = jnp.broadcast_to(action_mask_token, tokens["action"].shape)

    masked = dict(tokens)
    masked["vis"]    = jnp.where(is_robot, vis_replaced, tokens["vis"])
    masked["action"] = jnp.where(is_robot, tokens["action"], act_replaced)

    ones = jnp.ones((B, T), dtype=jnp.float32)
    zeros = jnp.zeros((B, T), dtype=jnp.float32)
    # action loss = 1 for robot rows, 0 for human rows.
    action_loss = jnp.where(source_mask[:, None], ones, zeros)
    loss_masks = {
        "vis":     zeros,        # never supervised (§3.5)
        "state":   ones,         # bridge — both sources supervise
        "box":     ones,
        "phase":   ones,
        "contact": ones,
        "action":  action_loss,  # robot only
    }
    return masked, loss_masks
