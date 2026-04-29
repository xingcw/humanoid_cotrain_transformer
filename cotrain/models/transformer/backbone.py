"""CoTrainTransformer — the §3.2 shared-bridge backbone.

Wires together the encoder, projection heads, slot/time embeddings,
transformer blocks, final LayerNorm, and the per-slot output heads. The
masking module from §3.4 is *applied here* but the swap-in logic itself
lives in `cotrain.training.masking` so this file stays focused on the
forward pass topology.

For §8 step 5 we skip the visual encoder integration: the trainer can pass
`vis` features directly (already DINO-encoded) so the backbone forward is
pure JAX and easy to smoke-test on CPU. The §8.5 test pins this contract;
the encoder will be wired in by the trainer (§8.10) so the encoder forward
can be cleanly excluded from the optimizer's gradient tree per §3.2.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx

from cotrain.models.heads.output import OutputHeads
from cotrain.models.heads.projection import (
    DINO_FEATURE_DIM,
    ProjectionHeads,
)
from cotrain.models.transformer.blocks import TransformerBlock
from cotrain.models.transformer.sequence import (
    NUM_SLOTS,
    SLOT_ORDER,
    SlotTimeEmbeds,
)


@dataclass(frozen=True)
class TransformerConfig:
    """All knobs for the §3.2 backbone in one place. Defaults match §5.3."""
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    T: int = 16
    D_p: int = 50    # robot proprio
    D_h: int = 157   # human kinematics
    D_a: int = 20    # robot action
    dino_dim: int = DINO_FEATURE_DIM
    ffn_expansion: int = 4
    dropout_rate: float = 0.1


def _split_per_slot(seq: jnp.ndarray, *, num_slots: int = NUM_SLOTS) -> dict[str, jnp.ndarray]:
    """Slice (B, T*S, d_model) into one (B, T, d_model) tensor per slot.

    Inverse of `interleave_slot_tokens` — required by the output heads
    (§2.3) which read from each slot's positions independently. The strided
    indexing `[:, s::S, :]` only works because we use the time-outer /
    slot-inner flattening in `cotrain.models.transformer.sequence`."""
    return {SLOT_ORDER[s]: seq[:, s::num_slots, :] for s in range(num_slots)}


class CoTrainTransformer(nnx.Module):
    """Shared-bridge multimodal transformer.

    Forward signature mirrors §3.2: takes a `batch` dict and a
    `deterministic` flag (False at train time so dropout fires, True at
    eval). The batch must contain projected-modality inputs *plus* a
    `source_mask` (B,) bool flagging robot samples — see
    `cotrain.models.heads.ProjectionHeads.__call__` for the per-key spec.
    """

    def __init__(self, cfg: TransformerConfig, *, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        self.heads = ProjectionHeads(
            d_model=cfg.d_model,
            D_p=cfg.D_p,
            D_h=cfg.D_h,
            D_a=cfg.D_a,
            dino_dim=cfg.dino_dim,
            rngs=rngs,
        )
        self.embeds = SlotTimeEmbeds(T=cfg.T, d_model=cfg.d_model, rngs=rngs)
        self.blocks = [
            TransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                ffn_expansion=cfg.ffn_expansion,
                dropout_rate=cfg.dropout_rate,
                rngs=rngs,
            )
            for _ in range(cfg.n_layers)
        ]
        self.norm = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.out_heads = OutputHeads(
            d_model=cfg.d_model,
            D_p=cfg.D_p,
            D_h=cfg.D_h,
            D_a=cfg.D_a,
            rngs=rngs,
        )

    def __call__(self, batch: dict[str, jnp.ndarray], *, deterministic: bool):
        # Project per-slot inputs and assemble the (B, 6T, d_model) sequence
        # with slot/time embeddings. Modality masking will hook in here in
        # §8.6 by replacing slot-0 (vis, for robot samples) and slot-5
        # (action, for human samples) with the learned mask Params before
        # the transformer blocks fire.
        projected = self.heads(**batch)               # dict slot -> (B, T, D)
        projected = self._apply_modality_masks(projected, batch["source_mask"])
        seq = self.embeds(projected)                  # (B, 6T, D)
        for block in self.blocks:
            seq = block(seq, deterministic=deterministic)
        seq = self.norm(seq)
        per_slot = _split_per_slot(seq, num_slots=NUM_SLOTS)
        return self.out_heads(per_slot)

    def _apply_modality_masks(
        self,
        projected: dict[str, jnp.ndarray],
        source_mask: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """§3.4 mask swap-in. Robot samples drop VIS for `vis_mask`; human
        samples drop ACTION for `action_mask`. Loss masking is computed in
        cotrain.training.masking and applied at loss time -- this function
        only handles the *input-side* token swap.

        Implemented inline as a thin shim until the standalone masking
        module lands in §8.6; both call sites must stay in sync because
        any drift here silently breaks training (§3.4 warning)."""
        is_robot = source_mask[:, None, None]                       # (B, 1, 1)
        vis_mask_token = jnp.broadcast_to(
            self.heads.vis_mask.value, projected["vis"].shape,
        )
        action_mask_token = jnp.broadcast_to(
            self.heads.action_mask.value, projected["action"].shape,
        )
        out = dict(projected)
        out["vis"]    = jnp.where(is_robot, vis_mask_token, projected["vis"])
        out["action"] = jnp.where(is_robot, projected["action"], action_mask_token)
        return out
