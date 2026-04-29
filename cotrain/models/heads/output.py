"""Per-modality output (prediction) heads (PROJECT_PLAN_1.md §2.3).

Each head reads from its slot's positions in the (B, 6T, d_model) sequence
and predicts the *next* timestep's same-slot value. VIS prediction is
deliberately skipped (§3.5). All heads use small MLPs to mirror the input
projections in §2.2.
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from cotrain.models.heads.projection import (  # re-export sizes for symmetry
    BOX_DIM,
    CONTACT_DIM,
    NUM_PHASES,
)


def _kernel_init():
    return nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, "model"))


class _MLP(nnx.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, *, rngs: nnx.Rngs) -> None:
        self.fc1 = nnx.Linear(d_in, d_hidden, rngs=rngs, kernel_init=_kernel_init())
        self.fc2 = nnx.Linear(d_hidden, d_out, rngs=rngs, kernel_init=_kernel_init())

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.fc2(nnx.gelu(self.fc1(x)))


class OutputHeads(nnx.Module):
    """One head per predicted slot. The two STATE outputs (robot, human) live
    here for symmetry with the input projection heads — at training time the
    masking module (§3.4) zeros the loss for the inactive source's STATE.
    """

    def __init__(
        self,
        d_model: int,
        D_p: int,
        D_h: int,
        D_a: int,
        *,
        rngs: nnx.Rngs,
        num_phases: int = NUM_PHASES,
        contact_dim: int = CONTACT_DIM,
        box_dim: int = BOX_DIM,
    ) -> None:
        self.state_robot = _MLP(d_model, 256, D_p, rngs=rngs)
        self.state_human = _MLP(d_model, 256, D_h, rngs=rngs)
        self.box = _MLP(d_model, 64, box_dim, rngs=rngs)
        self.phase = nnx.Linear(d_model, num_phases, rngs=rngs, kernel_init=_kernel_init())
        self.contact = nnx.Linear(d_model, contact_dim, rngs=rngs,
                                  kernel_init=_kernel_init())
        self.action = _MLP(d_model, 256, D_a, rngs=rngs)

    def __call__(self, slot_hidden: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        """Each input is (B, T, d_model) — one tensor per slot's hidden state.

        Returns a dict with predicted next-step values per slot.
        - VIS prediction is intentionally absent (§3.5).
        - The two STATE predictions are both produced; the trainer routes
          them via source_mask in the loss computation.
        - Contact head returns logits; sigmoid is applied at loss time.
        """
        return {
            "state_robot": self.state_robot(slot_hidden["state"]),
            "state_human": self.state_human(slot_hidden["state"]),
            "box":         self.box(slot_hidden["box"]),
            "phase":       self.phase(slot_hidden["phase"]),     # logits
            "contact":     self.contact(slot_hidden["contact"]),  # logits
            "action":      self.action(slot_hidden["action"]),
        }
