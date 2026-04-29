"""Per-modality input projection heads (PROJECT_PLAN_1.md §2.2 / §3.2).

Each of the six token slots gets its own tiny projection into the
transformer's `d_model` space. Two STATE heads exist (robot vs human) because
proprio and human kinematics live in different feature spaces; both produce
slot-1 tokens so the transformer sees them as the same token type — that's
the modality-aligned scheme of Radosavovic et al. (§2.2).

Mask tokens (`vis_mask`, `action_mask`) are stored as `nnx.Param` so they
live in the state tree and receive optimizer updates (§2.2 last paragraph).

The heads here only *project*. Source-aware swap-in of the mask tokens and
loss-mask construction live in `cotrain.training.masking` (§3.4).
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

DINO_FEATURE_DIM = 768   # DINOv2 ViT-B/14 last-hidden mean-pool
NUM_PHASES = 5           # APPROACH/REACH/CONTACT/LIFT/HOLD
CONTACT_DIM = 3          # [left_contact, right_contact, lifted]
BOX_DIM = 7              # [x, y, z, qw, qx, qy, qz]


def _kernel_init():
    """Xavier with FSDP-ready partitioning on the output axis (§3.2 #1).

    With a 1D ('data',) mesh these annotations are inert; they only become
    active when promoting to a 2D ('data', 'model') mesh, at which point
    sharding the larger Linear/Embed kernels is essential to avoid OOM."""
    return nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, "model"))


def _embed_init():
    """Embedding tables get the same model-axis annotation on the *features* axis."""
    return nnx.with_partitioning(nnx.initializers.normal(stddev=0.02), (None, "model"))


class _MLP(nnx.Module):
    """LayerNorm → Linear(in→hidden) → GELU → Linear(hidden→out).

    The plan's STATE/BOX/ACTION input heads share this exact recipe.
    LayerNorm sits *before* the projection so each modality is fed at a
    consistent scale regardless of upstream normalization choices."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, *, rngs: nnx.Rngs) -> None:
        self.norm = nnx.LayerNorm(d_in, rngs=rngs)
        self.fc1 = nnx.Linear(d_in, d_hidden, rngs=rngs, kernel_init=_kernel_init())
        self.fc2 = nnx.Linear(d_hidden, d_out, rngs=rngs, kernel_init=_kernel_init())

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.norm(x)
        x = nnx.gelu(self.fc1(x))
        return self.fc2(x)


class ProjectionHeads(nnx.Module):
    """One head per slot + two learned mask vectors.

    Inputs to `__call__` are organised as a dict to keep the call site
    explicit; the trainer assembles it from the batch produced by the
    sampler. `state_robot` / `state_human` are *both* present in every batch
    (zero-filled for the inapplicable source) so we can route them with a
    JIT-friendly `jnp.where` instead of Python-level branching (§3.4
    warning: branching on `source == "robot"` silently breaks under nnx.jit).
    """

    def __init__(
        self,
        d_model: int,
        D_p: int,
        D_h: int,
        D_a: int,
        *,
        rngs: nnx.Rngs,
        dino_dim: int = DINO_FEATURE_DIM,
        num_phases: int = NUM_PHASES,
        contact_dim: int = CONTACT_DIM,
        box_dim: int = BOX_DIM,
    ) -> None:
        self.d_model = d_model

        self.vis = nnx.Linear(dino_dim, d_model, rngs=rngs, kernel_init=_kernel_init())
        self.state_robot = _MLP(D_p, 256, d_model, rngs=rngs)
        self.state_human = _MLP(D_h, 256, d_model, rngs=rngs)
        self.box = _MLP(box_dim, 64, d_model, rngs=rngs)
        self.phase = nnx.Embed(num_phases, d_model, rngs=rngs, embedding_init=_embed_init())
        self.contact = nnx.Linear(contact_dim, d_model, rngs=rngs, kernel_init=_kernel_init())
        self.action = _MLP(D_a, 256, d_model, rngs=rngs)

        # Learned mask vectors: separate from slot embeddings, replace projected
        # content entirely (§2.2 final paragraph). Tracked in state tree.
        self.vis_mask = nnx.Param(jnp.zeros((d_model,)))
        self.action_mask = nnx.Param(jnp.zeros((d_model,)))

    def __call__(
        self,
        *,
        vis: jnp.ndarray,           # (B, T, dino_dim) — DINO features (already pooled)
        state_robot: jnp.ndarray,   # (B, T, D_p)      — zeros for human samples
        state_human: jnp.ndarray,   # (B, T, D_h)      — zeros for robot samples
        box: jnp.ndarray,           # (B, T, 7)
        phase: jnp.ndarray,         # (B, T, 1) int8/int32
        contact: jnp.ndarray,       # (B, T, 3)
        action: jnp.ndarray,        # (B, T, D_a)      — zeros for human samples
        source_mask: jnp.ndarray,   # (B,) bool — True = robot
    ) -> dict[str, jnp.ndarray]:
        """Returns one (B, T, d_model) tensor per slot. No mask swap-in here —
        that happens in `training.masking` so this module stays purely about
        feature projection (single responsibility, simpler to test)."""
        s_r = self.state_robot(state_robot)
        s_h = self.state_human(state_human)
        # Per-sample selection. `is_robot` broadcasts over (T, d_model).
        is_robot = source_mask[:, None, None]
        state = jnp.where(is_robot, s_r, s_h)

        # Phase: stored as (B, T, 1) int; squeeze the trailing dim and cast
        # to int32 for the embedding lookup (Embed wants integer indices).
        phase_idx = phase[..., 0].astype(jnp.int32)
        return {
            "vis":     self.vis(vis),
            "state":   state,
            "box":     self.box(box),
            "phase":   self.phase(phase_idx),
            "contact": self.contact(contact),
            "action":  self.action(action),
        }
