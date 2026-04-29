"""Per-episode HDF5 contract — see PROJECT_PLAN_1.md §1.1.

Both robot and human datasets land as one HDF5 file per episode, every
per-timestep array sharing a single 30 Hz timeline. The bridge slots
(`box_state`, `phase`, `contact_lift`) MUST be computed identically in both
pipelines or the shared-bridge mechanism collapses into Lei et al.'s
"disjoint" regime (§3.1).
"""
from __future__ import annotations

from enum import IntEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Image size after preprocess. RGB is uint8 to keep disk cost manageable —
# DINO's own ImageProcessor handles normalization at load time.
RGB_H = 224
RGB_W = 224

SAMPLE_RATE_HZ = 30  # all groups resampled here upstream of this code

# Default modality dimensions. These are *defaults for the synthetic
# generator*; the actual values per embodiment must be recorded in the
# manifest (`box_size_class` is a coarse class label; the dimensions
# themselves are inferred from each HDF5 at load time).
DEFAULT_D_P = 50    # q(20) + qdot(20) + root_quat(4) + root_lin_vel(3) + root_ang_vel(3)
DEFAULT_D_H = 157   # head pose(7) + 2 wrists * 7 + 2 hands * 21 * 3 + upper-body joints(10) - tweak
DEFAULT_D_A = 20    # robot joint position targets

Source = Literal["robot", "human"]


class Phase(IntEnum):
    APPROACH = 0
    REACH = 1
    CONTACT = 2
    LIFT = 3
    HOLD = 4


PHASE_VALUES: frozenset[int] = frozenset(p.value for p in Phase)


class EpisodeMeta(BaseModel):
    """Episode-level metadata; lives in the HDF5 `meta/` group and the manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    source: Source
    success: bool
    n_steps: int = Field(ge=1)
    box_size_class: str
    randomization_seed: int
    recording_date: str  # ISO-8601 YYYY-MM-DD
    hash: str            # md5 of the HDF5 file (validator recomputes & checks)


# --- Per-array shape/dtype contract ---------------------------------------
# Each entry: (shape_per_timestep, dtype, sources_that_must_have_it)
# Validator (§1.3) asserts these exactly. `proprio`/`action` accept variable
# trailing dim because per-embodiment D_p / D_a may differ — the validator
# only requires consistency *within one episode*.
ARRAY_SPEC: dict[str, tuple[tuple[int, ...] | tuple[None, ...], np.dtype, set[Source]]] = {
    "rgb":          ((RGB_H, RGB_W, 3), np.dtype(np.uint8),   {"robot", "human"}),
    "proprio":      ((None,),           np.dtype(np.float32), {"robot"}),
    "human_kin":    ((None,),           np.dtype(np.float32), {"human"}),
    "box_state":    ((7,),              np.dtype(np.float32), {"robot", "human"}),
    "action":       ((None,),           np.dtype(np.float32), {"robot"}),
    "phase":        ((1,),              np.dtype(np.int8),    {"robot", "human"}),
    "contact_lift": ((3,),              np.dtype(np.float32), {"robot", "human"}),
}

# Numeric bounds the validator sanity-checks. Translation in metres, in the
# camera frame; quaternion magnitude near 1; contact_lift binary.
BOX_STATE_TRANS_RANGE = (-2.0, 2.0)        # x,y,z components 0..2
BOX_STATE_QUAT_TOL = 0.05                  # |quat| within 1 +/- this
CONTACT_VALUES = (0.0, 1.0)


def array_dtype(name: str) -> np.dtype:
    return ARRAY_SPEC[name][1]


def array_shape_per_step(name: str) -> tuple[int | None, ...]:
    return ARRAY_SPEC[name][0]


def required_for(source: Source) -> list[str]:
    """Names of arrays that must be present for a given source."""
    return [name for name, (_, _, srcs) in ARRAY_SPEC.items() if source in srcs]
