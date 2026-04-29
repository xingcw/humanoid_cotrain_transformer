"""Host-side window reader: HDF5 episode → per-slot numpy dict.

This is the pre-device half of §8 step 4. It reads a single contiguous slice
of length T (default 16) from one of our §1.1 episodes and returns a dict
whose keys line up with what the model's projection heads expect. Robot- and
human-only modalities are zero-padded for the *other* source so every sample
in a mixed batch has the same key set — that's what lets the sampler stack
samples without case-splitting downstream.

The reader does no JAX work and no model-side projection. The sequence
assembly (interleave to 6T tokens + slot/time embeddings) lives device-side
in cotrain.models.transformer.sequence; doing it on the host would force
us to materialize d_model-wide tensors before the SPMD shard.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from cotrain.data.schemas import (
    DEFAULT_D_A,
    DEFAULT_D_H,
    DEFAULT_D_P,
    RGB_H,
    RGB_W,
    Source,
    array_dtype,
)


def _read_meta_str(group: h5py.Group, key: str) -> str:
    raw = group[key][()]
    return raw.decode() if isinstance(raw, (bytes, bytearray, np.bytes_)) else str(raw)


def read_window(
    h5_path: Path,
    start: int,
    T: int = 16,
    *,
    D_p: int = DEFAULT_D_P,
    D_h: int = DEFAULT_D_H,
    D_a: int = DEFAULT_D_A,
    pad_other_source: bool = True,
) -> dict[str, np.ndarray]:
    """Read a contiguous window of T timesteps from one episode.

    Returned keys (all (T, ...) leading-axis):
      rgb, box_state, phase, contact_lift, proprio, human_kin, action,
      source (str), source_is_robot (bool array shape ()).

    For robot episodes, `human_kin` is a zero array of shape (T, D_h);
    for human episodes, `proprio` and `action` are zero arrays of the
    appropriate shapes. This is the "pad the other source" behaviour the
    masking module (§3.4) expects, and lets the sampler stack mixed-source
    samples into one batch dict.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        if "meta" not in f or "source" not in f["meta"]:
            raise ValueError(f"{h5_path} missing meta/source")
        source: Source = _read_meta_str(f["meta"], "source")  # type: ignore[assignment]
        if source not in ("robot", "human"):
            raise ValueError(f"unexpected source {source!r} in {h5_path}")

        ep_len = f["phase"].shape[0]
        end = start + T
        if start < 0 or end > ep_len:
            raise IndexError(
                f"window [{start}:{end}] out of bounds for episode length {ep_len} "
                f"in {h5_path}"
            )

        # Bridge slots — present in both sources.
        rgb = f["rgb"][start:end]
        box_state = f["box_state"][start:end]
        phase = f["phase"][start:end]
        contact_lift = f["contact_lift"][start:end]

        if source == "robot":
            proprio = f["proprio"][start:end]
            action = f["action"][start:end]
            human_kin = (
                np.zeros((T, D_h), dtype=array_dtype("human_kin"))
                if pad_other_source
                else None
            )
        else:  # human
            human_kin = f["human_kin"][start:end]
            proprio = (
                np.zeros((T, D_p), dtype=array_dtype("proprio"))
                if pad_other_source
                else None
            )
            action = (
                np.zeros((T, D_a), dtype=array_dtype("action"))
                if pad_other_source
                else None
            )

    out: dict[str, np.ndarray] = {
        "rgb":          rgb,                      # (T, 224, 224, 3) uint8
        "box_state":    box_state,                # (T, 7)
        "phase":        phase,                    # (T, 1) int8
        "contact_lift": contact_lift,             # (T, 3) float32
    }
    if proprio is not None:
        out["proprio"] = proprio                  # (T, D_p)
    if human_kin is not None:
        out["human_kin"] = human_kin              # (T, D_h)
    if action is not None:
        out["action"] = action                    # (T, D_a)
    out["source"] = np.array(source)              # 0-d str
    out["source_is_robot"] = np.array(source == "robot", dtype=np.bool_)
    return out


# Sanity helpers used by tests / the sampler.
EXPECTED_SHAPES = {
    "rgb":          (RGB_H, RGB_W, 3),
    "box_state":    (7,),
    "phase":        (1,),
    "contact_lift": (3,),
}


def assert_window_shapes(window: dict[str, np.ndarray], T: int) -> None:
    """Cheap structural check used by the tokenizer test in §8.4."""
    for name, suffix in EXPECTED_SHAPES.items():
        arr = window[name]
        if arr.shape != (T,) + suffix:
            raise ValueError(
                f"{name} shape {arr.shape} != expected {(T,) + suffix}"
            )
    if "proprio" in window and window["proprio"].shape[0] != T:
        raise ValueError(f"proprio shape {window['proprio'].shape}, expected leading {T}")
    if "human_kin" in window and window["human_kin"].shape[0] != T:
        raise ValueError(f"human_kin shape {window['human_kin'].shape}, expected leading {T}")
    if "action" in window and window["action"].shape[0] != T:
        raise ValueError(f"action shape {window['action'].shape}, expected leading {T}")
