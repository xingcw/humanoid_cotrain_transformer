"""Bridge-signal derivation, world→camera transform, and 50→30 Hz resampling.

Pure-numpy helpers used by both `cotrain.scripts.preprocess_robot.py` (the
holosoma rollout → §1.1 HDF5 bridge) and `cotrain.scripts.preprocess_human
.py` (Aria capture → §1.1 HDF5). Lives outside the JAX path so it can run
in any conda env (holosoma's `hssim` for robot, the Aria env for human)
and produces NumPy arrays ready to be written to HDF5.

The §3.1 invariant — `box_state`, `phase`, and `contact_lift` are computed
identically in both pipelines — is enforced by both pipelines calling the
same functions in this module. If you're tempted to fork a copy in the
robot or human script, don't: that's exactly the bug the plan flags as
"the bridge collapses and co-training fails per Lei et al.'s disjoint
regime."

Quaternion conventions
----------------------
**§1.1 HDF5 stores box_state as wxyz** (`[x, y, z, qw, qx, qy, qz]`).
**holosoma's MotionCommand exposes xyzw** (`simulator_object_quat_w` is
already converted from raw motion-data wxyz to xyzw at load time).
**scipy.spatial.transform.Rotation uses xyzw**. We standardise on xyzw
internally and convert to wxyz only when packing the final array for
HDF5; helpers below accept xyzw and return xyzw unless explicitly named
otherwise.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np
from scipy.spatial.transform import Rotation


# --- phase / contact_lift derivation -------------------------------------

class Phase(IntEnum):
    APPROACH = 0
    REACH = 1
    CONTACT = 2
    LIFT = 3
    HOLD = 4


def derive_contact_lift(
    *,
    left_hand_contact_force: np.ndarray,
    right_hand_contact_force: np.ndarray,
    box_height: np.ndarray,
    initial_box_height: float,
    contact_force_threshold: float = 1.0,
    lift_height: float = 0.05,
) -> np.ndarray:
    """Per-step `[left_contact, right_contact, lifted]` ∈ {0, 1}.

    Args:
        left_hand_contact_force: (T,) magnitudes of contact-force vector on
            the left hand body (Newtons). For the human side: synthesise
            from hand-keypoint proximity to the box.
        right_hand_contact_force: same, right hand.
        box_height: (T,) world-frame z-coordinate of the box centroid.
        initial_box_height: scalar. The starting (resting) box height; the
            `lifted` flag fires when box_height exceeds this by `lift_height`.
        contact_force_threshold: contact fires above this magnitude.
        lift_height: vertical clearance (m) above `initial_box_height` at
            which `lifted` flips to 1.

    Returns: (T, 3) float32 with values in {0.0, 1.0}.
    """
    T = box_height.shape[0]
    if left_hand_contact_force.shape != (T,) or right_hand_contact_force.shape != (T,):
        raise ValueError(
            f"contact-force shapes mismatch: left={left_hand_contact_force.shape} "
            f"right={right_hand_contact_force.shape} box_height={box_height.shape}"
        )
    out = np.zeros((T, 3), dtype=np.float32)
    out[:, 0] = (left_hand_contact_force > contact_force_threshold).astype(np.float32)
    out[:, 1] = (right_hand_contact_force > contact_force_threshold).astype(np.float32)
    out[:, 2] = (box_height > initial_box_height + lift_height).astype(np.float32)
    return out


def derive_phase(
    *,
    hand_box_distance: np.ndarray,
    contact_lift: np.ndarray,
    reach_distance: float = 0.30,
    hold_frames: int = 30,
) -> np.ndarray:
    """Rule-based 5-phase trajectory.

    The §1.1 phase enum:
        0 APPROACH — neither hand near the box, no contact
        1 REACH    — at least one hand within `reach_distance`, no contact
        2 CONTACT  — at least one hand contacting, box still on ground
        3 LIFT     — contacting and box above `lift_height`
        4 HOLD     — sustained LIFT for ≥ `hold_frames` consecutive frames

    Args:
        hand_box_distance: (T,) min(left, right) hand-to-box centre distance.
        contact_lift: (T, 3) — output of `derive_contact_lift`.
        reach_distance: distance (m) below which we count REACH.
        hold_frames: consecutive lifted frames after which phase becomes HOLD.
            Default 30 = 1 second at 30 Hz.

    Returns: (T, 1) int8 array of phase ids.
    """
    T = hand_box_distance.shape[0]
    if contact_lift.shape != (T, 3):
        raise ValueError(f"contact_lift shape {contact_lift.shape} != ({T}, 3)")
    has_contact = (contact_lift[:, 0] > 0.5) | (contact_lift[:, 1] > 0.5)
    is_lifted = contact_lift[:, 2] > 0.5

    # Run-length count of consecutive `is_lifted=True` frames ending at t.
    consec_lift = np.zeros(T, dtype=np.int32)
    cnt = 0
    for t in range(T):
        cnt = cnt + 1 if is_lifted[t] else 0
        consec_lift[t] = cnt

    phase = np.empty((T, 1), dtype=np.int8)
    for t in range(T):
        if has_contact[t]:
            if is_lifted[t]:
                phase[t, 0] = (
                    Phase.HOLD.value if consec_lift[t] >= hold_frames
                    else Phase.LIFT.value
                )
            else:
                phase[t, 0] = Phase.CONTACT.value
        else:
            phase[t, 0] = (
                Phase.REACH.value if hand_box_distance[t] < reach_distance
                else Phase.APPROACH.value
            )
    return phase


# --- world→camera transform ---------------------------------------------

def world_to_camera_pose(
    obj_pos_w: np.ndarray,
    obj_quat_w_xyzw: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform an object pose from world frame to camera frame.

    All inputs are world-frame; outputs are in the camera's frame.

    Args:
        obj_pos_w: (T, 3) object positions in world.
        obj_quat_w_xyzw: (T, 4) object orientations as xyzw quaternions.
        cam_pos_w: (T, 3) camera positions in world.
        cam_quat_w_xyzw: (T, 4) camera orientations as xyzw quaternions.

    Returns: (pos_c (T, 3), quat_c_xyzw (T, 4)) both float32.
    """
    if obj_pos_w.shape != cam_pos_w.shape:
        raise ValueError(f"obj/cam pos shape mismatch: {obj_pos_w.shape} vs {cam_pos_w.shape}")
    if obj_quat_w_xyzw.shape != cam_quat_w_xyzw.shape:
        raise ValueError(
            f"obj/cam quat shape mismatch: {obj_quat_w_xyzw.shape} vs {cam_quat_w_xyzw.shape}"
        )

    cam_rot = Rotation.from_quat(cam_quat_w_xyzw)
    obj_rot = Rotation.from_quat(obj_quat_w_xyzw)
    cam_inv = cam_rot.inv()

    pos_c = cam_inv.apply(obj_pos_w - cam_pos_w)
    obj_rot_c = cam_inv * obj_rot
    quat_c = obj_rot_c.as_quat()                    # xyzw

    return pos_c.astype(np.float32), quat_c.astype(np.float32)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """xyzw → wxyz convention swap. Last axis must be 4."""
    if q.shape[-1] != 4:
        raise ValueError(f"expected last dim 4 for quaternion, got {q.shape}")
    return q[..., [3, 0, 1, 2]].copy()


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """wxyz → xyzw convention swap. Last axis must be 4."""
    if q.shape[-1] != 4:
        raise ValueError(f"expected last dim 4 for quaternion, got {q.shape}")
    return q[..., [1, 2, 3, 0]].copy()


def pack_box_state_camera_frame(
    pos_c: np.ndarray,
    quat_c_xyzw: np.ndarray,
) -> np.ndarray:
    """Pack camera-frame pose into the §1.1 (T, 7) box_state layout
    `[x, y, z, qw, qx, qy, qz]` (wxyz, the convention the HDF5 uses)."""
    quat_wxyz = quat_xyzw_to_wxyz(quat_c_xyzw)
    return np.concatenate([pos_c, quat_wxyz], axis=-1).astype(np.float32)


# --- resampling ----------------------------------------------------------

def resample_to_rate(
    arr: np.ndarray,
    *,
    src_hz: float,
    target_hz: float,
    mode: str = "linear",
) -> np.ndarray:
    """Resample `arr` along axis 0 from `src_hz` to `target_hz`.

    `mode='linear'` is per-feature linear interpolation — use for
    continuous fields (positions, velocities, joint angles).
    `mode='nearest'` picks the nearest source frame — use for discrete
    fields (phase enum, integer flags).

    Output length is `round(arr.shape[0] * target_hz / src_hz)`. With
    src=50 and target=30 this is `round(0.6 * T_src)`."""
    if arr.shape[0] < 1:
        raise ValueError("source has no timesteps")
    if src_hz <= 0 or target_hz <= 0:
        raise ValueError(f"non-positive rate: src={src_hz}, target={target_hz}")
    if mode not in ("linear", "nearest"):
        raise ValueError(f"mode must be 'linear' or 'nearest'; got {mode!r}")

    T_src = arr.shape[0]
    T_target = max(1, int(round(T_src * target_hz / src_hz)))
    if T_target == T_src:
        return arr.copy()

    src_times = np.arange(T_src, dtype=np.float64) / src_hz
    tgt_times = np.arange(T_target, dtype=np.float64) / target_hz

    if mode == "nearest":
        # For each target time, pick the closest source frame.
        idx = np.clip(np.round(tgt_times * src_hz).astype(np.int64), 0, T_src - 1)
        return arr[idx]

    # Linear: per-feature 1D interpolation. Reshape trailing dims to a flat
    # axis so we can call np.interp once per output column.
    flat = arr.reshape(T_src, -1).astype(np.float64)
    out_flat = np.empty((T_target, flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        out_flat[:, j] = np.interp(tgt_times, src_times, flat[:, j])
    out = out_flat.reshape((T_target,) + arr.shape[1:])
    return out.astype(arr.dtype)
