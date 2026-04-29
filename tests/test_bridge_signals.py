"""Tests for the §1.1 bridge-signal derivation helpers.

These functions are pure-numpy and shared between robot and human
preprocessing — the §3.1 invariant ('bridge slots are computed identically
in both pipelines') depends on both pipelines calling exactly these.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cotrain.data.pipelines.bridge import (
    Phase,
    derive_contact_lift,
    derive_phase,
    pack_box_state_camera_frame,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    resample_to_rate,
    world_to_camera_pose,
)


# --- contact_lift --------------------------------------------------------

def test_contact_lift_shape_and_dtype() -> None:
    T = 8
    out = derive_contact_lift(
        left_hand_contact_force=np.zeros(T),
        right_hand_contact_force=np.zeros(T),
        box_height=np.full(T, 0.20),
        initial_box_height=0.20,
    )
    assert out.shape == (T, 3) and out.dtype == np.float32


def test_contact_lift_force_threshold() -> None:
    out = derive_contact_lift(
        left_hand_contact_force=np.array([0.5, 1.5, 5.0]),
        right_hand_contact_force=np.array([0.0, 0.0, 2.0]),
        box_height=np.full(3, 0.20),
        initial_box_height=0.20,
        contact_force_threshold=1.0,
    )
    np.testing.assert_array_equal(out[:, 0], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(out[:, 1], [0.0, 0.0, 1.0])


def test_contact_lift_height_threshold() -> None:
    out = derive_contact_lift(
        left_hand_contact_force=np.zeros(4),
        right_hand_contact_force=np.zeros(4),
        box_height=np.array([0.20, 0.24, 0.26, 0.30]),
        initial_box_height=0.20,
        lift_height=0.05,
    )
    # only frame where box_height > 0.20 + 0.05 = 0.25
    np.testing.assert_array_equal(out[:, 2], [0.0, 0.0, 1.0, 1.0])


# --- phase --------------------------------------------------------------

def test_derive_phase_full_trajectory() -> None:
    T = 100
    # 0..29 APPROACH, 30..49 REACH, 50..59 CONTACT, 60..69 LIFT, 70..99 HOLD.
    hand_dist = np.full(T, 1.0)            # far by default
    hand_dist[30:50] = 0.20                # reach distance
    hand_dist[50:] = 0.05                  # contact distance
    contact_lift = np.zeros((T, 3), dtype=np.float32)
    contact_lift[50:, 0] = 1.0             # left contact 50..end
    contact_lift[50:, 1] = 1.0             # right contact 50..end
    contact_lift[60:, 2] = 1.0             # lifted 60..end

    phase = derive_phase(
        hand_box_distance=hand_dist,
        contact_lift=contact_lift,
        reach_distance=0.30,
        hold_frames=10,                    # HOLD after 10 lifted frames
    ).squeeze(-1)

    np.testing.assert_array_equal(phase[:30], Phase.APPROACH.value)
    np.testing.assert_array_equal(phase[30:50], Phase.REACH.value)
    np.testing.assert_array_equal(phase[50:60], Phase.CONTACT.value)
    # Lift starts at frame 60 (consec_lift=1). HOLD fires when consec_lift
    # reaches hold_frames=10, which is at frame 69. So 60..68 are LIFT,
    # 69 onward is HOLD.
    np.testing.assert_array_equal(phase[60:69], Phase.LIFT.value)
    np.testing.assert_array_equal(phase[69:], Phase.HOLD.value)


def test_derive_phase_consecutive_lift_resets() -> None:
    """Brief un-lift in the middle should reset the HOLD counter."""
    T = 30
    hand_dist = np.full(T, 0.05)
    contact_lift = np.zeros((T, 3), dtype=np.float32)
    contact_lift[:, 0] = 1.0
    contact_lift[:, 1] = 1.0
    # Lift at frames 0..9, drop at 10..14, lift again at 15..end.
    contact_lift[:10, 2] = 1.0
    contact_lift[15:, 2] = 1.0

    phase = derive_phase(
        hand_box_distance=hand_dist,
        contact_lift=contact_lift,
        hold_frames=10,
    ).squeeze(-1)

    # Frames 0..8 are LIFT (consec 1..9), frame 9 hits HOLD (consec=10).
    assert phase[8] == Phase.LIFT.value
    assert phase[9] == Phase.HOLD.value
    # Frames 10..14 drop to CONTACT (no lift).
    np.testing.assert_array_equal(phase[10:15], Phase.CONTACT.value)
    # Re-lift at 15: counter resets so 15..23 are LIFT, 24 reaches HOLD.
    assert phase[23] == Phase.LIFT.value
    assert phase[24] == Phase.HOLD.value


def test_derive_phase_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="contact_lift shape"):
        derive_phase(
            hand_box_distance=np.zeros(5),
            contact_lift=np.zeros((4, 3)),  # mismatched T
        )


# --- world→camera ------------------------------------------------------

def test_world_to_camera_identity_camera_returns_world() -> None:
    """Camera at world origin with identity rotation: object pose unchanged."""
    T = 4
    obj_pos = np.array([[1.0, 2.0, 3.0]] * T)
    obj_quat = np.array([[0.0, 0.0, 0.0, 1.0]] * T)  # xyzw identity
    cam_pos = np.zeros((T, 3))
    cam_quat = np.array([[0.0, 0.0, 0.0, 1.0]] * T)
    pos_c, quat_c = world_to_camera_pose(obj_pos, obj_quat, cam_pos, cam_quat)
    np.testing.assert_allclose(pos_c, obj_pos, atol=1e-6)
    np.testing.assert_allclose(quat_c, obj_quat, atol=1e-6)


def test_world_to_camera_rotated_camera() -> None:
    """Object at (1, 0, 0); camera rotated 90° around z. Object should sit
    at (0, -1, 0) in the camera frame."""
    T = 1
    obj_pos = np.array([[1.0, 0.0, 0.0]])
    obj_quat = np.array([[0.0, 0.0, 0.0, 1.0]])
    cam_pos = np.zeros((T, 3))
    # 90° around z (ccw): xyzw = (0, 0, sin(45°), cos(45°))
    s = np.sin(np.pi / 4)
    cam_quat = np.array([[0.0, 0.0, s, s]])
    pos_c, _ = world_to_camera_pose(obj_pos, obj_quat, cam_pos, cam_quat)
    np.testing.assert_allclose(pos_c[0], [0.0, -1.0, 0.0], atol=1e-6)


def test_world_to_camera_translates() -> None:
    """Camera at (5, 0, 0) identity rotation; object at (5, 1, 0) world ⇒ (0, 1, 0) camera."""
    obj_pos = np.array([[5.0, 1.0, 0.0]])
    obj_quat = np.array([[0.0, 0.0, 0.0, 1.0]])
    cam_pos = np.array([[5.0, 0.0, 0.0]])
    cam_quat = np.array([[0.0, 0.0, 0.0, 1.0]])
    pos_c, _ = world_to_camera_pose(obj_pos, obj_quat, cam_pos, cam_quat)
    np.testing.assert_allclose(pos_c[0], [0.0, 1.0, 0.0], atol=1e-6)


def test_world_to_camera_round_trip() -> None:
    """Random poses: (world → camera → world) returns the original."""
    rng = np.random.default_rng(0)
    T = 8
    obj_pos = rng.normal(size=(T, 3))
    obj_quat_xyzw = Rotation.random(num=T, random_state=0).as_quat()
    cam_pos = rng.normal(size=(T, 3))
    cam_quat_xyzw = Rotation.random(num=T, random_state=1).as_quat()

    pos_c, quat_c = world_to_camera_pose(obj_pos, obj_quat_xyzw, cam_pos, cam_quat_xyzw)
    # Inverse transform: cam → world. Compose: cam_rot * pos_c + cam_pos.
    cam_rot = Rotation.from_quat(cam_quat_xyzw)
    obj_pos_w_back = cam_rot.apply(pos_c) + cam_pos
    obj_rot_back = cam_rot * Rotation.from_quat(quat_c)
    np.testing.assert_allclose(obj_pos_w_back, obj_pos, atol=1e-5)
    # Quaternion equality up to sign:
    np.testing.assert_allclose(
        np.minimum(np.linalg.norm(obj_rot_back.as_quat() - obj_quat_xyzw, axis=-1),
                    np.linalg.norm(obj_rot_back.as_quat() + obj_quat_xyzw, axis=-1)),
        0.0, atol=1e-5,
    )


# --- quaternion convention swap -----------------------------------------

def test_quat_convention_swap_round_trip() -> None:
    rng = np.random.default_rng(0)
    q_xyzw = rng.normal(size=(5, 4))
    q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
    np.testing.assert_array_equal(q_wxyz[:, 0], q_xyzw[:, 3])
    np.testing.assert_array_equal(q_wxyz[:, 1:], q_xyzw[:, :3])
    np.testing.assert_array_equal(quat_wxyz_to_xyzw(q_wxyz), q_xyzw)


def test_pack_box_state_layout() -> None:
    pos = np.array([[1.0, 2.0, 3.0]])
    quat_xyzw = np.array([[0.0, 0.0, 0.0, 1.0]])  # identity
    packed = pack_box_state_camera_frame(pos, quat_xyzw)
    # §1.1 layout: [x, y, z, qw, qx, qy, qz]
    np.testing.assert_array_equal(packed, [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])


# --- resample -----------------------------------------------------------

def test_resample_50_to_30_length() -> None:
    arr = np.arange(50, dtype=np.float32)[:, None]
    out = resample_to_rate(arr, src_hz=50, target_hz=30)
    assert out.shape == (30, 1)


def test_resample_linear_endpoints() -> None:
    """Linear resample preserves first and last source values exactly."""
    arr = np.linspace(0.0, 1.0, 51, dtype=np.float64)[:, None]   # 51 frames @ 50 Hz
    out = resample_to_rate(arr, src_hz=50, target_hz=30)
    assert abs(out[0, 0] - 0.0) < 1e-9
    # Last target time = (T_target - 1) / 30. With T_target = round(51 * 30/50) = 31,
    # last time = 30/30 = 1.0 sec. Source ends at 50/50 = 1.0 sec. So matches.
    assert abs(out[-1, 0] - 1.0) < 1e-9


def test_resample_nearest_phase_preserves_dtype() -> None:
    """Nearest-mode keeps int dtype intact (phase enum)."""
    phase = np.array([[0], [0], [1], [1], [2], [2]] * 5, dtype=np.int8)  # 30 frames
    out = resample_to_rate(phase, src_hz=50, target_hz=30, mode="nearest")
    assert out.dtype == np.int8
    assert out.shape == (18, 1)        # round(30 * 30/50) = 18


def test_resample_no_op_when_rates_match() -> None:
    arr = np.arange(10, dtype=np.float32)[:, None]
    out = resample_to_rate(arr, src_hz=30, target_hz=30)
    np.testing.assert_array_equal(out, arr)


def test_resample_rejects_bad_args() -> None:
    arr = np.zeros((10, 2))
    with pytest.raises(ValueError):
        resample_to_rate(arr, src_hz=-1, target_hz=30)
    with pytest.raises(ValueError):
        resample_to_rate(arr, src_hz=50, target_hz=30, mode="bogus")
