# -*- coding: utf-8 -*-
"""
player.py — playable SPECTATOR hawk (keyboard-piloted, ghost).

Shares the AI pack's exact physics stack (6-DOF body-axis quaternion
attitude, soft wall nose-deflection, ceiling pitch-down, velocity soft
blend, XZ hard clamp, terrain ground floor) but is a pure observer:
it is NEVER fed to predators.update_pack() or rabbits.update_rabbits(),
so starlings and rabbits never see it, it cannot catch birds, and the
30-second generation resets never touch its state.

State lives in module globals — access via `player.player_pos` etc.
Controls are fully remappable in config.toml [player] (<action>_keys =
"key" or ["key", ...]); flight speeds/rates come from [player] too and
fall back to the AI [hawk] values when not set. The default binding:
W/S pitch (W = nose down, matching the AI body_r sign), A/D roll,
Q/E yaw (+ gentle auto-coordination from bank), Shift/Ctrl throttle,
R respawn (wired in hud_cam.handle_input via PLAYER_KEYS['reset']).
"""
import numpy as np
from ursina import Entity, Vec3, color, time, held_keys
from panda3d.core import Quat as PQuat

import config
import terrain
from math3d import _q_mul_b, _q_from_aa_b, _q_rot_b


def _held(action):
    """1.0 if ANY key bound to this action is held (config.PLAYER_KEYS)."""
    return 1.0 if any(held_keys[k] for k in config.PLAYER_KEYS[action]) else 0.0

# --- player state (single hawk; arrays shaped (1,N) feed the batched helpers) ---
player_pos = np.array([0.0, config.ROOST_BASE_ALTITUDE + 10.0, -80.0], dtype=np.float32)
player_vel = np.zeros(3, dtype=np.float32)
player_speed = float(config.PLAYER_CRUISE_SPEED)
# unit quaternion [x,y,z,w] — identity faces +Z (toward the arena center from
# the spawn point at z=-80), same convention as pack_q rows.
player_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
player_entity = None


def player_axes():
    """local (forward, right, up) unit vectors of the player: each shape (3,)."""
    f = _q_rot_b(player_q[None, :], np.array([[0.0, 0.0, 1.0]]))[0]
    r = _q_rot_b(player_q[None, :], np.array([[1.0, 0.0, 0.0]]))[0]
    u = _q_rot_b(player_q[None, :], np.array([[0.0, 1.0, 0.0]]))[0]
    return f, r, u


def reset_player():
    """Spawn/reset the player: above terrain at (0,-80), facing +Z, cruise."""
    global player_speed
    player_pos[:] = [0.0, float(terrain.height_at(0.0, -80.0)) + 25.0, -80.0]
    player_vel[:] = 0.0
    player_speed = float(config.PLAYER_CRUISE_SPEED)
    player_q[:] = [0.0, 0.0, 0.0, 1.0]


def init_player_visuals():
    """Spawn the player hawk entity — call only AFTER app = Ursina(...) exists.
    Cyan/sky-blue tint so it reads as distinct from the AI pack's reds/golds."""
    global player_entity
    player_entity = Entity(model='assets/hawk.glb', color=color.rgb(80, 220, 255))
    reset_player()


def update_player():
    """One frame of player flight: keys -> 6-DOF attitude -> walls -> move."""
    global player_speed, player_q, player_pos, player_vel
    # dt cap: same policy as update_predators — a heavy-frame time.dt spike
    # must never fire a giant rate rotation.
    adt = min(max(time.dt, 0.0), 0.05)

    # --- controls (held_keys) ---
    # pitch: W = +1 = nose down (matches the AI's +about-body-r convention,
    # verified numerically: +angle about body r pitches f downward).
    pitch_in = _held('pitch_down') - _held('pitch_up')
    # roll: A = +1 banks LEFT (+angle about body f lifts the right wing —
    # verified against entity.right/up), D banks right.
    roll_in = _held('roll_left') - _held('roll_right')
    # yaw: explicit Q/E (+about body u = right), plus a gentle auto-
    # coordinated turn when banking (bank left -> yaw left) so keyboard
    # flying feels natural without fighting the roll/yaw coupling.
    yaw_in = (_held('yaw_right') - _held('yaw_left')) + (-config.PLAYER_AUTO_COORD * roll_in)
    # throttle: Shift boost / nothing cruise / Ctrl slow
    thr = _held('boost') - _held('slow')

    # --- 6-DOF attitude: body-axis deltas RIGHT-multiply (q_new = q * dq),
    # identical to the pack's convention (see predators.py header). ---
    q1 = player_q[None, :]                      # (1,4) view of the (4,) state
    body_u = np.array([[0.0, 1.0, 0.0]])
    body_r = np.array([[1.0, 0.0, 0.0]])
    body_f = np.array([[0.0, 0.0, 1.0]])
    dq = _q_from_aa_b(body_u, np.array([yaw_in * config.PLAYER_YAW_RATE * adt]))
    dq = _q_mul_b(dq, _q_from_aa_b(body_r, np.array([pitch_in * config.PLAYER_PITCH_RATE * adt])))
    dq = _q_mul_b(dq, _q_from_aa_b(body_f, np.array([roll_in * config.PLAYER_ROLL_RATE * adt])))
    player_q[:] = _q_mul_b(q1, dq)[0]
    player_q /= max(float(np.linalg.norm(player_q)), 1e-9)

    # --- boundary attitude deflection: EXACTLY the pack's minimum-rotation
    # steer construction (single hawk shaped (1,3)); RATE deg/s * adt. ---
    f, r, u = player_axes()
    f1 = f[None, :]
    half = config.TERRAIN_SIZE / 2 - 1.0
    soft = 25.0
    for ax in (0, 2):
        out_sign = 1.0 if player_pos[ax] > 0 else -1.0
        edge_in = half - abs(player_pos[ax])
        depth = float(np.clip((soft - edge_in) / soft, 0.0, 1.0))
        outward = max(float(f[ax]) * out_sign, 0.0)    # >0 = nose aims at wall
        inward_vec = np.zeros((1, 3)); inward_vec[0, ax] = -out_sign
        steer_axis = np.cross(f1, inward_vec)
        # cross() vanishes when the nose points EXACTLY at the wall —
        # fall back to world up (same guard as the pack).
        if np.linalg.norm(steer_axis, axis=1)[0] < 1e-6:
            steer_axis[0] = (0.0, 1.0, 0.0)
        turn = min(outward * depth * 120.0, 60.0) * adt
        player_q[:] = _q_mul_b(_q_from_aa_b(steer_axis, np.array([turn])),
                               player_q[None, :])[0]
    # ceiling: nose pitched UP at/above ceiling -> rotate nose toward straight
    # down via the same minimum-rotation construction (valid at any attitude).
    at_ceiling = player_pos[1] > config.MAX_ALTITUDE - 1.0
    down_vec = np.zeros((1, 3)); down_vec[0, 1] = -1.0
    ceil_axis = np.cross(f1, down_vec)
    if np.linalg.norm(ceil_axis, axis=1)[0] < 1e-6:
        ceil_axis[0] = (1.0, 0.0, 0.0)
    nose_up = float(np.clip(f[1], 0.0, 1.0))
    ceil_turn = (min(np.degrees(np.arcsin(nose_up)) * 2.0, 60.0)
                 if at_ceiling else 0.0) * adt
    player_q[:] = _q_mul_b(_q_from_aa_b(ceil_axis, np.array([ceil_turn])),
                           player_q[None, :])[0]
    player_q /= max(float(np.linalg.norm(player_q)), 1e-9)

    # --- STALL RECOVERY: identical to the pack — below STALL_SPEED with the
    # nose pitched up, lift is gone and gravity drops the nose toward the
    # horizon (minimum-rotation about cross(f, down), degenerate -> +X). ---
    f, r, u = player_axes()
    f1 = f[None, :]
    if player_speed < config.STALL_SPEED and f[1] > 0.0:
        stall_factor = min(max((config.STALL_SPEED - player_speed)
                               / max(config.STALL_SPEED - config.MIN_FLIGHT_SPEED, 1e-6), 0.0), 1.0)
        down_vec = np.zeros((1, 3)); down_vec[0, 1] = -1.0
        stall_axis = np.cross(f1, down_vec)
        if np.linalg.norm(stall_axis, axis=1)[0] < 1e-6:
            stall_axis[0] = (1.0, 0.0, 0.0)
        stall_turn = stall_factor * max(float(f[1]), 0.0) * 75.0 * adt
        player_q[:] = _q_mul_b(_q_from_aa_b(stall_axis, np.array([stall_turn])),
                               player_q[None, :])[0]
        player_q /= max(float(np.linalg.norm(player_q)), 1e-9)
        f, r, u = player_axes()
        f1 = f[None, :]

    # --- throttle: same targets + dual-rate blend as the pack: gentler
    # drag decay (0.8) above the target so dive momentum carries through
    # the swoop, snappier thrust (1.5) below it. ---
    if thr >= 0:
        target = config.PLAYER_CRUISE_SPEED + thr * (config.PLAYER_BOOST_SPEED - config.PLAYER_CRUISE_SPEED)
    else:
        target = config.PLAYER_CRUISE_SPEED * (1.0 + thr * (1.0 - config.PLAYER_SLOW_FACTOR))
    blend_rate = 0.8 if player_speed > target else 1.5
    player_speed += (target - player_speed) * min(adt * blend_rate, 1.0)
    # --- gravity energy exchange (same as the pack): stoop builds speed up
    # to DIVE_MAX_SPEED, climbing bleeds it down toward MIN_FLIGHT_SPEED. ---
    player_speed += (-config.HAWK_GRAVITY * float(f[1])) * adt
    player_speed = float(np.clip(player_speed, config.MIN_FLIGHT_SPEED, config.DIVE_MAX_SPEED))

    # --- velocity always along the nose ---
    f, r, u = player_axes()
    player_vel[:] = f * player_speed

    # --- invisible walls: soft velocity deflection, then hard clamp
    # (same blend as update_predators, single-hawk form) ---
    half = config.TERRAIN_SIZE / 2 - 1.0
    soft = 15.0
    for ax in (0, 2):
        depth = max(-half + soft - player_pos[ax], 0.0)
        inward = 1.0 if depth > 0 else 0.0
        depth2 = max(player_pos[ax] - (half - soft), 0.0)
        if depth2 > 0:
            inward -= 1.0
        depth_in = max(depth, depth2) / soft
        k = min(depth_in * 1.2, 1.0)
        target_v = inward * player_speed
        player_vel[ax] = player_vel[ax] * (1 - k) + target_v * k

    player_pos += player_vel * time.dt * 60
    player_pos[0] = np.clip(player_pos[0], -half, half)
    player_pos[2] = np.clip(player_pos[2], -half, half)
    ground = float(terrain.height_at(player_pos[0], player_pos[2]))
    player_pos[1] = min(config.MAX_ALTITUDE, max(player_pos[1], ground + 0.4))

    if not np.all(np.isfinite(player_pos)):
        # same nan-guard spirit as the pack: snap back to spawn
        print("[nan-guard] player went non-finite; resetting")
        reset_player()

    # --- sync entity ---
    if player_entity is not None:
        player_entity.position = Vec3(*player_pos)
        player_entity.quaternion = PQuat(player_q[3], player_q[0], player_q[1], player_q[2])
