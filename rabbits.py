# -*- coding: utf-8 -*-
"""
rabbits.py — ground prey layer (layer 7: imports backend/config/terrain/predators).

--- 3d. GROUND PREY: RABBITS (pure NumPy 4-state machine) ---
Forage(0) / Vigilance(1) / Freeze(2) / Flee(3). They react to the hawk
pack AND the player hawk but do not affect hawk fitness yet. Burrows are
INVISIBLE: physics only, nothing rendered.
"""
import numpy as np
from ursina import Entity, Mesh, color, time

from config import (NUM_BURROWS, NUM_RABBITS, RABBIT_FLEE_RADIUS,
                    RABBIT_FREEZE_RADIUS, RABBIT_RECOVER_RADIUS,
                    RABBIT_FORAGE_SPEED, RABBIT_FLEE_SPEED,
                    RABBIT_ESCAPE_RADIUS, TERRAIN_SIZE,
                    FOX_STEALTH_FLEE_RADIUS,
                    FOX_STEALTH_FREEZE_RADIUS, FOX_SPRINT_FLEE_RADIUS,
                    FOX_CATCH_RADIUS, FOX_BOOST_SPEED, FOX_CRUISE_SPEED,
                    FOX_STALK_FACTOR)
import predators
import player
import foxes
import terrain

_rabbit_rng = np.random.default_rng()
_rab_edge = TERRAIN_SIZE / 2 - 10.0
# invisible burrow field: (X, Z) pairs inside the playable terrain
burrow_xz = _rabbit_rng.uniform(-_rab_edge, _rab_edge, size=(NUM_BURROWS, 2)).astype(np.float32)

rabbit_pos = np.zeros((NUM_RABBITS, 3), dtype=np.float32)
rabbit_vel = np.zeros((NUM_RABBITS, 3), dtype=np.float32)
rabbit_state = np.zeros(NUM_RABBITS, dtype=np.int32)   # 0 = Forage
rabbit_timer = np.zeros(NUM_RABBITS, dtype=np.float32)
_sx = _rabbit_rng.uniform(-_rab_edge, _rab_edge, NUM_RABBITS)
_sz = _rabbit_rng.uniform(-_rab_edge, _rab_edge, NUM_RABBITS)
rabbit_pos[:, 0] = _sx
rabbit_pos[:, 2] = _sz
rabbit_pos[:, 1] = terrain.height_at(_sx, _sz)

def update_rabbits():
    """One frame for the whole rabbit population: threat triggers -> state
    machine -> vectorized movement -> integration. Pure NumPy."""
    global rabbit_pos, rabbit_vel, rabbit_state, rabbit_timer
    # same dt-cap policy as the rest of the sim (no giant teleport on a hitch)
    dt = min(max(time.dt, 0.0), 0.05)

    # distance from each rabbit to the nearest threat: AI pack hawks AND
    # the player hawk (freeze/flee/recover thresholds both react to it)
    d_pack = np.linalg.norm(rabbit_pos[:, None, :] - predators.pack_pos[None, :, :], axis=2).min(axis=1)
    d_player = np.linalg.norm(rabbit_pos - player.player_pos[None, :], axis=1)

    # --- fox threats: speed-modulated stealth. A creeping fox (min stalk
    # speed) is only heard at FOX_STEALTH_FLEE_RADIUS and freezes at
    # FOX_STEALTH_FREEZE_RADIUS; a sprinting one at the rabbit's own hawk
    # thresholds (FOX_SPRINT_FLEE_RADIUS / RABBIT_FREEZE_RADIUS). The raw
    # distance is rescaled per fox so those two anchor radii map onto
    # RABBIT_FLEE_RADIUS / RABBIT_FREEZE_RADIUS — at full sprint the map is
    # the identity (matching the task's simple formula), and the existing
    # state-machine thresholds then work unchanged. ---
    d_fox_raw = np.linalg.norm(rabbit_pos[:, None, :] - foxes.fox_pos[None, :, :], axis=2)  # (R,F)
    stalk_min_spd = FOX_CRUISE_SPEED * FOX_STALK_FACTOR
    noise = np.clip((foxes.fox_speed - stalk_min_spd)
                    / max(FOX_BOOST_SPEED - stalk_min_spd, 1e-6), 0.0, 1.0)
    eff_flee_r = FOX_STEALTH_FLEE_RADIUS + noise * (FOX_SPRINT_FLEE_RADIUS - FOX_STEALTH_FLEE_RADIUS)
    eff_freeze_r = FOX_STEALTH_FREEZE_RADIUS + noise * (RABBIT_FREEZE_RADIUS - FOX_STEALTH_FREEZE_RADIUS)
    inside = d_fox_raw <= eff_flee_r[None, :]
    beyond_freeze = d_fox_raw > eff_freeze_r[None, :]
    d_fox_perceived = np.where(
        inside,
        d_fox_raw * (RABBIT_FLEE_RADIUS / eff_flee_r[None, :]),
        np.where(
            beyond_freeze,
            RABBIT_FREEZE_RADIUS + (d_fox_raw - eff_freeze_r[None, :]),
            RABBIT_FLEE_RADIUS + (d_fox_raw - eff_flee_r[None, :])
            * ((RABBIT_FREEZE_RADIUS - RABBIT_FLEE_RADIUS)
               / np.maximum(eff_freeze_r[None, :] - eff_flee_r[None, :], 1e-6))))
    d_fox = d_fox_perceived.min(axis=1)
    d = np.minimum(np.minimum(d_pack, d_player), d_fox)

    # --- fox catches: any rabbit a fox literally touches is eaten; the
    # rabbit respawns Frozen at a random burrow (like the burrow escape),
    # the catching fox's pair is credited in foxes.register_fox_catch. ---
    caught_fox_idx = d_fox_raw.argmin(axis=1)
    caught = d_fox_raw.min(axis=1) < FOX_CATCH_RADIUS
    if caught.any():
        ci = np.where(caught)[0]
        foxes.register_fox_catch(caught_fox_idx[ci])
        new_b = burrow_xz[_rabbit_rng.integers(0, NUM_BURROWS, size=len(ci))]
        rabbit_pos[ci, 0] = new_b[:, 0]
        rabbit_pos[ci, 2] = new_b[:, 1]
        rabbit_pos[ci, 1] = terrain.height_at(new_b[:, 0], new_b[:, 1])
        rabbit_vel[ci] = 0.0
        rabbit_state[ci] = 2         # Frozen at the burrow mouth
        rabbit_timer[ci] = 2.0

    # --- threat triggers ---
    rabbit_state[d < RABBIT_FLEE_RADIUS] = 3
    freeze_m = (d >= RABBIT_FLEE_RADIUS) & (d <= RABBIT_FREEZE_RADIUS) & (rabbit_state != 3)
    rabbit_state[freeze_m] = 2

    # --- timers ---
    rabbit_timer -= dt

    # --- state transitions ---
    m = (rabbit_state == 3) & (d > RABBIT_RECOVER_RADIUS)
    rabbit_state[m] = 1                      # hawk lost: Vigilance, 2s watch
    rabbit_timer[m] = 2.0
    m = ((rabbit_state == 1) | (rabbit_state == 2)) & (d > RABBIT_FREEZE_RADIUS) & (rabbit_timer < 0)
    rabbit_state[m] = 0                      # safe & timer up: back to Forage
    rabbit_timer[m] = _rabbit_rng.uniform(3.0, 6.0, size=int(m.sum()))
    m = (rabbit_state == 0) & (rabbit_timer < 0)
    rabbit_state[m] = 1                      # periodic Forage -> Vigilance check
    rabbit_timer[m] = 2.0

    # --- vectorized movement per state ---
    # Forage: slow random walk (small random 2D kick, normalize, 0.8 m/s)
    fi = np.where(rabbit_state == 0)[0]
    if len(fi):
        v = rabbit_vel[fi].copy()
        v[:, 0] += _rabbit_rng.normal(0.0, 0.5, size=len(fi))
        v[:, 2] += _rabbit_rng.normal(0.0, 0.5, size=len(fi))
        v[:, 1] = 0.0
        n = np.linalg.norm(v, axis=1, keepdims=True)
        rabbit_vel[fi] = v / np.maximum(n, 1e-6) * RABBIT_FORAGE_SPEED
    # Vigilance & Freeze: dead stop
    rabbit_vel[(rabbit_state == 1) | (rabbit_state == 2)] = 0.0
    # Flee: sprint to the nearest burrow at 6 m/s
    li = np.where(rabbit_state == 3)[0]
    if len(li):
        pos_xz = rabbit_pos[li][:, [0, 2]]                          # (fleeing, 2)
        rel = burrow_xz[None, :, :] - pos_xz[:, None, :]            # (fleeing, B, 2)
        nb = np.argmin(np.einsum('ijb,ijb->ij', rel, rel), axis=1)
        target = burrow_xz[nb]                                      # (fleeing, 2)
        to_b = target - pos_xz
        n = np.linalg.norm(to_b, axis=1, keepdims=True)
        v = np.zeros((len(li), 3), dtype=np.float32)
        v[:, [0, 2]] = to_b / np.maximum(n, 1e-6) * RABBIT_FLEE_SPEED
        rabbit_vel[li] = v
        # burrow escape: dive underground -> pop up at a random other burrow
        esc = np.linalg.norm(to_b, axis=1) < RABBIT_ESCAPE_RADIUS
        if esc.any():
            ei = li[esc]
            new_b = burrow_xz[_rabbit_rng.integers(0, NUM_BURROWS, size=len(ei))]
            rabbit_pos[ei, 0] = new_b[:, 0]
            rabbit_pos[ei, 2] = new_b[:, 1]
            rabbit_pos[ei, 1] = terrain.height_at(new_b[:, 0], new_b[:, 1])
            rabbit_vel[ei] = 0.0
            rabbit_state[ei] = 2           # frozen in the burrow mouth
            rabbit_timer[ei] = 2.0

    # --- integration (true m/s: plain time.dt, no per-frame-60 scaling) ---
    rabbit_pos[:, [0, 2]] += rabbit_vel[:, [0, 2]] * dt
    rabbit_pos[:, 0] = np.clip(rabbit_pos[:, 0], -_rab_edge, _rab_edge)
    rabbit_pos[:, 2] = np.clip(rabbit_pos[:, 2], -_rab_edge, _rab_edge)
    rabbit_pos[:, 1] = terrain.height_at(rabbit_pos[:, 0], rabbit_pos[:, 2])

# --- 4b. RABBIT VISUALS (batched mesh, mirrors FlockVisuals) ---
class RabbitVisuals(Entity):
    def __init__(self):
        super().__init__()
        self.model = Mesh(vertices=[], triangles=[], colors=[], mode='triangle', static=False)
        self.texture = 'white_cube'
        self.color = color.rgba32(140, 100, 70)   # small brown rabbits (rgba32: rgb() is 0..1-scale here)
        self.double_sided = True
        self.rabbit_shape = np.array([[0, 0, 0.4], [-0.15, 0, -0.15], [0.15, 0, -0.15]], dtype=np.float32)

    def update(self):
        update_rabbits()
        # hop modulation: bounce scales with ground speed -> still rabbits sit flat
        speed = np.linalg.norm(rabbit_vel, axis=1)
        # phase INTEGRATED per rabbit (see rabbit_phase note: epoch-scale
        # time.time() in float32 freezes the sine; and phase continuity
        # survives speed changes). dt-capped like the physics step.
        rabbit_phase[:] += speed * 2.0 * min(max(time.dt, 0.0), 0.05)
        # a rabbit that stops mid-hop must settle onto the ground: snap the
        # frozen phase to the nearest multiple of pi (hop = |sin| = 0 there)
        still = speed < 0.01
        if still.any():
            rabbit_phase[still] = np.round(rabbit_phase[still] / np.pi) * np.pi
        hop = np.abs(np.sin(rabbit_phase)) * 0.3
        yaws = np.arctan2(rabbit_vel[:, 0], rabbit_vel[:, 2])
        c, s = np.cos(yaws), np.sin(yaws)
        verts = np.tile(self.rabbit_shape, (NUM_RABBITS, 1, 1))
        x, z = verts[:, :, 0].copy(), verts[:, :, 2].copy()   # copies: see FlockVisuals yaw note
        verts[:, :, 0] = x * c[:, None] + z * s[:, None]
        verts[:, :, 2] = -x * s[:, None] + z * c[:, None]
        verts += rabbit_pos[:, None, :]
        verts[:, :, 1] += hop[:, None]
        self.model.vertices = verts.reshape(-1, 3).tolist()
        self.model.generate()

# Hop phase accumulator: time.time() is epoch-scale (~1.8e9) and
# time.time() * speed is computed in float32, whose ulp there is 128 —
# the sine argument stays frozen for minutes. Integrate phase per-rabbit
# in float32 instead (small accumulating values), which also keeps the
# gait continuous when speed changes instead of re-phasing on a jump.
rabbit_phase = np.zeros(NUM_RABBITS, dtype=np.float32)
