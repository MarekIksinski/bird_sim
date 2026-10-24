# -*- coding: utf-8 -*-
"""
foxes.py — ground predator pairs (layer 7.5: rabbits call into this module;
rabbits are reached by a LAZY import inside update_foxes, mirroring the
predators<->flock cycle).

10 foxes in 5 PAIRS; each pair shares one small MLP (Moran process on
starvation, mirroring flock.py's tribe architecture: contiguous blocks
fox i -> pair i // FOXES_PER_PAIR, batched einsum over (P, ni, no) numpy
tensors — foxes are pure NumPy like rabbits, no cp). A pair that goes
FOX_STARVE_SECONDS without a catch is replaced by a mutated copy of a
top-scoring pair, so evolution never interrupts an active stalk (no
global generation reset).

Ground physics mirror rabbits.py: TRUE m/s with plain dt (no per-frame-60
dt_scale), 2D heading (yaw) on the terrain surface, edge clamp at _rab_edge.

ALWAYS access this module's state through the module namespace
(foxes.fox_pos, foxes.pair_pop, ...): updates and replacements REBIND names.
"""
import numpy as np
from ursina import Entity, Mesh, color, time

from config import (NUM_FOXES, NUM_PAIRS, FOXES_PER_PAIR, TERRAIN_SIZE,
                    FOX_CRUISE_SPEED, FOX_BOOST_SPEED, FOX_STALK_FACTOR,
                    FOX_YAW_RATE, FOX_TURN_DRAG,
                    FOX_BOOST_DURATION, FOX_BOOST_RECHARGE,
                    FOX_BOOST_COOLDOWN_RECOVER, FOX_CATCH_RADIUS,
                    FOX_STEALTH_FLEE_RADIUS, FOX_STEALTH_FREEZE_RADIUS,
                    FOX_SPRINT_FLEE_RADIUS, FOX_PROXIMITY_RADIUS,
                    FOX_STARVE_SECONDS, FOX_STALK_HUNGER_RELIEF,
                    FOX_MUTATION_SIGMA, FOX_HIDDEN_LAYERS,
                    RABBIT_FLEE_SPEED, COLOR_FOX)
import terrain
from math3d import _q_from_aa_b, _q_mul_b

_fox_rng = np.random.default_rng(4321)
_fox_edge = TERRAIN_SIZE / 2 - 10.0          # same arena line rabbits clamp to

# --- pair MLPs: (12 sensors) + hidden + (2 outputs), one per pair ---
# 0,1 nearest-rabbit dir on body axes f=(sin yaw, cos yaw), r=(cos yaw,-sin yaw)
# 2   rabbit distance / FOX_PROXIMITY_RADIUS
# 3,4 rabbit velocity / RABBIT_FLEE_SPEED on f,r
# 5   rabbit state (-1 forage/vigilant, 0 frozen, +1 fleeing)
# 6,7,8 from the rabbit to ITS nearest burrow (dir f,r + dist/50) — burrow cutoffs
# 9   own speed (-1 stalk .. 0 cruise .. +1 sprint): the fox feels its noise
# 10  stamina sensor (-1.0 exhausted, else -0.8 + 1.8 * stamina)
# 11  arena wall repulsion projected on f (within 80m of the edge)
# Outputs: [0] yaw rate (x FOX_YAW_RATE deg/s), [1] throttle (stamina-gated)
FOX_NET_SHAPE = (12,) + FOX_HIDDEN_LAYERS + (2,)
FOX_GENOME_LEN = sum(FOX_NET_SHAPE[i] * FOX_NET_SHAPE[i+1] +
                     FOX_NET_SHAPE[i+1] for i in range(len(FOX_NET_SHAPE) - 1))
fox_pair_np = np.arange(NUM_FOXES) // FOXES_PER_PAIR

pair_pop = np.random.default_rng(99).normal(
    0.0, 0.5, size=(NUM_PAIRS, FOX_GENOME_LEN)).astype(np.float32)

def _unpack_pair_genome(g):
    """flat float32 vector -> list of (W, b) pairs for FOX_NET_SHAPE."""
    layers, off = [], 0
    for i in range(len(FOX_NET_SHAPE) - 1):
        ni, no = FOX_NET_SHAPE[i], FOX_NET_SHAPE[i + 1]
        size = ni * no + no
        W = g[off:off + ni * no].reshape(ni, no)
        b = g[off + ni * no:off + size]
        layers.append((W, b))
        off += size
    return layers

# batched pair layers: (P, ni, no) / (P, no) — rebuilt from pair_pop by
# _upload_pair_layers() at startup/load and whenever a pair is replaced.
pair_Ws, pair_bs = [], []

def _upload_pair_layers():
    pair_Ws.clear()
    pair_bs.clear()
    Ws = [np.stack([_unpack_pair_genome(pair_pop[p])[li][0]
                    for p in range(NUM_PAIRS)]).astype(np.float32)
          for li in range(len(FOX_NET_SHAPE) - 1)]
    bs = [np.stack([_unpack_pair_genome(pair_pop[p])[li][1]
                    for p in range(NUM_PAIRS)]).astype(np.float32)
          for li in range(len(FOX_NET_SHAPE) - 1)]
    pair_Ws.extend(Ws)
    pair_bs.extend(bs)

_upload_pair_layers()

# --- fox state (pure NumPy, true m/s) ---
fox_pos = np.zeros((NUM_FOXES, 3), dtype=np.float32)
_sx = _fox_rng.uniform(-_fox_edge + 20, _fox_edge - 20, NUM_FOXES)
_sz = _fox_rng.uniform(-_fox_edge + 20, _fox_edge - 20, NUM_FOXES)
fox_pos[:, 0] = _sx
fox_pos[:, 2] = _sz
fox_pos[:, 1] = terrain.height_at(_sx, _sz)
fox_yaw = _fox_rng.uniform(-np.pi, np.pi, NUM_FOXES).astype(np.float32)
fox_speed = np.full(NUM_FOXES, FOX_CRUISE_SPEED, dtype=np.float32)
fox_stamina = np.ones(NUM_FOXES, dtype=np.float32)     # 1.0 = full tank
fox_exhausted = np.zeros(NUM_FOXES, dtype=bool)
fox_hop_phase = np.zeros(NUM_FOXES, dtype=np.float32)

# --- camera tracking indices (mirror predators.best_idx / predators.sel_idx) ---
best_fox_idx = 0        # fox of the highest score-RATE pair, closest to prey
sel_fox_idx = 0         # TAB-cycled in the fox camera modes (hud_cam)
sel_used = False        # True once TAB has picked a fox in a fox cam (mode 5
                        # then follows sel_fox_idx instead of best_fox_idx)

# smoothed pitch (deg) along the local terrain slope, for fox_quat()
_fox_pitch_deg = np.zeros(NUM_FOXES, dtype=np.float32)


def fox_quat(idx):
    """Unit quaternion [x,y,z,w] for fox idx: yaw from fox_yaw + a pitch
    sampled from the terrain slope 1.0m ahead (smoothed), so the fox chase
    and first-person cams tilt naturally over hills instead of clipping."""
    yaw = float(fox_yaw[idx])
    fx, fz = np.sin(yaw), np.cos(yaw)
    x, z = float(fox_pos[idx, 0]), float(fox_pos[idx, 2])
    h0 = float(terrain.height_at(x, z))
    h1 = float(terrain.height_at(x + fx, z + fz))
    slope_deg = np.degrees(np.arctan2(h1 - h0, 1.0))
    _fox_pitch_deg[idx] += (slope_deg - _fox_pitch_deg[idx]) * 0.15
    q_yaw = _q_from_aa_b(np.array([[0.0, 1.0, 0.0]]), np.array([np.degrees(yaw)]))[0]
    # pitch about the body right axis r = f x up = (cos yaw, 0, -sin yaw);
    # positive slope (ground rising ahead) must pitch the nose UP
    q_pitch = _q_from_aa_b(np.array([[np.cos(yaw), 0.0, -np.sin(yaw)]]),
                           np.array([-float(_fox_pitch_deg[idx])]))[0]
    q = _q_mul_b(q_yaw[None, :], q_pitch[None, :])[0]
    return q / np.linalg.norm(q)

# --- pair bookkeeping (Moran process) ---
pair_score = np.zeros(NUM_PAIRS, dtype=np.float32)
pair_survival_time = np.zeros(NUM_PAIRS, dtype=np.float32)
# STAGGERED hunger clocks: all-zero clocks make every fresh pair hit
# starve_seconds at the same frame -> synchronized mass-replacement waves
# that spike Fox Evo. Random 0-20s offsets keep replacements spread out,
# and each replaced pair gets a fresh random offset (see _pair_starve_check).
pair_starvation = _fox_rng.uniform(0.0, 20.0, size=NUM_PAIRS).astype(np.float32)
pair_kills = np.zeros(NUM_PAIRS, dtype=np.int32)
fox_evolutions = 0
fox_total_catches = 0

FOX_STALK_MIN_SPEED = FOX_CRUISE_SPEED * FOX_STALK_FACTOR  # silent creep floor

# stalk-bonus geometry (module constants, not config-tuned)
_STALK_BONUS_RADIUS = 35.0    # m: "stalking" proximity for the +3/s bonus
_STALK_AIM_DOT = 0.5          # nose must point roughly at the rabbit
_SCRAPE_MARGIN = 10.0         # m from the arena edge counted as scraping
_WALL_SENSOR = 80.0           # wall-repulsion sensor range (obs 11)
_STATE_MAP = np.array([-1.0, -1.0, 0.0, 1.0], dtype=np.float32)  # 0/1,2,3 -> -1,0,+1


def update_foxes():
    """One frame for the whole fox pack: obs -> pair MLPs -> yaw/throttle
    physics -> scoring -> starvation replacement. Pure NumPy, true m/s."""
    global fox_pos, fox_yaw, fox_speed, fox_stamina, fox_exhausted
    global pair_score, pair_survival_time, pair_starvation
    import rabbits   # lazy: rabbits imports THIS module at top level

    dt = min(max(time.dt, 0.0), 0.05)

    # --- observations ---
    diff = rabbits.rabbit_pos[None, :, :] - fox_pos[:, None, :]   # (F,R,3)
    dist = np.linalg.norm(diff, axis=2)                           # (F,R)
    near = np.argmin(dist, axis=1)                                # (F,)
    dmin = dist[np.arange(NUM_FOXES), near]
    dir2 = diff[np.arange(NUM_FOXES), near][:, [0, 2]] / np.maximum(dmin[:, None], 1e-6)

    f2 = np.stack([np.sin(fox_yaw), np.cos(fox_yaw)], axis=1)     # (F,2)
    r2 = np.stack([np.cos(fox_yaw), -np.sin(fox_yaw)], axis=1)

    rv = rabbits.rabbit_vel[near][:, [0, 2]] / RABBIT_FLEE_SPEED
    rstate = _STATE_MAP[rabbits.rabbit_state[near]]

    # the rabbit's escape route: nearest burrow from ITS point of view
    rxz = rabbits.rabbit_pos[near][:, [0, 2]]
    rel = rabbits.burrow_xz[None, :, :] - rxz[:, None, :]         # (F,B,2)
    nb = np.argmin(np.einsum('ijb,ijb->ij', rel, rel), axis=1)
    to_b = rel[np.arange(NUM_FOXES), nb]
    db = np.linalg.norm(to_b, axis=1)
    bd = to_b / np.maximum(db[:, None], 1e-6)

    # own speed sensor: -1 at the silent creep floor, 0 at cruise, +1 sprint
    spd_norm = np.where(
        fox_speed <= FOX_CRUISE_SPEED,
        (fox_speed - FOX_STALK_MIN_SPEED) / max(FOX_CRUISE_SPEED - FOX_STALK_MIN_SPEED, 1e-6) - 1.0,
        (fox_speed - FOX_CRUISE_SPEED) / max(FOX_BOOST_SPEED - FOX_CRUISE_SPEED, 1e-6))

    obs = np.empty((NUM_FOXES, FOX_NET_SHAPE[0]), dtype=np.float32)
    obs[:, 0] = np.sum(dir2 * f2, axis=1)
    obs[:, 1] = np.sum(dir2 * r2, axis=1)
    obs[:, 2] = np.clip(dmin / FOX_PROXIMITY_RADIUS, 0.0, 1.0)
    obs[:, 3] = np.sum(rv * f2, axis=1)
    obs[:, 4] = np.sum(rv * r2, axis=1)
    obs[:, 5] = rstate
    obs[:, 6] = np.sum(bd * f2, axis=1)
    obs[:, 7] = np.sum(bd * r2, axis=1)
    obs[:, 8] = np.clip(db / 50.0, 0.0, 1.0)
    obs[:, 9] = np.clip(spd_norm, -1.0, 1.0)
    obs[:, 10] = np.where(fox_exhausted, -1.0, -0.8 + 1.8 * fox_stamina)
    # wall repulsion projected on the heading (same construction as the
    # starling/hawk sensors, range _WALL_SENSOR around the rabbit arena line)
    rep_x = (1.0 - np.clip((_fox_edge - np.abs(fox_pos[:, 0])) / _WALL_SENSOR, 0.0, 1.0)) * -np.sign(fox_pos[:, 0])
    rep_z = (1.0 - np.clip((_fox_edge - np.abs(fox_pos[:, 2])) / _WALL_SENSOR, 0.0, 1.0)) * -np.sign(fox_pos[:, 2])
    obs[:, 11] = rep_x * f2[:, 0] + rep_z * f2[:, 1]
    np.clip(obs, -1.0, 1.0, out=obs)

    # --- batched pair forward ---
    a = obs.reshape(NUM_PAIRS, FOXES_PER_PAIR, FOX_NET_SHAPE[0])
    for W, b in zip(pair_Ws, pair_bs):
        a = np.tanh(np.einsum('pfi,pio->pfo', a, W) + b[:, None, :])
    out = a.reshape(NUM_FOXES, FOX_NET_SHAPE[-1])

    # --- stamina-gated throttle (same either/or + lockout design as hawks) ---
    raw_thr = out[:, 1]
    thr = np.where(fox_exhausted | (fox_stamina <= 0.0),
                   np.minimum(raw_thr, 0.0), raw_thr)
    boosting = thr > 0.0
    delta = np.where(boosting,
                     -thr * dt / max(FOX_BOOST_DURATION, 1e-6),
                     dt / max(FOX_BOOST_RECHARGE, 1e-6))
    fox_stamina += delta                     # in-place keeps float32
    np.clip(fox_stamina, 0.0, 1.0, out=fox_stamina)
    # empty tank trips the lockout; it clears only once the tank recovers
    # to boost_cooldown_recover (same either/or + lockout design as hawks)
    fox_exhausted |= fox_stamina <= 1e-5
    fox_exhausted[fox_stamina >= (FOX_BOOST_COOLDOWN_RECOVER - 1e-5)] = False

    # --- heading + speed physics ---
    yaw_rate = out[:, 0] * FOX_YAW_RATE           # deg/s
    fox_yaw += np.deg2rad(yaw_rate) * dt
    # target speed from throttle: -1 -> creep, 0 -> cruise, +1 -> sprint
    target = np.where(thr >= 0.0,
                      FOX_CRUISE_SPEED + thr * (FOX_BOOST_SPEED - FOX_CRUISE_SPEED),
                      FOX_CRUISE_SPEED * (FOX_STALK_FACTOR + (1.0 - FOX_STALK_FACTOR) * (thr + 1.0)))
    # induced turn drag: sharp rudders bleed speed (out[:,0] is the rudder)
    f2 = np.stack([np.sin(fox_yaw), np.cos(fox_yaw)], axis=1)
    fox_speed += (target - fox_speed) * min(1.0, 2.0 * dt)
    fox_speed -= FOX_TURN_DRAG * np.abs(out[:, 0]) * fox_speed * dt
    np.clip(fox_speed, FOX_STALK_MIN_SPEED, FOX_BOOST_SPEED, out=fox_speed)

    fox_pos[:, 0] += f2[:, 0] * fox_speed * dt
    fox_pos[:, 2] += f2[:, 1] * fox_speed * dt
    fox_pos[:, 0] = np.clip(fox_pos[:, 0], -_fox_edge, _fox_edge)
    fox_pos[:, 2] = np.clip(fox_pos[:, 2], -_fox_edge, _fox_edge)
    fox_pos[:, 1] = terrain.height_at(fox_pos[:, 0], fox_pos[:, 2])

    # --- pair scoring (per-fox rates summed per pair) ---
    prox = (dmin < FOX_PROXIMITY_RADIUS).astype(np.float32) * 0.5
    stalking = ((dmin < _STALK_BONUS_RADIUS)
                & (rabbits.rabbit_state[near] != 3)
                & (fox_speed < FOX_CRUISE_SPEED)
                & (np.sum(dir2 * f2, axis=1) > _STALK_AIM_DOT))
    bonus = stalking.astype(np.float32) * 3.0
    edge_dist = _fox_edge - np.maximum(np.abs(fox_pos[:, 0]), np.abs(fox_pos[:, 2]))
    scrape = (edge_dist < _SCRAPE_MARGIN).astype(np.float32) * 1.5
    fox_rate = prox + bonus - scrape
    pair_score += dt * np.bincount(fox_pair_np, weights=fox_rate, minlength=NUM_PAIRS)
    pair_survival_time += np.float32(dt)

    # hunger clock: creeps at full rate, slowed while the pair stalks silently
    pair_stalking = np.zeros(NUM_PAIRS, dtype=bool)
    np.maximum.at(pair_stalking, fox_pair_np, stalking)
    pair_starvation += dt * np.where(pair_stalking,
                                     1.0 - FOX_STALK_HUNGER_RELIEF, 1.0)

    # --- camera: best fox = member of the top score-RATE pair closest to a
    # rabbit (mirrors predators.best_idx; fox_quat/speed/stamina read by hud) ---
    global best_fox_idx
    pair_rate = pair_score / np.maximum(pair_survival_time, 10.0)
    best_P = int(np.argmax(pair_rate))
    members = np.arange(NUM_FOXES)[fox_pair_np == best_P]
    best_fox_idx = int(members[np.argmin(dmin[members])])

    _pair_starve_check()


def register_fox_catch(caught_fox_indices):
    """Called by rabbits.update_rabbits() when a fox touches a rabbit:
    credits the pair (+50 score, kill), resets its hunger clock."""
    global fox_total_catches
    idx = np.asarray(caught_fox_indices)
    if idx.size == 0:
        return
    pids = fox_pair_np[idx]
    np.add.at(pair_score, pids, 50.0)
    np.add.at(pair_kills, pids, 1)
    pair_starvation[pids] = 0.0
    fox_total_catches += int(idx.size)


def _respawn_pair(P):
    """Put the pair's foxes back inside the arena at cruise, full tank."""
    sl = slice(P * FOXES_PER_PAIR, (P + 1) * FOXES_PER_PAIR)
    x = _fox_rng.uniform(-_fox_edge + 20, _fox_edge - 20, FOXES_PER_PAIR)
    z = _fox_rng.uniform(-_fox_edge + 20, _fox_edge - 20, FOXES_PER_PAIR)
    fox_pos[sl, 0] = x
    fox_pos[sl, 2] = z
    fox_pos[sl, 1] = terrain.height_at(x, z)
    fox_yaw[sl] = _fox_rng.uniform(-np.pi, np.pi, FOXES_PER_PAIR)
    fox_speed[sl] = FOX_CRUISE_SPEED
    fox_stamina[sl] = 1.0
    fox_exhausted[sl] = False


def _pair_starve_check():
    """Moran replacement: a pair that starves is swapped for a mutated copy
    of a top-rate parent (score RATE ranking + maturity gate, exactly like
    flock._tribe_respawn_check — no global reset, active stalks survive)."""
    global fox_evolutions
    starving = np.where(pair_starvation >= FOX_STARVE_SECONDS)[0]
    if len(starving) == 0:
        return
    import predators   # lazy (foxes <-> predators cycle, same as flock)

    # ONE replacement per call — the single most starved pair. Batch-wiping
    # every starving pair at once turns any residual clock sync into a
    # visible Fox Evo spike wave.
    P = int(starving[np.argmax(pair_starvation[starving])])
    alive = np.setdiff1d(np.arange(NUM_PAIRS), [P])
    rate = pair_score / np.maximum(pair_survival_time, 10.0)
    mature = alive[pair_survival_time[alive] >= 15.0] if len(alive) else alive
    pool = mature if len(mature) else alive
    if len(pool) == 0:
        pool = np.array([int(np.argmax(rate))])   # all starving: best of a bad lot
    parent = int(pool[np.argmax(rate[pool])])

    child = pair_pop[parent] + _fox_rng.normal(
        0.0, FOX_MUTATION_SIGMA, size=FOX_GENOME_LEN).astype(np.float32)
    pair_pop[P] = child
    pair_score[P] = 0.0
    pair_survival_time[P] = 0.0
    # staggered re-entry clock (like the fresh init) so replacements never
    # re-synchronize into a second wave
    pair_starvation[P] = _fox_rng.uniform(0.0, 10.0)
    pair_kills[P] = 0
    _respawn_pair(P)
    _upload_pair_layers()
    fox_evolutions += 1
    predators.log_event(
        f"[fox_nn] pair {P} replaced by parent {parent} "
        f"| top_pair_rate {float(rate[pool][np.argmax(rate[pool])]):.3f}")
    predators.save_checkpoint()


# --- FoxVisuals: batched dart mesh (mirrors RabbitVisuals) ---
class FoxVisuals(Entity):
    def __init__(self):
        super().__init__()
        self.model = Mesh(vertices=[], triangles=[], colors=[], mode='triangle', static=False)
        self.texture = 'white_cube'
        # rgba32, NOT rgb(): in this ursina build rgb() is 0..1-scale
        # (rgb = rgba alias) — rgb(215,85,25) stores a 215x color scale that
        # clamps to WHITE in the shader (probed: case1 white vs case2 rust).
        self.color = color.rgba32(*COLOR_FOX)   # sleek rust-orange darts
        self.double_sided = True
        self.fox_shape = np.array([[0, 0, 0.85], [-0.28, 0, -0.45], [0.28, 0, -0.45]], dtype=np.float32)

    def update(self):
        update_foxes()
        # speed-scaled bounding hop: creepers sit flat, sprinters bound
        speed = fox_speed
        fox_hop_phase[:] += speed * 1.6 * min(max(time.dt, 0.0), 0.05)
        hop = np.abs(np.sin(fox_hop_phase)) * np.clip(speed / FOX_BOOST_SPEED, 0.0, 1.0) * 0.5
        yaws = fox_yaw
        c, s = np.cos(yaws), np.sin(yaws)
        verts = np.tile(self.fox_shape, (NUM_FOXES, 1, 1))
        x, z = verts[:, :, 0].copy(), verts[:, :, 2].copy()   # copies: yaw aliasing note
        verts[:, :, 0] = x * c[:, None] + z * s[:, None]
        verts[:, :, 2] = -x * s[:, None] + z * c[:, None]
        verts += fox_pos[:, None, :]
        verts[:, :, 1] += hop[:, None]
        self.model.vertices = verts.reshape(-1, 3).tolist()
        self.model.generate()
