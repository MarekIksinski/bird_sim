# -*- coding: utf-8 -*-
"""
wolves.py — apex ground predator pairs (layer 7.6: hunt foxes AND rabbits;
rabbits/foxes/predators are reached by LAZY imports inside update_wolves,
mirroring the foxes<->rabbits and predators<->flock cycles).

8 wolves in 4 PAIRS; each pair shares one MLP (Moran process on
starvation, exactly like foxes.py: contiguous blocks wolf i -> pair
i // WOLVES_PER_PAIR, batched einsum over (P, ni, no) NumPy tensors —
pure NumPy, true m/s, no cp).

THE EXPERIMENT: the MLP has THREE outputs — yaw, throttle, and a HOWL
vocal signal in [-1, 1]. Every other wolf within WOLF_HOWL_RADIUS hears
the loudest/nearest packmate (obs 9..12, weighted by distance /
(1 + |signal|) so an actively howling wolf draws attention from farther
away), and a kill inside WOLF_ASSIST_RADIUS shares score + hunger relief
with EVERY nearby wolf's pair — including wolves from DIFFERENT pairs —
with an extra WOLF_HOWL_COOP_BONUS when someone was actively howling.
Cooperative pincer hunting and vocal communication can therefore emerge
across pair boundaries; nothing about the reward is pair-local.

ALWAYS access this module's state through the module namespace
(wolves.wolf_pos, wolves.pair_pop, ...): updates and replacements REBIND names.
"""
import numpy as np
from ursina import Entity, Mesh, color, time

from config import (NUM_WOLVES, NUM_WOLF_PAIRS, WOLVES_PER_PAIR, TERRAIN_SIZE,
                    WOLF_CRUISE_SPEED, WOLF_BOOST_SPEED, WOLF_STALK_FACTOR,
                    WOLF_YAW_RATE, WOLF_TURN_DRAG,
                    WOLF_BOOST_DURATION, WOLF_BOOST_RECHARGE,
                    WOLF_BOOST_COOLDOWN_RECOVER, WOLF_CATCH_RADIUS,
                    WOLF_PROXIMITY_RADIUS, WOLF_HOWL_RADIUS,
                    WOLF_ASSIST_RADIUS, WOLF_ASSIST_SHARE,
                    WOLF_HOWL_COOP_BONUS, WOLF_STARVE_SECONDS,
                    WOLF_MUTATION_SIGMA, WOLF_HIDDEN_LAYERS,
                    FOX_BOOST_SPEED, COLOR_WOLF, COLOR_WOLF_HOWL)
import terrain
from math3d import _q_from_aa_b, _q_mul_b

_wolf_rng = np.random.default_rng(8765)
_wolf_edge = TERRAIN_SIZE / 2 - 10.0          # same arena line rabbits/foxes clamp to

# --- pair MLPs: (16 sensors) + hidden + (3 outputs), one per pair ---
# 0,1   nearest-FOX dir on body axes f=(sin yaw, cos yaw), r=(cos yaw,-sin yaw)
# 2     fox distance / WOLF_PROXIMITY_RADIUS
# 3,4   fox velocity / FOX_BOOST_SPEED on f,r          (high-value prey)
# 5,6   nearest-rabbit dir on f,r
# 7     rabbit distance / WOLF_PROXIMITY_RADIUS
# 8     rabbit fleeing? (state 3 -> +1.0, else -1.0)   (secondary prey)
# 9,10  nearest packmate dir on f,r — the pick is weighted by
#       dist / (1 + |signal|): a loudly howling wolf attracts from farther
# 11    packmate distance / WOLF_HOWL_RADIUS
# 12    HEARD HOWL: wolf_signal[mate] in [-1,1] within howl radius, else 0
#       (the open inter-pair communication channel)
# 13    own speed (-1 stalk .. 0 cruise .. +1 sprint)
# 14    stamina sensor (-1.0 exhausted, else -0.8 + 1.8 * stamina)
# 15    arena wall repulsion projected on f (within 80m of the edge)
# Outputs: [0] yaw rate (x WOLF_YAW_RATE deg/s), [1] throttle (stamina-gated),
#          [2] HOWL vocal signal -> wolf_signal (read by every other wolf)
WOLF_NET_SHAPE = (16,) + WOLF_HIDDEN_LAYERS + (3,)
WOLF_GENOME_LEN = sum(WOLF_NET_SHAPE[i] * WOLF_NET_SHAPE[i+1] +
                      WOLF_NET_SHAPE[i+1] for i in range(len(WOLF_NET_SHAPE) - 1))
wolf_pair_np = np.arange(NUM_WOLVES) // WOLVES_PER_PAIR

pair_pop = np.random.default_rng(123).normal(
    0.0, 0.5, size=(NUM_WOLF_PAIRS, WOLF_GENOME_LEN)).astype(np.float32)

def _unpack_pair_genome(g):
    """flat float32 vector -> list of (W, b) pairs for WOLF_NET_SHAPE."""
    layers, off = [], 0
    for i in range(len(WOLF_NET_SHAPE) - 1):
        ni, no = WOLF_NET_SHAPE[i], WOLF_NET_SHAPE[i + 1]
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
                    for p in range(NUM_WOLF_PAIRS)]).astype(np.float32)
          for li in range(len(WOLF_NET_SHAPE) - 1)]
    bs = [np.stack([_unpack_pair_genome(pair_pop[p])[li][1]
                    for p in range(NUM_WOLF_PAIRS)]).astype(np.float32)
          for li in range(len(WOLF_NET_SHAPE) - 1)]
    pair_Ws.extend(Ws)
    pair_bs.extend(bs)

_upload_pair_layers()

# --- wolf state (pure NumPy, true m/s) ---
wolf_pos = np.zeros((NUM_WOLVES, 3), dtype=np.float32)
_sx = _wolf_rng.uniform(-_wolf_edge + 20, _wolf_edge - 20, NUM_WOLVES)
_sz = _wolf_rng.uniform(-_wolf_edge + 20, _wolf_edge - 20, NUM_WOLVES)
wolf_pos[:, 0] = _sx
wolf_pos[:, 2] = _sz
wolf_pos[:, 1] = terrain.height_at(_sx, _sz)
wolf_yaw = _wolf_rng.uniform(-np.pi, np.pi, NUM_WOLVES).astype(np.float32)
wolf_speed = np.full(NUM_WOLVES, WOLF_CRUISE_SPEED, dtype=np.float32)
wolf_stamina = np.ones(NUM_WOLVES, dtype=np.float32)   # 1.0 = full tank
wolf_exhausted = np.zeros(NUM_WOLVES, dtype=bool)
wolf_signal = np.zeros(NUM_WOLVES, dtype=np.float32)   # last frame's HOWL output [-1,1]
wolf_hop_phase = np.zeros(NUM_WOLVES, dtype=np.float32)

# --- camera tracking indices (mirror foxes.best_fox_idx / sel_fox_idx) ---
best_wolf_idx = 0       # wolf of the highest score-RATE pair, closest to prey
sel_wolf_idx = 0        # TAB-cycled in the wolf camera modes (hud_cam)
sel_used = False        # True once TAB has picked a wolf in a wolf cam

# smoothed pitch (deg) along the local terrain slope, for wolf_quat()
_wolf_pitch_deg = np.zeros(NUM_WOLVES, dtype=np.float32)


def wolf_quat(idx):
    """Unit quaternion [x,y,z,w] for wolf idx: yaw + terrain-slope pitch
    (same construction as foxes.fox_quat) so the wolf cams ride the hills."""
    yaw = float(wolf_yaw[idx])
    fx, fz = np.sin(yaw), np.cos(yaw)
    x, z = float(wolf_pos[idx, 0]), float(wolf_pos[idx, 2])
    h0 = float(terrain.height_at(x, z))
    h1 = float(terrain.height_at(x + fx, z + fz))
    slope_deg = np.degrees(np.arctan2(h1 - h0, 1.0))
    _wolf_pitch_deg[idx] += (slope_deg - _wolf_pitch_deg[idx]) * 0.15
    q_yaw = _q_from_aa_b(np.array([[0.0, 1.0, 0.0]]), np.array([np.degrees(yaw)]))[0]
    # pitch about the body right axis r = f x up = (cos yaw, 0, -sin yaw);
    # positive slope (ground rising ahead) must pitch the nose UP
    q_pitch = _q_from_aa_b(np.array([[np.cos(yaw), 0.0, -np.sin(yaw)]]),
                           np.array([-float(_wolf_pitch_deg[idx])]))[0]
    q = _q_mul_b(q_yaw[None, :], q_pitch[None, :])[0]
    return q / np.linalg.norm(q)

# --- pair bookkeeping (Moran process) ---
pair_score = np.zeros(NUM_WOLF_PAIRS, dtype=np.float32)
pair_survival_time = np.zeros(NUM_WOLF_PAIRS, dtype=np.float32)
# STAGGERED hunger clocks (anti mass-extinction wave, same lesson as foxes):
# spread across 0 .. 75% of the starve window, and each check replaces ONLY
# the single most-starved pair with a fresh random re-entry clock.
pair_starvation = _wolf_rng.uniform(0.0, WOLF_STARVE_SECONDS * 0.75,
                                    size=NUM_WOLF_PAIRS).astype(np.float32)
pair_kills = np.zeros(NUM_WOLF_PAIRS, dtype=np.int32)
wolf_evolutions = 0
wolf_fox_catches = 0
wolf_rabbit_catches = 0
wolf_coop_kills = 0     # assisted kills where someone was actively howling

WOLF_STALK_MIN_SPEED = WOLF_CRUISE_SPEED * WOLF_STALK_FACTOR  # 1.8 m/s stalk floor

# scoring geometry (module constants, not config-tuned)
_SCRAPE_MARGIN = 10.0    # m from the arena edge counted as scraping
_WALL_SENSOR = 80.0      # wall-repulsion sensor range (obs 15)
_PINCER_INNER = 45.0     # m: this wolf stalking a fox...
_PINCER_OUTER = 60.0     # m: ...while another wolf flanks the SAME fox
_FLANK_DOT = -0.2        # from the fox's POV the two wolves must be on
                         # roughly opposite sides (dir dot below this)


def update_wolves():
    """One frame for the whole wolf pack: obs -> pair MLPs -> yaw/throttle/
    howl physics -> catches + pack-share rewards -> scoring -> starvation
    replacement. Pure NumPy, true m/s."""
    global wolf_pos, wolf_yaw, wolf_speed, wolf_stamina, wolf_exhausted
    global wolf_signal, pair_score, pair_survival_time, pair_starvation
    global wolf_fox_catches, wolf_rabbit_catches, wolf_coop_kills
    import rabbits   # lazy: rabbits may import this module at top level
    import foxes     # lazy: foxes imports rabbits/predators; cycle-safe

    dt = min(max(time.dt, 0.0), 0.05)

    f2 = np.stack([np.sin(wolf_yaw), np.cos(wolf_yaw)], axis=1)   # (W,2)
    r2 = np.stack([np.cos(wolf_yaw), -np.sin(wolf_yaw)], axis=1)

    # --- prey sensors: nearest fox (high value) + nearest rabbit ---
    fdiff = foxes.fox_pos[None, :, :] - wolf_pos[:, None, :]      # (Wf,F,3)
    fdist = np.linalg.norm(fdiff, axis=2)                         # (Wf,F)
    fnear = np.argmin(fdist, axis=1)
    d_fox = fdist[np.arange(NUM_WOLVES), fnear]
    fd2 = fdiff[np.arange(NUM_WOLVES), fnear][:, [0, 2]] / np.maximum(d_fox[:, None], 1e-6)
    # fox heading velocity reconstructed from speed+yaw (foxes keep no vel array)
    ff2 = np.stack([np.sin(foxes.fox_yaw), np.cos(foxes.fox_yaw)], axis=1)  # (F,2)
    fv = (ff2[fnear] * foxes.fox_speed[fnear][:, None]) / FOX_BOOST_SPEED

    rdiff = rabbits.rabbit_pos[None, :, :] - wolf_pos[:, None, :]  # (Wf,R,3)
    rdist = np.linalg.norm(rdiff, axis=2)                          # (Wf,R)
    rnear = np.argmin(rdist, axis=1)
    d_rab = rdist[np.arange(NUM_WOLVES), rnear]
    rd2 = rdiff[np.arange(NUM_WOLVES), rnear][:, [0, 2]] / np.maximum(d_rab[:, None], 1e-6)
    rstate = np.where(rabbits.rabbit_state[rnear] == 3, 1.0, -1.0)

    # --- packmate / howl channel (ALL other wolves, pair boundaries open) ---
    mdiff = wolf_pos[None, :, :] - wolf_pos[:, None, :]            # (i,j,3) i->j
    mdist = np.linalg.norm(mdiff, axis=2)                          # (Wf,Wf)
    # pick the NEAREST or LOUDEST packmate: loud howls shrink the effective
    # distance, so an actively howling wolf attracts attention from farther
    eff = mdist / (1.0 + np.abs(wolf_signal)[None, :])
    np.fill_diagonal(eff, np.inf)
    mate = np.argmin(eff, axis=1)
    d_mate = mdist[np.arange(NUM_WOLVES), mate]
    md2 = mdiff[np.arange(NUM_WOLVES), mate][:, [0, 2]] / np.maximum(d_mate[:, None], 1e-6)
    heard = np.where(d_mate <= WOLF_HOWL_RADIUS, wolf_signal[mate], 0.0)

    # --- self / arena proprioception ---
    spd_norm = np.where(
        wolf_speed <= WOLF_CRUISE_SPEED,
        (wolf_speed - WOLF_STALK_MIN_SPEED) / max(WOLF_CRUISE_SPEED - WOLF_STALK_MIN_SPEED, 1e-6) - 1.0,
        (wolf_speed - WOLF_CRUISE_SPEED) / max(WOLF_BOOST_SPEED - WOLF_CRUISE_SPEED, 1e-6))
    rep_x = (1.0 - np.clip((_wolf_edge - np.abs(wolf_pos[:, 0])) / _WALL_SENSOR, 0.0, 1.0)) * -np.sign(wolf_pos[:, 0])
    rep_z = (1.0 - np.clip((_wolf_edge - np.abs(wolf_pos[:, 2])) / _WALL_SENSOR, 0.0, 1.0)) * -np.sign(wolf_pos[:, 2])

    obs = np.empty((NUM_WOLVES, WOLF_NET_SHAPE[0]), dtype=np.float32)
    obs[:, 0] = np.sum(fd2 * f2, axis=1)
    obs[:, 1] = np.sum(fd2 * r2, axis=1)
    obs[:, 2] = np.clip(d_fox / WOLF_PROXIMITY_RADIUS, 0.0, 1.0)
    obs[:, 3] = np.sum(fv * f2, axis=1)
    obs[:, 4] = np.sum(fv * r2, axis=1)
    obs[:, 5] = np.sum(rd2 * f2, axis=1)
    obs[:, 6] = np.sum(rd2 * r2, axis=1)
    obs[:, 7] = np.clip(d_rab / WOLF_PROXIMITY_RADIUS, 0.0, 1.0)
    obs[:, 8] = rstate
    obs[:, 9] = np.sum(md2 * f2, axis=1)
    obs[:, 10] = np.sum(md2 * r2, axis=1)
    obs[:, 11] = np.clip(d_mate / WOLF_HOWL_RADIUS, 0.0, 1.0)
    obs[:, 12] = heard
    obs[:, 13] = np.clip(spd_norm, -1.0, 1.0)
    obs[:, 14] = np.where(wolf_exhausted, -1.0, -0.8 + 1.8 * wolf_stamina)
    obs[:, 15] = rep_x * f2[:, 0] + rep_z * f2[:, 1]
    np.clip(obs, -1.0, 1.0, out=obs)

    # --- batched pair forward ---
    a = obs.reshape(NUM_WOLF_PAIRS, WOLVES_PER_PAIR, WOLF_NET_SHAPE[0])
    for W, b in zip(pair_Ws, pair_bs):
        a = np.tanh(np.einsum('pfi,pio->pfo', a, W) + b[:, None, :])
    out = a.reshape(NUM_WOLVES, WOLF_NET_SHAPE[-1])

    # publish the HOWL immediately: catches this frame can trigger the
    # howl-coop bonus, and next frame's packmates hear it via obs 12
    wolf_signal[:] = out[:, 2]

    # --- stamina-gated throttle (same either/or + lockout design as foxes;
    # wolves are endurance cursors: 4.5s tank vs the foxes' 1.8s) ---
    raw_thr = out[:, 1]
    thr = np.where(wolf_exhausted | (wolf_stamina <= 0.0),
                   np.minimum(raw_thr, 0.0), raw_thr)
    boosting = thr > 0.0
    delta = np.where(boosting,
                     -thr * dt / max(WOLF_BOOST_DURATION, 1e-6),
                     dt / max(WOLF_BOOST_RECHARGE, 1e-6))
    wolf_stamina += delta                      # in-place keeps float32
    np.clip(wolf_stamina, 0.0, 1.0, out=wolf_stamina)
    wolf_exhausted |= wolf_stamina <= 1e-5
    wolf_exhausted[wolf_stamina >= (WOLF_BOOST_COOLDOWN_RECOVER - 1e-5)] = False

    # --- heading + speed physics ---
    yaw_rate = out[:, 0] * WOLF_YAW_RATE           # deg/s (wide circle: 95)
    wolf_yaw += np.deg2rad(yaw_rate) * dt
    target = np.where(thr >= 0.0,
                      WOLF_CRUISE_SPEED + thr * (WOLF_BOOST_SPEED - WOLF_CRUISE_SPEED),
                      WOLF_CRUISE_SPEED * (WOLF_STALK_FACTOR + (1.0 - WOLF_STALK_FACTOR) * (thr + 1.0)))
    wolf_speed += (target - wolf_speed) * min(1.0, 2.0 * dt)
    wolf_speed -= WOLF_TURN_DRAG * np.abs(out[:, 0]) * wolf_speed * dt
    np.clip(wolf_speed, WOLF_STALK_MIN_SPEED, WOLF_BOOST_SPEED, out=wolf_speed)

    f2 = np.stack([np.sin(wolf_yaw), np.cos(wolf_yaw)], axis=1)
    wolf_pos[:, 0] += f2[:, 0] * wolf_speed * dt
    wolf_pos[:, 2] += f2[:, 1] * wolf_speed * dt
    wolf_pos[:, 0] = np.clip(wolf_pos[:, 0], -_wolf_edge, _wolf_edge)
    wolf_pos[:, 2] = np.clip(wolf_pos[:, 2], -_wolf_edge, _wolf_edge)
    wolf_pos[:, 1] = terrain.height_at(wolf_pos[:, 0], wolf_pos[:, 2])

    # --- catches: foxes (big meal) and rabbits (snack) ---
    # Fox catch: fresh distances (post-integration) so a wolf that ran the
    # fox down this frame bites it.
    fdiff = foxes.fox_pos[None, :, :] - wolf_pos[:, None, :]
    fdist = np.linalg.norm(fdiff, axis=2)
    fox_eaten = np.where(fdist.min(axis=0) < WOLF_CATCH_RADIUS)[0]
    if fox_eaten.size:
        killers = fdist.argmin(axis=0)[fox_eaten]
        kill_pos = foxes.fox_pos[fox_eaten]
        np.add.at(pair_score, wolf_pair_np[killers], 120.0)
        pair_starvation[wolf_pair_np[killers]] = 0.0
        np.add.at(pair_kills, wolf_pair_np[killers], 1)
        wolf_fox_catches += int(fox_eaten.size)
        _credit_assistants(killers, kill_pos, 120.0, WOLF_STARVE_SECONDS)
        foxes.register_fox_eaten(fox_eaten)

    # Rabbit catch: respawn Frozen at a random burrow (same treatment as the
    # fox-catch/burrow-escape path in rabbits.py).
    rdiff = rabbits.rabbit_pos[None, :, :] - wolf_pos[:, None, :]
    rdist = np.linalg.norm(rdiff, axis=2)
    rab_eaten = np.where(rdist.min(axis=0) < WOLF_CATCH_RADIUS)[0]
    if rab_eaten.size:
        killers = rdist.argmin(axis=0)[rab_eaten]
        kill_pos = rabbits.rabbit_pos[rab_eaten]
        np.add.at(pair_score, wolf_pair_np[killers], 45.0)
        relief = 0.6 * WOLF_STARVE_SECONDS
        pair_starvation[wolf_pair_np[killers]] = np.maximum(
            pair_starvation[wolf_pair_np[killers]] - relief, 0.0)
        np.add.at(pair_kills, wolf_pair_np[killers], 1)
        wolf_rabbit_catches += int(rab_eaten.size)
        _credit_assistants(killers, kill_pos, 45.0, relief)
        new_b = rabbits.burrow_xz[_wolf_rng.integers(
            0, rabbits.NUM_BURROWS, size=rab_eaten.size)]
        rabbits.rabbit_pos[rab_eaten, 0] = new_b[:, 0]
        rabbits.rabbit_pos[rab_eaten, 2] = new_b[:, 1]
        rabbits.rabbit_pos[rab_eaten, 1] = terrain.height_at(new_b[:, 0], new_b[:, 1])
        rabbits.rabbit_vel[rab_eaten] = 0.0
        rabbits.rabbit_state[rab_eaten] = 2      # Frozen at the burrow mouth
        rabbits.rabbit_timer[rab_eaten] = 2.0

    # --- continuous per-frame score rate ---
    prey_close = ((d_fox < WOLF_PROXIMITY_RADIUS) | (d_rab < WOLF_PROXIMITY_RADIUS))
    prox = prey_close.astype(np.float32) * 0.5
    # pack-pincer stalking: this wolf inside 45m of a fox while ANOTHER
    # wolf flanks that same fox from the roughly-opposite side
    srt = np.sort(fdist, axis=0)                 # per-fox column: 1st/2nd nearest wolf dist
    arg2 = np.argsort(fdist, axis=0)[:2]         # (2,F) wolf ids of those two
    # for each wolf i and ITS nearest fox f: the nearest OTHER wolf to f
    _is_nearest = (arg2[0][None, :] == np.arange(NUM_WOLVES)[:, None])  # (W,F)
    sec_d_grid = np.where(_is_nearest, srt[1][None, :], srt[0][None, :])
    sec_w_grid = np.where(_is_nearest, arg2[1][None, :], arg2[0][None, :])
    d_second = sec_d_grid[np.arange(NUM_WOLVES), fnear]
    second = sec_w_grid[np.arange(NUM_WOLVES), fnear]
    near_fox = fnear                              # each wolf's nearest fox
    u_i = -fd2                                    # fox -> me (fd2 is me -> fox)
    fox_xy = foxes.fox_pos[near_fox][:, [0, 2]]
    other_xy = wolf_pos[second][:, [0, 2]]
    u_o = (other_xy - fox_xy)
    u_o /= np.maximum(np.linalg.norm(u_o, axis=1, keepdims=True), 1e-6)
    flanked = ((d_fox < _PINCER_INNER)
               & (d_second < _PINCER_OUTER)
               & (np.sum(u_i * u_o, axis=1) < _FLANK_DOT))
    pincer = flanked.astype(np.float32) * 2.5
    edge_dist = _wolf_edge - np.maximum(np.abs(wolf_pos[:, 0]), np.abs(wolf_pos[:, 2]))
    scrape = (edge_dist < _SCRAPE_MARGIN).astype(np.float32) * 1.5
    wolf_rate = prox + pincer - scrape
    pair_score += dt * np.bincount(wolf_pair_np, weights=wolf_rate, minlength=NUM_WOLF_PAIRS)
    pair_survival_time += np.float32(dt)
    pair_starvation += dt

    # --- camera: best wolf = member of the top score-RATE pair closest to
    # a fox (mirrors foxes.best_fox_idx) ---
    global best_wolf_idx
    pair_rate = pair_score / np.maximum(pair_survival_time, 10.0)
    best_P = int(np.argmax(pair_rate))
    members = np.arange(NUM_WOLVES)[wolf_pair_np == best_P]
    best_wolf_idx = int(members[np.argmin(d_fox[members])])

    _wolf_starve_check()


def _credit_assistants(killers, kill_pos, kill_score, relief):
    """Inter-PAIR pack-share: every other wolf within WOLF_ASSIST_RADIUS of
    the kill feeds — its pair gets WOLF_ASSIST_SHARE of the score and of
    the killer's starvation relief. If the killer OR the assister was
    actively howling (|signal| > 0.5), both pairs also collect the
    +WOLF_HOWL_COOP_BONUS and wolf_coop_kills ticks up: the experimental
    payoff for the open vocal channel."""
    global wolf_coop_kills
    kill_pos = np.asarray(kill_pos, dtype=np.float32)
    d = np.linalg.norm(wolf_pos[None, :, :] - kill_pos[:, None, :], axis=2)  # (kills, W)
    for k_i in range(len(killers)):
        killer = int(killers[k_i])
        assist = np.where((d[k_i] < WOLF_ASSIST_RADIUS)
                          & (np.arange(NUM_WOLVES) != killer))[0]
        if assist.size == 0:
            continue
        apairs = np.unique(wolf_pair_np[assist])
        np.add.at(pair_score, apairs, kill_score * WOLF_ASSIST_SHARE)
        pair_starvation[apairs] = np.maximum(
            pair_starvation[apairs] - relief * WOLF_ASSIST_SHARE, 0.0)
        howling = (abs(float(wolf_signal[killer])) > 0.5
                   or np.any(np.abs(wolf_signal[assist]) > 0.5))
        if howling:
            kp = wolf_pair_np[killer]
            np.add.at(pair_score, apairs, WOLF_HOWL_COOP_BONUS)
            pair_score[kp] += WOLF_HOWL_COOP_BONUS
            wolf_coop_kills += 1


def _respawn_pair(P):
    """Put the pair's wolves back inside the arena at cruise, full tank."""
    sl = slice(P * WOLVES_PER_PAIR, (P + 1) * WOLVES_PER_PAIR)
    x = _wolf_rng.uniform(-_wolf_edge + 20, _wolf_edge - 20, WOLVES_PER_PAIR)
    z = _wolf_rng.uniform(-_wolf_edge + 20, _wolf_edge - 20, WOLVES_PER_PAIR)
    wolf_pos[sl, 0] = x
    wolf_pos[sl, 2] = z
    wolf_pos[sl, 1] = terrain.height_at(x, z)
    wolf_yaw[sl] = _wolf_rng.uniform(-np.pi, np.pi, WOLVES_PER_PAIR)
    wolf_speed[sl] = WOLF_CRUISE_SPEED
    wolf_stamina[sl] = 1.0
    wolf_exhausted[sl] = False
    wolf_signal[sl] = 0.0


def _wolf_starve_check():
    """Moran replacement: the single most-starved pair is swapped for a
    mutated copy of a top-rate parent (score RATE ranking + maturity gate,
    exactly like foxes._pair_starve_check — one replacement per call and a
    10s random re-entry clock so pairs never mass-extinct simultaneously)."""
    global wolf_evolutions
    starving = np.where(pair_starvation >= WOLF_STARVE_SECONDS)[0]
    if len(starving) == 0:
        return
    import predators   # lazy: log_event + save_checkpoint

    P = int(starving[np.argmax(pair_starvation[starving])])
    alive = np.setdiff1d(np.arange(NUM_WOLF_PAIRS), [P])
    rate = pair_score / np.maximum(pair_survival_time, 10.0)
    mature = alive[pair_survival_time[alive] >= 15.0] if len(alive) else alive
    pool = mature if len(mature) else alive
    if len(pool) == 0:
        pool = np.array([int(np.argmax(rate))])   # all starving: best of a bad lot
    parent = int(pool[np.argmax(rate[pool])])

    child = pair_pop[parent] + _wolf_rng.normal(
        0.0, WOLF_MUTATION_SIGMA, size=WOLF_GENOME_LEN).astype(np.float32)
    pair_pop[P] = child
    pair_score[P] = 0.0
    pair_survival_time[P] = 0.0
    pair_starvation[P] = _wolf_rng.uniform(0.0, 10.0)   # grace buffer
    pair_kills[P] = 0
    _respawn_pair(P)
    _upload_pair_layers()
    wolf_evolutions += 1
    predators.log_event(
        f"[wolf_nn] pair {P} replaced by parent {parent} "
        f"| top_pair_rate {float(rate[pool][np.argmax(rate[pool])]):.3f}")
    predators.save_checkpoint()


# --- WolfVisuals: batched dart mesh (mirrors FoxVisuals, larger body +
# per-wolf vertex colors so a howling wolf lights up ice-blue on screen) ---
class WolfVisuals(Entity):
    def __init__(self):
        super().__init__()
        self.model = Mesh(vertices=[], triangles=[], colors=[], mode='triangle', static=False)
        self.texture = 'white_cube'
        # rgba32, NOT rgb(): in this ursina build rgb() is 0..1-scale
        # (rgb = rgba alias) — 0..255 values clamp to WHITE in the shader.
        # Entity color is WHITE: the coat/howl tint lives in the PER-VERTEX
        # colors below (entity color multiplies vertex color — tinting both
        # would square the coat and darken it).
        self.color = color.white
        self.double_sided = True
        self.wolf_shape = np.array([[0, 0, 1.3], [-0.42, 0, -0.65], [0.42, 0, -0.65]], dtype=np.float32)
        self._coat = color.rgba32(*COLOR_WOLF)
        self._howl = color.rgba32(*COLOR_WOLF_HOWL)

    def update(self):
        update_wolves()
        # speed-scaled bounding hop: stalkers sit flat, sprinters bound
        speed = wolf_speed
        wolf_hop_phase[:] += speed * 1.4 * min(max(time.dt, 0.0), 0.05)
        hop = np.abs(np.sin(wolf_hop_phase)) * np.clip(speed / WOLF_BOOST_SPEED, 0.0, 1.0) * 0.6
        yaws = wolf_yaw
        c, s = np.cos(yaws), np.sin(yaws)
        verts = np.tile(self.wolf_shape, (NUM_WOLVES, 1, 1))
        x, z = verts[:, :, 0].copy(), verts[:, :, 2].copy()   # copies: yaw aliasing note
        verts[:, :, 0] = x * c[:, None] + z * s[:, None]
        verts[:, :, 2] = -x * s[:, None] + z * c[:, None]
        verts += wolf_pos[:, None, :]
        verts[:, :, 1] += hop[:, None]
        self.model.vertices = verts.reshape(-1, 3).tolist()
        # per-wolf vertex colors: howling (|signal| > 0.5) wolves light up
        # ice-blue so the vocal channel is VISIBLE on the terrain. Ursina
        # Mesh wants ONE color per VERTEX (W*3), not per wolf.
        howling = np.abs(wolf_signal) > 0.5
        col = np.where(howling[:, None],
                       np.tile(np.array(self._howl, dtype=np.float32), (NUM_WOLVES, 1)),
                       np.tile(np.array(self._coat, dtype=np.float32), (NUM_WOLVES, 1)))
        col = np.repeat(col, self.wolf_shape.shape[0], axis=0)   # (W*3, 4)
        self.model.colors = [tuple(c4) for c4 in col]
        self.model.generate()
