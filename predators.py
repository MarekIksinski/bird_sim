# -*- coding: utf-8 -*-
"""
predators.py — hawk pack state, neuroevolution, flight physics, checkpointing
(layer 5: imports backend/config/math3d/terrain; flock imports this module,
never the other way around — cross-references to flock state are lazy).

ALWAYS access this module's state through the module namespace
(`predators.best_idx`, `predators.population`, ...): generation/evolution
steps REBIND these names, and `from predators import x` would capture a
stale reference.
"""
import os

import numpy as np
from ursina import Entity, Vec3, color, time
from panda3d.core import Quat as PQuat

from backend import to_cpu
from config import (POPULATION_SIZE, NUM_BIRDS, MUTATION_TINY,
                    MUTATION_MEDIUM, MUTATION_HEAVY, HOF_DECAY_KEEP,
                    FITNESS_CATCH,
                    PROXIMITY_RADIUS, PROXIMITY_GAIN, GROUND_PENALTY,
                    GROUND_CLEARANCE, DOGFIGHT_RADIUS, DOGFIGHT_GAIN,
                    BOUNDARY_PENALTY, BOUNDARY_SCRAPE, PERSIST, EVOLVE_DIR,
                    TERRAIN_SIZE, MAX_ALTITUDE, PANIC_RADIUS, CATCH_RADIUS,
                    PRED_BOOST, CRUISE_SPEED, SLOW_FACTOR, YAW_RATE,
                    PITCH_RATE, ROLL_SPEED, PANIC_MAX_SPEED)
import terrain
from math3d import _q_mul_b, _q_from_aa_b, _q_rot_b

# --- 3b. PREDATOR PACK STATE (vectorized over POPULATION_SIZE) ---
# Attitudes are unit quaternions [x,y,z,w] per row. Convention (verified
# numerically against entity.quaternion/e.forward): _q_rot_b's +w*uv form
# matches Panda3D's rotation exactly. Control deltas are built from BODY
# axes and RIGHT-multiply the attitude (q_new = q * dq_body), which is the
# identical rotation as left-multiplying the delta built from the same
# axes' world images.
pack_rng = np.random.default_rng()
pack_pos = np.zeros((POPULATION_SIZE, 3), dtype=np.float32)
pack_vel = np.zeros((POPULATION_SIZE, 3), dtype=np.float32)
pack_speed = np.zeros(POPULATION_SIZE, dtype=np.float32)
pack_q = np.zeros((POPULATION_SIZE, 4), dtype=np.float64)
pack_q[:, 3] = 1.0

def pack_axes():
    """local forward/right/up per hawk: each (N,3)."""
    f = _q_rot_b(pack_q, np.array([0.0, 0.0, 1.0]))   # nose
    r = _q_rot_b(pack_q, np.array([1.0, 0.0, 0.0]))   # right wing
    u = _q_rot_b(pack_q, np.array([0.0, 1.0, 0.0]))   # dorsal
    return f, r, u

def respawn_pack():
    """Respawn all hawks around the map edge, spread out, facing the roost."""
    global gen_elapsed, pack_q, pack_pos, pack_vel, pack_speed
    import flock   # lazy: flock imports this module at top level
    ang = np.linspace(0, 2 * np.pi, POPULATION_SIZE, endpoint=False) + pack_rng.uniform(-0.3, 0.3)
    rad = TERRAIN_SIZE * 0.42
    px = np.cos(ang) * rad; pz = np.sin(ang) * rad
    rc = to_cpu(flock.roost_center)
    pack_pos[:, 0] = px
    pack_pos[:, 2] = pz
    pack_pos[:, 1] = terrain.height_at(px, pz) + 3.0
    pack_vel[:] = 0.0
    pack_speed[:] = CRUISE_SPEED   # airborne from the drop
    # yaw to face the roost center (world-up rotation, 45deg default style)
    yaw = np.degrees(np.arctan2(rc[0] - px, rc[2] - pz))
    ax = np.zeros((POPULATION_SIZE, 3)); ax[:, 1] = 1.0
    pack_q[:] = _q_from_aa_b(ax, yaw)

# --- 3c. NEURAL PACK (neuroevolution) ---
# MLP 12 -> 16 -> 8 -> 4, tanh everywhere; genome = flattened weights+biases.
NET_SHAPE = (12, 16, 8, 4)
GENOME_LEN = sum(NET_SHAPE[i] * NET_SHAPE[i+1] + NET_SHAPE[i+1] for i in range(len(NET_SHAPE) - 1))
# observation normalisation for the target-velocity inputs
OBS_VEL_SCALE = PANIC_MAX_SPEED

population = pack_rng.normal(0.0, 0.5, size=(POPULATION_SIZE, GENOME_LEN))
fitness = np.zeros(POPULATION_SIZE)
gen_kills = np.zeros(POPULATION_SIZE, dtype=np.int32)
total_catches = 0
generation = 1
gen_elapsed = 0.0
best_idx = 0
sel_idx = 0   # TAB-cycled hawk for chase / first-person cameras

# --- King of the Hill (Hall of Fame): the all-time best genome ---
# The reigning King always flies in Slot 0 and is the Tier-1 breeding
# parent; its rating is re-evaluated live via an EMA (see evolve_generation)
# so a lucky one-generation fluke cannot lock Slot 0 forever.
hof_genome = None      # 1D float64 copy, shape (GENOME_LEN,) once crowned
hof_fitness = -1e9     # float; EMA-blended all-time standing of the King

# --- persistence: keep the evolved pack between runs (folder from config) ---
_CKPT = os.path.join(EVOLVE_DIR, 'population.npz')
_HISTORY = os.path.join(EVOLVE_DIR, 'history.csv')
_STARLING_CKPT = os.path.join(EVOLVE_DIR, 'starling_tribes.npz')

def load_checkpoint():
    """Restore population/generation counters from the checkpoint if it
    matches the current config (population_size, genome length)."""
    global population, generation, total_catches
    global hof_genome, hof_fitness
    if not PERSIST or not os.path.exists(_CKPT):
        return
    try:
        d = np.load(_CKPT)
        pop = d['population']
        if pop.shape != (POPULATION_SIZE, GENOME_LEN):
            print(f"[persist] checkpoint shape {pop.shape} != config "
                  f"({POPULATION_SIZE},{GENOME_LEN}); starting fresh")
            return
        population = pop.astype(np.float64)
        generation = int(d['generation'])
        total_catches = int(d['total_catches'])
        # King of the Hill state (backward compatible: old checkpoints
        # lack these keys -> the loaded Slot 0 genome becomes the
        # incumbent King so Tier 1 keeps the evolved champion).
        if 'hof_genome' in d.files and 'hof_fitness' in d.files \
                and d['hof_genome'].shape == (GENOME_LEN,):
            hof_genome = d['hof_genome'].astype(np.float64)
            hof_fitness = float(d['hof_fitness'])
        else:
            hof_genome = population[0].copy()
            hof_fitness = 0.0
            print("[persist] no HoF in checkpoint: Slot 0 becomes incumbent King")
        print(f"[persist] resumed at gen {generation} from {_CKPT}")
    except Exception as e:
        print(f"[persist] could not load checkpoint ({e}); starting fresh")
    # starling tribe genomes (separate file, own shape guard)
    if os.path.exists(_STARLING_CKPT):
        try:
            import flock   # lazy: flock imports this module at top level
            s = np.load(_STARLING_CKPT)
            tp = s['tribe_pop']
            if tp.shape != (flock.NUM_TRIBES, flock.STARLING_GENOME_LEN):
                print(f"[persist] starling checkpoint shape {tp.shape} != config "
                      f"({flock.NUM_TRIBES},{flock.STARLING_GENOME_LEN}); tribes start fresh")
                return
            flock.tribe_pop = tp.astype(np.float32)
            flock.starling_evolutions = int(s['starling_evolutions'])
            print(f"[persist] {flock.starling_evolutions} starling tribe evolutions resumed")
        except Exception as e:
            print(f"[persist] could not load starling checkpoint ({e}); tribes start fresh")

def save_checkpoint():
    """Persist the CURRENT population under the CURRENT generation number.
    Call AFTER evolve_generation has built the new population and bumped
    `generation` — a checkpoint must always be a consistent (gen, pop) pair,
    otherwise a resume replays the previous generation's already-evaluated
    genomes and silently skips one selection step."""
    if not PERSIST:
        return
    try:
        os.makedirs(EVOLVE_DIR, exist_ok=True)
        np.savez(_CKPT, population=population, generation=generation,
                 total_catches=total_catches,
                 hof_genome=hof_genome if hof_genome is not None else population[0],
                 hof_fitness=hof_fitness)
        import flock   # lazy: the starling checkpoint lives in flock's state
        np.savez(_STARLING_CKPT, tribe_pop=flock.tribe_pop,
                 starling_evolutions=flock.starling_evolutions)
    except Exception as e:
        print(f"[persist] could not save checkpoint ({e})")

def append_history():
    """One progression row per completed generation. Must run while fitness
    and gen_kills still hold this generation's results (i.e. BEFORE they are
    zeroed for the next one)."""
    if not PERSIST:
        return
    try:
        new = not os.path.exists(_HISTORY)
        with open(_HISTORY, 'a') as fh:
            if new:
                fh.write('gen,best_fitness,mean_fitness,total_kills\n')
            fh.write(f'{generation},{fitness.max():.1f},{fitness.mean():.1f},'
                     f'{int(gen_kills.sum())}\n')
    except Exception as e:
        print(f"[persist] could not write history ({e})")

def _unpack_genome(g):
    """flat vector -> list of (W, b) pairs."""
    layers = []; k = 0
    for i in range(len(NET_SHAPE) - 1):
        ni, no = NET_SHAPE[i], NET_SHAPE[i + 1]
        W = g[k:k + ni * no].reshape(ni, no); k += ni * no
        b = g[k:k + no]; k += no
        layers.append((W, b))
    return layers

def _stack_population():
    """population (N,GENOME_LEN) -> per-layer stacked arrays so the MLP
    forward runs vectorized across ALL hawks at once:
    W_list[i] shape (N, ni, no), b_list[i] shape (N, no)."""
    Ws, bs = [], []
    for i in range(len(NET_SHAPE) - 1):
        ni, no = NET_SHAPE[i], NET_SHAPE[i + 1]
        Ws.append(np.empty((POPULATION_SIZE, ni, no)))
        bs.append(np.empty((POPULATION_SIZE, no)))
    for h in range(POPULATION_SIZE):
        for li, (W, b) in enumerate(_unpack_genome(population[h])):
            Ws[li][h] = W; bs[li][h] = b
    return Ws, bs

def _nearest_bird_np(pos_np, vel_np):
    """Per-hawk nearest starling: (N,3) dir, dist, (N,3) bird vel, (N,3) bird pos.
    Chunked over birds so the (POP,NUM_BIRDS) scratch never blows up memory."""
    d2 = np.empty((POPULATION_SIZE, pos_np.shape[0]), dtype=np.float32)
    step = 1024
    for s in range(0, pos_np.shape[0], step):
        diff = pos_np[s:s + step][None, :, :] - pack_pos[:, None, :]
        d2[:, s:s + diff.shape[1]] = np.einsum('ijk,ijk->ij', diff, diff)
    nb = np.argmin(d2, axis=1)
    near_pos = pos_np[nb]
    near_vel = vel_np[nb]
    dist = np.sqrt(d2[np.arange(POPULATION_SIZE), nb])
    direction = near_pos - pack_pos
    direction /= np.maximum(dist[:, None], 1e-6)
    return direction, dist, near_vel, near_pos

def _hawk_observations(pos_np, vel_np):
    """12 normalized inputs per hawk: line-of-sight axis cues, clamped
    distance, target-velocity axis cues, terrain clearance, arena awareness
    (wall-repulsion projected on f/r, edge proximity, ceiling clearance)."""
    f, r, u = pack_axes()
    direction, dist, near_vel, _ = _nearest_bird_np(pos_np, vel_np)
    clearance = pack_pos[:, 1] - terrain.height_at(pack_pos[:, 0], pack_pos[:, 2])
    obs = np.empty((POPULATION_SIZE, 12), dtype=np.float64)
    obs[:, 0] = np.einsum('ij,ij->i', f, direction)
    obs[:, 1] = np.einsum('ij,ij->i', r, direction)
    obs[:, 2] = np.einsum('ij,ij->i', u, direction)
    obs[:, 3] = np.clip(dist / PROXIMITY_RADIUS, 0.0, 1.0)
    vn = near_vel / OBS_VEL_SCALE
    obs[:, 4] = np.einsum('ij,ij->i', f, vn)
    obs[:, 5] = np.einsum('ij,ij->i', r, vn)
    obs[:, 6] = np.einsum('ij,ij->i', u, vn)
    obs[:, 7] = np.clip(clearance / 50.0, 0.0, 1.0)
    # --- arena awareness (in [-1,1]): wall-proximity vector in world XZ.
    # Points INWARD (-sign) with magnitude ramping 0 -> 1 as the hawk nears
    # the wall (inside wall_sensor): a repulsion warning that is LOUDEST at
    # the wall. (The old form ramped 1 -> 0, vanishing exactly when scraping.)
    half = TERRAIN_SIZE / 2 - 1.0
    wall_sensor = 100.0
    rep_x = (1.0 - np.clip((half - np.abs(pack_pos[:, 0])) / wall_sensor, 0.0, 1.0)) * -np.sign(pack_pos[:, 0])
    rep_z = (1.0 - np.clip((half - np.abs(pack_pos[:, 2])) / wall_sensor, 0.0, 1.0)) * -np.sign(pack_pos[:, 2])
    obs[:, 8] = rep_x * f[:, 0] + rep_z * f[:, 2]   # "safety is ahead"
    obs[:, 9] = rep_x * r[:, 0] + rep_z * r[:, 2]   # "...is to my right"
    edge = np.minimum(half - np.abs(pack_pos[:, 0]), half - np.abs(pack_pos[:, 2]))
    obs[:, 10] = np.clip(edge / wall_sensor, 0.0, 1.0)   # 0 = at wall, 1 = safe
    obs[:, 11] = np.clip((MAX_ALTITUDE - pack_pos[:, 1]) / 30.0, 0.0, 1.0)
    return np.clip(obs, -1.0, 1.0)

def update_predators(layers, pos_np, vel_np):
    """All hawks fly simultaneously: NN -> rates/throttle -> attitude+velocity."""
    global pack_q, pack_pos, pack_vel, pack_speed
    # Attitude dt is CAPPED: a heavy frame (CPU-fallback boids, GC hitch) can
    # hand us time.dt of seconds, and rate*dt rotations then snap 180+ deg in
    # a single frame — the "pointing straight up flips to straight down" bug.
    # Cap at 50ms (same policy as dt_scale in the boids step).
    adt = min(max(time.dt, 0.0), 0.05)
    obs = _hawk_observations(pos_np, vel_np)
    # forward pass: batched MLP across the whole pack (tanh in and out)
    a = obs
    for W, b in layers:
        a = np.tanh(np.einsum('ni,nio->no', a, W) + b)
    out = a
    # --- TRUE 6-DOF attitude: yaw/pitch/roll about the hawk's OWN axes.
    # Deltas are built from BODY-frame axes (fixed unit vectors) and RIGHT-
    # multiply the attitude: q_new = q * dq_yaw * dq_pitch * dq_roll. That is
    # the identical rotation as left-multiplying a delta built from those
    # axes' world images, so the sign conventions are unchanged (positive
    # about body r = nose down; positive about body u = heading psi up when
    # level) — but there is no world_up yaw axis and no +-85deg pitch clamp,
    # so loops and barrel rolls pass through vertical with no gimbal pole
    # and no 180deg snap.
    body_r = np.tile(np.array([1.0, 0.0, 0.0]), (POPULATION_SIZE, 1))
    body_u = np.tile(np.array([0.0, 1.0, 0.0]), (POPULATION_SIZE, 1))
    body_f = np.tile(np.array([0.0, 0.0, 1.0]), (POPULATION_SIZE, 1))
    dq = _q_from_aa_b(body_u, out[:, 0] * YAW_RATE * adt)
    dq = _q_mul_b(dq, _q_from_aa_b(body_r, out[:, 1] * PITCH_RATE * adt))
    dq = _q_mul_b(dq, _q_from_aa_b(body_f, out[:, 2] * ROLL_SPEED * adt))
    pack_q = _q_mul_b(pack_q, dq)
    pack_q /= np.linalg.norm(pack_q, axis=1, keepdims=True)
    # --- boundary attitude deflection: rotate the NOSE back into the arena.
    # Velocity is f*speed, so deflecting only pack_vel (below) is pointless —
    # it gets overwritten next frame while the nose still points at the wall.
    # NOTE: every correction below is a RATE (deg/s) times adt — never a raw
    # per-frame angle (an old version multiplied by 60 AND dt and flipped a
    # nose-up hawk 180deg in one frame). adt is capped so a heavy-frame dt
    # spike can't fire a giant rotation.
    f, r, u = pack_axes()
    half = TERRAIN_SIZE / 2 - 1.0
    soft = 25.0
    for ax in (0, 2):
        # signed outward direction of the wall this hawk is near (-1 or +1 axis),
        # and how deep in the soft zone it is (0..1)
        out_sign = np.where(pack_pos[:, ax] > 0, 1.0, -1.0)
        edge_in = half - np.abs(pack_pos[:, ax])          # distance to that wall
        depth = np.clip((soft - edge_in) / soft, 0.0, 1.0)
        # nose component pointing outward through that wall (only steer when
        # the nose actually aims at the wall — never push an inward nose)
        outward = np.maximum(f[:, ax] * out_sign, 0.0)    # >0 = aiming at wall
        # Minimum-rotation axis that carries the nose f toward the inward
        # normal. WORLD-frame axis -> LEFT-multiply. The rotation direction is
        # correct by construction (f rotates toward inward), so no psi sign
        # derivation is needed, and it stays valid for a pitched/looping nose
        # where a world-up yaw would be degenerate. When f already points
        # inward, cross() ~ 0 and _q_from_aa_b returns identity.
        inward_vec = np.zeros((POPULATION_SIZE, 3))
        inward_vec[:, ax] = -out_sign
        steer_axis = np.cross(f, inward_vec)
        # cross() vanishes when the nose points EXACTLY at the wall (the 180deg
        # worst case) — fall back to world up (perpendicular to a horizontal
        # nose; the turn then sweeps sideways off the wall in either direction).
        degen = np.linalg.norm(steer_axis, axis=1) < 1e-6
        steer_axis[degen] = (0.0, 1.0, 0.0)
        turn = np.minimum(outward * depth * 120.0, 60.0) * adt   # deg/s rate, capped
        pack_q = _q_mul_b(_q_from_aa_b(steer_axis, turn), pack_q)
    # ceiling: nose pitched UP while at/above ceiling -> rotate the nose toward
    # straight down via the same minimum-rotation construction (valid at any
    # bank/loop attitude, unlike pitch about local r when inverted).
    at_ceiling = pack_pos[:, 1] > MAX_ALTITUDE - 1.0
    down_vec = np.zeros((POPULATION_SIZE, 3)); down_vec[:, 1] = -1.0
    ceil_axis = np.cross(f, down_vec)
    # cross() vanishes with the nose pointing straight up at the ceiling —
    # fall back to a horizontal axis (a roll-neutral pitch-over in +X).
    ceil_degen = np.linalg.norm(ceil_axis, axis=1) < 1e-6
    ceil_axis[ceil_degen] = (1.0, 0.0, 0.0)
    nose_up = np.clip(f[:, 1], 0.0, 1.0)          # >0 = nose above horizon
    ceil_turn = np.where(at_ceiling,
                         np.minimum(np.degrees(np.arcsin(nose_up)) * 2.0, 60.0),
                         0.0) * adt              # deg/s rate, capped
    pack_q = _q_mul_b(_q_from_aa_b(ceil_axis, ceil_turn), pack_q)
    pack_q /= np.linalg.norm(pack_q, axis=1, keepdims=True)
    # --- throttle: -1 slow .. 0 cruise .. +1 boost (smooth approach) ---
    thr = out[:, 3]
    target = np.where(thr >= 0,
                      CRUISE_SPEED + thr * (PRED_BOOST - CRUISE_SPEED),
                      CRUISE_SPEED * (1.0 + thr * (1.0 - SLOW_FACTOR)))
    pack_speed += (target - pack_speed) * min(time.dt * 1.5, 1.0)
    # --- velocity always along the nose ---
    f, r, u = pack_axes()
    pack_vel[:] = f * pack_speed[:, None]
    # --- invisible walls: soft deflection, then hard clamp (vectorized) ---
    half = TERRAIN_SIZE / 2 - 1.0
    soft = 15.0
    for ax in (0, 2):
        depth = np.maximum(-half + soft - pack_pos[:, ax], 0.0)
        inward = np.where(depth > 0, 1.0, 0.0)
        depth2 = np.maximum(pack_pos[:, ax] - (half - soft), 0.0)
        inward = inward - np.where(depth2 > 0, 1.0, 0.0)
        depth_in = np.maximum(depth, depth2) / soft
        k = np.minimum(depth_in * 1.2, 1.0)
        target_v = inward * pack_speed
        pack_vel[:, ax] = pack_vel[:, ax] * (1 - k) + target_v * k
    pack_pos += pack_vel * time.dt * 60
    pack_pos[:, 0] = np.clip(pack_pos[:, 0], -half, half)
    pack_pos[:, 2] = np.clip(pack_pos[:, 2], -half, half)
    ground = terrain.height_at(pack_pos[:, 0], pack_pos[:, 2])
    pack_pos[:, 1] = np.minimum(MAX_ALTITUDE,
                                np.maximum(pack_pos[:, 1], ground + 0.4))
    if not np.all(np.isfinite(pack_pos)):
        bad = np.where(~np.isfinite(pack_pos).all(axis=1))[0]
        print(f"[nan-guard] hawks {bad} went non-finite; q={pack_q[bad[0]]} spd={pack_speed[bad[0]]}")
        respawn_pack()

def evolve_generation():
    """Rank -> 3-tiered speciation: Tier 1 breeds the ALL-TIME King of the
    Hill (Hall of Fame genome, re-evaluated live via EMA), Tiers 2/3 breed
    this generation's top non-King challengers. Each lineage keeps an
    untouched clone plus offspring at a lineage-specific mutation strength.
    Tier 1 (King, tiny): fine-tune the all-time best. Tier 2 (runner-up,
    medium): explore around second place. Tier 3 (wildcard, heavy):
    high-variance lottery tickets. Hardcoded 10-slot layout."""
    global population, fitness, gen_kills, generation, gen_elapsed, sel_idx
    global hof_genome, hof_fitness
    order = np.argsort(-fitness)
    # history row FIRST: fitness/gen_kills still hold THIS generation's
    # results (and `generation` is still the completed gen's number).
    append_history()

    # --- King of the Hill: live re-evaluation & crowning ---
    # The reigning King always flew in Slot 0 this generation, so blend
    # 70% of its historical standing with 30% of its LIVE Slot-0 fitness
    # (HOF_DECAY_KEEP from config). Why an EMA instead of "best ever":
    # fitness is stochastic (prey positions, spawn luck), so a mediocre
    # genome can score one lucky spike and, under a max-ever rule, lock
    # Slot 0 forever while never reproducing that score. The live blend
    # pulls a fluke King's rating down toward its true performance over
    # a few generations until a real challenger crosses it.
    if hof_genome is not None:
        hof_fitness = HOF_DECAY_KEEP * hof_fitness + (1.0 - HOF_DECAY_KEEP) * float(fitness[0])
    crowned = hof_genome is None or fitness[order[0]] >= hof_fitness
    if crowned:
        hof_genome = population[order[0]].copy()
        hof_fitness = float(fitness[order[0]])
    print(f"[evolution] gen {generation}: best fitness {fitness[order[0]]:.0f}, "
          f"kills best {gen_kills[order[0]]}, total {gen_kills.sum()} | "
          f"HoF {'CROWNED' if crowned else 'holds'} at {hof_fitness:.0f}")

    new_pop = np.empty_like(population)
    # --- Tier 1: King lineage (exploitation) — the all-time best genome
    # always flies untouched in Slot 0; slots 1-3 are tiny-mutation clones.
    new_pop[0] = hof_genome.copy()
    for i in (1, 2, 3):
        new_pop[i] = hof_genome + pack_rng.normal(
            0.0, MUTATION_TINY, size=GENOME_LEN)
    # --- Tier 2/3 parents: top challengers, never the King's own genome.
    # If order[0] was just crowned (or IS the King in slot 0), its genome
    # is already in new_pop[0] — skip that slot so the same genome isn't
    # cloned into both Tier 1 and Tier 2. If the old King kept the crown
    # while a different hawk won the generation, that challenger still
    # gets to breed: promote order[0]/order[1] past slot 0.
    king_slot = 0 if not crowned else int(order[0])
    challengers = [int(i) for i in order if int(i) != king_slot][:2]
    tier2_parent, tier3_parent = challengers[0], challengers[1]
    # --- Tier 2: runner-up lineage (medium exploration) ---
    new_pop[4] = population[tier2_parent]                   # untouched runner-up
    for i in (5, 6):
        new_pop[i] = population[tier2_parent] + pack_rng.normal(
            0.0, MUTATION_MEDIUM, size=GENOME_LEN)
    # --- Tier 3: wildcard lineage (heavy exploration) ---
    new_pop[7] = population[tier3_parent]                   # untouched wildcard
    for i in (8, 9):
        new_pop[i] = population[tier3_parent] + pack_rng.normal(
            0.0, MUTATION_HEAVY, size=GENOME_LEN)
    population = new_pop
    fitness[:] = 0.0
    gen_kills[:] = 0
    generation += 1
    gen_elapsed = 0.0
    sel_idx = 0
    # checkpoint AFTER the selection step: the saved (population, generation)
    # pair is exactly what the next generation will fly, so a resume never
    # replays already-evaluated genomes or skips a selection round.
    save_checkpoint()
    respawn_pack()

# (the batched per-frame weight stacking happens in update_pack/_stack_population)

def update_pack(pos_np, vel_np):
    """One simulation frame for the whole pack: NN flight + panic/catch/fitness."""
    global total_catches, best_idx, fitness, gen_kills
    Ws, bs = _stack_population()
    layers = list(zip(Ws, bs))
    update_predators(layers, pos_np, vel_np)

    # --- hawk <-> flock interaction (chunked over birds) ---
    step = 1024
    all_caught = np.zeros(NUM_BIRDS, dtype=bool)
    caught_by = np.full(NUM_BIRDS, -1, dtype=np.int32)
    nearest_dir = np.zeros((NUM_BIRDS, 3), dtype=np.float32)
    nearest_d = np.full(NUM_BIRDS, 1e9, dtype=np.float32)
    hawk_dmin = np.full(POPULATION_SIZE, 1e9, dtype=np.float32)  # per-hawk dist to nearest bird
    for s in range(0, NUM_BIRDS, step):
        diff = pack_pos[:, None, :] - pos_np[s:s + step][None, :, :]  # (POP, chunk, 3)
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        d = np.sqrt(d2)
        hawk_dmin = np.minimum(hawk_dmin, d.min(axis=1))
        # per-bird nearest hawk
        bh = np.argmin(d, axis=0)
        rows = np.arange(d.shape[1])
        bd = d[bh, rows]
        nearest_dir[s:s + len(rows)] = -diff[bh, rows] / np.maximum(bd[:, None], 1e-6)
        nearest_d[s:s + len(rows)] = bd
        hit = d < CATCH_RADIUS
        if hit.any():
            first_hawk = np.argmax(hit, axis=0)
            caugh_cols = np.any(hit, axis=0)
            all_caught[s:s + len(rows)] = caugh_cols
            caught_by[s:s + len(rows)] = np.where(caugh_cols, first_hawk, -1)
    # panic: flee from the NEAREST active predator (stronger the closer)
    panic_mask = nearest_d < PANIC_RADIUS
    flee = nearest_dir * panic_mask[:, None]
    flee *= np.clip(1.0 - nearest_d / PANIC_RADIUS, 0, 1)[:, None]
    # (flee applied to vel_gpu by the boids step via these numpy arrays)
    caught_idx = np.where(all_caught)[0]
    if len(caught_idx):
        np.add.at(gen_kills, caught_by[caught_idx], 1)
        np.add.at(fitness, caught_by[caught_idx], FITNESS_CATCH)
        total_catches += int(len(caught_idx))
    # --- per-frame fitness: catches + proximity ramp + dogfighting bonus
    #     - ground/wall/ceiling scrape penalties ---
    fitness += PROXIMITY_GAIN * np.clip(1.0 - hawk_dmin / PROXIMITY_RADIUS, 0.0, 1.0)
    # dogfighting: inside DOGFIGHT_RADIUS, reward closing AND aiming:
    # (range term) x (alignment of nose with the target). Tail-chasers earn a
    # much higher baseline than hawks loitering at 80m, smoothing the
    # catch-lottery boom/bust (catches still worth FITNESS_CATCH).
    direction, dist, _nv, _np_ = _nearest_bird_np(pos_np, vel_np)
    f, r, u = pack_axes()
    aim = np.einsum('ij,ij->i', f, direction)          # 1 = nose on target
    range_term = np.clip(1.0 - hawk_dmin / DOGFIGHT_RADIUS, 0.0, 1.0)
    fitness += DOGFIGHT_GAIN * range_term * range_term * np.clip(aim, 0.0, 1.0)
    clearance = pack_pos[:, 1] - terrain.height_at(pack_pos[:, 0], pack_pos[:, 2])
    fitness -= (clearance < GROUND_CLEARANCE) * GROUND_PENALTY
    half = TERRAIN_SIZE / 2 - 1.0
    edge = np.minimum(half - np.abs(pack_pos[:, 0]), half - np.abs(pack_pos[:, 2]))
    fitness -= (edge < BOUNDARY_SCRAPE) * BOUNDARY_PENALTY
    fitness -= (pack_pos[:, 1] > MAX_ALTITUDE - 1.0) * BOUNDARY_PENALTY
    best_idx = int(np.argmax(fitness))
    # stash for the boids step (applied there AFTER its own clamp chain).
    # nearest_dir/nearest_d are per-bird (NUM_BIRDS): dir points AWAY from
    # the nearest hawk — the starling NN sensory input reuses this pass.
    return flee, all_caught, caught_by, nearest_dir, nearest_d

# The hawk pack — the GLB models + rank colors (spawned by init_pack_visuals
# AFTER Ursina() exists; update_pack_visuals rides them every frame).
pack_entities = []
GOLD = color.rgb(255, 200, 40)
MUTED = color.rgb(150, 70, 50)
BRIGHT = color.rgb(230, 50, 40)

def init_pack_visuals():
    """Spawn the hawk entities — call only AFTER app = Ursina(...) exists."""
    for i in range(POPULATION_SIZE):
        e = Entity(model='assets/hawk.glb', color=color.rgb(150, 70, 50))
        pack_entities.append(e)

def update_pack_visuals():
    """--- hawk visuals + rank colors (top performer gold, runner-up bright) ---"""
    order = np.argsort(-fitness)
    for i, e in enumerate(pack_entities):
        e.position = Vec3(*pack_pos[i])
        q = pack_q[i]
        e.quaternion = PQuat(q[3], q[0], q[1], q[2])
        if i == order[0]:
            e.color = GOLD
        elif i == order[1]:
            e.color = BRIGHT
        else:
            e.color = MUTED
