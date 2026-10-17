# -*- coding: utf-8 -*-
"""
flock.py — starling GPU/CPU state, tribe NNs, boids step, flock visuals
(layer 6: imports backend/config/terrain/predators; predators' references
back into this module are lazy imports inside functions).

ALWAYS access this module's state through the module namespace
(`flock.pos_gpu`, `flock.roost_center`, `flock.tribe_pop`, ...): the boids
step REBINDS pos_gpu/vel_gpu/roost_center every frame and evolution rebinds
tribe tensors — `from flock import x` would capture a stale reference.
"""
import numpy as np
from ursina import Entity, Mesh, color, time

from backend import cp, to_cpu
from config import (NUM_BIRDS, NEIGHBOR_COUNT, SEPARATION_DIST, ROOST_RADIUS,
                    FORCE_SEPARATE, FORCE_ALIGN, FORCE_COHERE, FORCE_ROOST,
                    FORCE_CENTER, ROOST_ORBIT_RADIUS, ROOST_ORBIT_SPEED,
                    ORBIT_WOBBLE_RADIUS, ORBIT_WOBBLE_SPEED,
                    ROOST_BASE_ALTITUDE, VERTICAL_SPEED, VERTICAL_AMPLITUDE,
                    MAX_SPEED, MAX_FORCE, NUM_TRIBES, BIRDS_PER_TRIBE,
                    STARLING_MUTATION_SIGMA, TRIBE_REPLACE_DEATHS,
                    CASUALTY_DECAY, TERRAIN_SIZE, PANIC_RADIUS, FORCE_FLEE,
                    PANIC_MAX_SPEED, GENERATION_SECONDS,
                    FLOCK_GRAVITY, MIN_BIRD_SPEED, DIVE_SPEED_MULT,
                    CLIMB_SPEED_MULT)
import predators
import terrain

# --- 3. GPU PHYSICS (Topological) ---
# dtype=float32 is REQUIRED: an untyped cp.array([0, alt, 0]) is int-float64
# mixed -> the sum promotes pos_gpu to float64, and the whole N x N distance/
# argsort/neighbor chain then runs in double precision (2x the memory: 72MB
# vs 36MB per matrix at 3000 birds, and far slower on consumer GPUs).
pos_gpu = (cp.random.random((NUM_BIRDS, 3)).astype(cp.float32) * 40 - 20) + cp.array([0, ROOST_BASE_ALTITUDE, 0], dtype=cp.float32)
vel_gpu = cp.random.random((NUM_BIRDS, 3)).astype(cp.float32) - 0.5
vel_gpu = vel_gpu / cp.linalg.norm(vel_gpu, axis=1, keepdims=True) * MAX_SPEED

# Roost Marker (Invisible center point)
roost_center = cp.array([0, ROOST_BASE_ALTITUDE, 0], dtype=cp.float32)

# --- 3a. FLOCK NEURAL NETS (tribe genomes, co-evolving vs the hawk pack) ---
# One MLP per tribe (NUM_TRIBES small nets, NOT one per bird): birds are
# CONTIGUOUS blocks (bird i -> tribe i // BIRDS_PER_TRIBE) so the forward
# pass reshapes to (T, BIRDS_PER_TRIBE, n) and einsums against tiny
# (T, ni, no) weights — fast on CPU cache AND on GPU, identical code paths.
# (17 inputs since the gravity update: +bird airspeed, +vertical pitch f_y,
# so tribes can learn to dive for speed and manage climb stalls. The
# starling checkpoint shape guard starts fresh tribes on mismatch.)
STARLING_NET_SHAPE = (17, 16, 8, 6)
STARLING_GENOME_LEN = sum(STARLING_NET_SHAPE[i] * STARLING_NET_SHAPE[i+1] +
                          STARLING_NET_SHAPE[i+1] for i in range(len(STARLING_NET_SHAPE) - 1))
bird_tribe_np = np.arange(NUM_BIRDS) // BIRDS_PER_TRIBE

tribe_pop = np.random.default_rng(1234).normal(
    0.0, 0.5, size=(NUM_TRIBES, STARLING_GENOME_LEN)).astype(np.float32)

def _unpack_starling_genome(g):
    """flat float32 vector -> list of (W, b) pairs for STARLING_NET_SHAPE."""
    layers = []; k = 0
    for i in range(len(STARLING_NET_SHAPE) - 1):
        ni, no = STARLING_NET_SHAPE[i], STARLING_NET_SHAPE[i + 1]
        W = g[k:k + ni * no].reshape(ni, no); k += ni * no
        b = g[k:k + no]; k += no
        layers.append((W, b))
    return layers

def _upload_tribe_layers():
    """Rebuild the cp-resident per-layer tensors from tribe_pop. Only called
    at startup and when a tribe genome actually mutates — NOT every frame."""
    global tribe_Ws, tribe_bs
    tribe_Ws, tribe_bs = [], []
    for li in range(len(STARLING_NET_SHAPE) - 1):
        ni, no = STARLING_NET_SHAPE[li], STARLING_NET_SHAPE[li + 1]
        W = np.empty((NUM_TRIBES, ni, no), dtype=np.float32)
        b = np.empty((NUM_TRIBES, no), dtype=np.float32)
        for t in range(NUM_TRIBES):
            layers = _unpack_starling_genome(tribe_pop[t])
            W[t] = layers[li][0]; b[t] = layers[li][1]
        tribe_Ws.append(cp.asarray(W))
        tribe_bs.append(cp.asarray(b))

_upload_tribe_layers()

# Darwinian-respawn bookkeeping (lightweight CPU arrays)
bird_survival_time = np.zeros(NUM_BIRDS, dtype=np.float32)
bird_score = np.zeros(NUM_BIRDS, dtype=np.float32)
tribe_casualties = np.zeros(NUM_TRIBES, dtype=np.float32)  # float: decay + fractional accrual
starling_evolutions = 0
_starling_rng = np.random.default_rng()

def _tribe_respawn_check(caught_idx):
    """Darwinian respawn (Moran process): caught birds zero their score and
    tally tribe casualties; a tribe that loses TRIBE_REPLACE_DEATHS birds is
    replaced by a mutated copy of a top-3-scoring tribe. CPU/NumPy only;
    re-uploads cp layer tensors ONLY when a tribe actually mutated."""
    global tribe_casualties, starling_evolutions
    for b in caught_idx:
        bird_survival_time[b] = 0.0
        bird_score[b] = 0.0
        tribe_casualties[bird_tribe_np[b]] += 1
    dead = np.where(tribe_casualties >= TRIBE_REPLACE_DEATHS)[0]
    if len(dead) == 0:
        return
    tribe_means = bird_score.reshape(NUM_TRIBES, BIRDS_PER_TRIBE).mean(axis=1)
    # Parent pool: top-3 LIVE tribes only. A dead tribe must never parent
    # itself (the old 1/3 self-draw wiped an evolved brain to pure random),
    # and a tribe currently being replaced must not seed another replacement.
    dead_set = set(int(t) for t in dead)
    pool = [int(t) for t in np.argsort(-tribe_means) if int(t) not in dead_set][:3]
    mutated = False
    for T in dead:
        if pool:
            parent = int(_starling_rng.choice(pool))
            tribe_pop[T] = tribe_pop[parent] + _starling_rng.normal(
                0.0, STARLING_MUTATION_SIGMA, size=STARLING_GENOME_LEN)
        else:                               # every tribe dead -> fresh random
            tribe_pop[T] = _starling_rng.normal(0.0, 0.5, size=STARLING_GENOME_LEN)
        # the survivors inherit the NEW brain: score the fresh genome from
        # zero, otherwise old-brain survival points keep propping up a tribe
        # (or poisoning a replacement) that hasn't earned them
        bird_score[bird_tribe_np == T] = 0.0
        bird_survival_time[bird_tribe_np == T] = 0.0
        tribe_casualties[T] = 0
        starling_evolutions += 1
        mutated = True
    if mutated:
        _upload_tribe_layers()                # only on actual mutation
        print(f"[flock_nn] tribe(s) {list(dead)} replaced "
              f"(total starling evolutions: {starling_evolutions})")

# --- 6. SIMULATION LOOP (boids step, predator interaction folded in) ---
def update_boids_gpu():
    global pos_gpu, vel_gpu, roost_center
    global bird_survival_time, bird_score, tribe_casualties

    # Delta-time scale: *60 so all config.toml speeds/forces keep their
    # current magnitude and feel (they are tuned per-frame at 60 FPS).
    # Capped at 50ms: a hitch (first frame, window drag) must not fire a
    # giant force/position jump through the flock.
    dt_scale = min(time.dt, 0.05) * 60

    # 0. ROOST MIGRATION — main orbit + a secondary off-beat wave (x1.37 on
    # X, x0.79 on Z). The asymmetrical frequencies keep the path from ever
    # closing on itself, so the roost sweeps the map organically. Y rides the
    # terrain AND oscillates on a slow vertical wave: true 3D murmuration.
    t = time.time() * ROOST_ORBIT_SPEED
    wob_r = ROOST_ORBIT_RADIUS * ORBIT_WOBBLE_RADIUS
    new_x = np.cos(t) * ROOST_ORBIT_RADIUS + np.sin(t * 1.37 * ORBIT_WOBBLE_SPEED) * wob_r
    new_z = np.sin(t) * ROOST_ORBIT_RADIUS + np.cos(t * 0.79 * ORBIT_WOBBLE_SPEED) * wob_r
    # stacked amplitudes can exceed the map — keep the roost inside it
    edge = TERRAIN_SIZE / 2 - 5.0
    new_x = float(cp.clip(new_x, -edge, edge))
    new_z = float(cp.clip(new_z, -edge, edge))
    base_altitude = float(terrain.height_at(new_x, new_z)) + ROOST_BASE_ALTITUDE
    new_y = base_altitude + np.sin(time.time() * VERTICAL_SPEED) * VERTICAL_AMPLITUDE
    roost_center = cp.array([new_x, new_y, new_z], dtype=cp.float32)

    # read flock state once per frame (pack logic needs numpy; boids stay on backend)
    pos_np = to_cpu(pos_gpu)
    vel_np = to_cpu(vel_gpu)

    # --- predator pack step BEFORE the boids rules: panic/catch results feed in ---
    predators.gen_elapsed += time.dt
    if predators.gen_elapsed >= GENERATION_SECONDS:
        predators.evolve_generation()
        pos_np = to_cpu(pos_gpu)  # respawns may have moved birds last frame; cheap refresh
    flee_np, caught_np, caught_by, nearest_dir_np, nearest_d_np = predators.update_pack(pos_np, vel_np)

    # 1. Distances
    diff_matrix = pos_gpu[:, None, :] - pos_gpu[None, :, :]
    dist_sq_matrix = cp.sum(diff_matrix**2, axis=2)

    # 2. Topological Neighbors (Nearest 7)
    sorted_indices = cp.argsort(dist_sq_matrix, axis=1)[:, 1:NEIGHBOR_COUNT+1]
    neighbor_pos = pos_gpu[sorted_indices]
    neighbor_vel = vel_gpu[sorted_indices]

    # 3. Local Rules
    # Alignment
    avg_vel = cp.mean(neighbor_vel, axis=1)
    align_force = avg_vel - vel_gpu

    # Cohesion (Local Group Hug)
    avg_pos = cp.mean(neighbor_pos, axis=1)
    cohere_force = avg_pos - pos_gpu # Vector pointing to center of neighbors

    # Separation
    my_pos_expanded = pos_gpu[:, None, :]
    rel_pos = my_pos_expanded - neighbor_pos
    dists = cp.linalg.norm(rel_pos, axis=2, keepdims=True)
    sep_mask = dists < SEPARATION_DIST
    sep_push = (rel_pos / (dists + 0.001)) * sep_mask
    sep_force = cp.sum(sep_push, axis=1)

    # 4. Roost (Keep within bounds)
    to_roost = roost_center - pos_gpu
    dist_to_roost = cp.linalg.norm(to_roost, axis=1, keepdims=True)
    roost_force = cp.where(dist_to_roost > ROOST_RADIUS, to_roost, 0)
    roost_force = roost_force / (cp.linalg.norm(roost_force, axis=1, keepdims=True) + 0.001)

    # 5. GLOBAL CENTER GRAVITY (The "Glue")
    flock_center = cp.mean(pos_gpu, axis=0)
    to_flock_center = flock_center - pos_gpu
    center_force = to_flock_center / (cp.linalg.norm(to_flock_center, axis=1, keepdims=True) + 0.001)

    # 5b. STARLING NEURAL NETS (tribe MLPs): local-frame observations ->
    # boids-weight modulation + evasive juke. All on cp, float32; the
    # forward pass reshapes to contiguous tribe blocks and einsums against
    # tiny (NUM_TRIBES, ni, no) weights (no (NUM_BIRDS, ni, no) materialized).
    speed_cur = cp.linalg.norm(vel_gpu, axis=1, keepdims=True) + 1e-6
    f = vel_gpu / speed_cur                                    # forward
    world_up = cp.array([[0.0, 1.0, 0.0]], dtype=cp.float32)
    r = cp.cross(world_up, f)                                  # right wing
    r_norm = cp.linalg.norm(r, axis=1, keepdims=True)
    r = cp.where(r_norm > 1e-4, r / cp.maximum(r_norm, 1e-6),
                 cp.array([[1.0, 0.0, 0.0]], dtype=cp.float32))  # vertical fallback
    u = cp.cross(f, r)                                         # dorsal up

    nearest_dir_gpu = cp.asarray(nearest_dir_np, dtype=cp.float32)   # away from hawk
    nearest_d_gpu = cp.asarray(nearest_d_np, dtype=cp.float32)
    ground_np_obs = np.asarray(terrain.height_at(pos_np[:, 0], pos_np[:, 2]))
    clearance_obs = pos_gpu[:, 1] - cp.array(ground_np_obs, dtype=cp.float32)

    cohere_dir = cohere_force / (cp.linalg.norm(cohere_force, axis=1, keepdims=True) + 1e-6)
    align_norm = avg_vel / MAX_SPEED
    sep_norm = sep_force / (cp.linalg.norm(sep_force, axis=1, keepdims=True) + 1.0)

    obs_gpu = cp.empty((NUM_BIRDS, 17), dtype=cp.float32)
    # threat: nearest-hawk direction in the local frame + proximity
    obs_gpu[:, 0] = cp.sum(nearest_dir_gpu * f, axis=1)
    obs_gpu[:, 1] = cp.sum(nearest_dir_gpu * r, axis=1)
    obs_gpu[:, 2] = cp.sum(nearest_dir_gpu * u, axis=1)
    obs_gpu[:, 3] = cp.clip(1.0 - nearest_d_gpu / (PANIC_RADIUS * 2.0), 0.0, 1.0)
    # cohesion / alignment / separation cues in the local frame
    obs_gpu[:, 4] = cp.sum(cohere_dir * f, axis=1)
    obs_gpu[:, 5] = cp.sum(cohere_dir * r, axis=1)
    obs_gpu[:, 6] = cp.sum(cohere_dir * u, axis=1)
    obs_gpu[:, 7] = cp.sum(align_norm * f, axis=1)
    obs_gpu[:, 8] = cp.sum(align_norm * r, axis=1)
    obs_gpu[:, 9] = cp.sum(align_norm * u, axis=1)
    obs_gpu[:, 10] = cp.sum(sep_norm * f, axis=1)
    obs_gpu[:, 11] = cp.sum(sep_norm * r, axis=1)
    obs_gpu[:, 12] = cp.sum(sep_norm * u, axis=1)
    # environment: ground clearance + roost distance
    obs_gpu[:, 13] = cp.clip(clearance_obs / 25.0, -1.0, 1.0)
    obs_gpu[:, 14] = cp.clip(dist_to_roost[:, 0] / ROOST_RADIUS, 0.0, 1.0)
    # --- gravity proprioception: airspeed (0 at the MIN_BIRD_SPEED floor,
    # +1 at the deepest-dive cap) + vertical pitch f_y so tribes can feel
    # the stoop/climb energy exchange and learn to dive for speed.
    obs_gpu[:, 15] = cp.clip((speed_cur[:, 0] - MIN_BIRD_SPEED)
                             / (PANIC_MAX_SPEED * DIVE_SPEED_MULT - MIN_BIRD_SPEED) * 2.0 - 1.0,
                             -1.0, 1.0)
    obs_gpu[:, 16] = cp.clip(f[:, 1], -1.0, 1.0)
    cp.clip(obs_gpu, -1.0, 1.0, out=obs_gpu)

    a = obs_gpu.reshape(NUM_TRIBES, BIRDS_PER_TRIBE, 17)
    for W, b in zip(tribe_Ws, tribe_bs):
        a = cp.tanh(cp.einsum('tbi,tio->tbo', a, W) + b[:, None, :])
    out_gpu = a.reshape(NUM_BIRDS, 6)

    # 6. Integration — NN modulates the boids weights per bird (0.25x..1.75x)
    sep_mult = 1.0 + 0.75 * out_gpu[:, 0:1]
    align_mult = 1.0 + 0.75 * out_gpu[:, 1:2]
    cohere_mult = 1.0 + 0.75 * out_gpu[:, 2:3]
    total_force = (sep_force * FORCE_SEPARATE * sep_mult) + \
                  (align_force * FORCE_ALIGN * align_mult) + \
                  (cohere_force * FORCE_COHERE * cohere_mult) + \
                  (roost_force * FORCE_ROOST) + \
                  (center_force * FORCE_CENTER)

    # Clamp Force
    force_mag = cp.linalg.norm(total_force, axis=1, keepdims=True)
    total_force = cp.where(force_mag > MAX_FORCE, total_force / force_mag * MAX_FORCE, total_force)

    # Update Physics (acceleration scaled by delta time)
    vel_gpu += total_force * dt_scale

    # --- PITCH-GRAVITY ENERGY EXCHANGE (starlings): push velocity ALONG
    # the flight trajectory. Nose down (f_y<0) converts height into speed;
    # nose up bleeds it. Applied along f_cur (not as a raw world -Y pull)
    # so the flock's overall altitude is not dragged below the roost. ---
    speed_cur2 = cp.linalg.norm(vel_gpu, axis=1, keepdims=True) + 1e-6
    f_cur = vel_gpu / speed_cur2
    vel_gpu += f_cur * (-FLOCK_GRAVITY * f_cur[:, 1:2]) * dt_scale
    # slight extra vertical climb resistance: steep climbs need real effort
    vel_gpu[:, 1] -= 0.008 * cp.maximum(f_cur[:, 1], 0.0) * dt_scale

    # --- Dynamic speed caps: diving raises the cruise/panic ceiling toward
    # DIVE_SPEED_MULT (1.35x at straight-down), climbing lowers it toward
    # CLIMB_SPEED_MULT (0.65x at straight-up) — a stooping starling can
    # outrun a level-flying one, a climbing one bails slower. ---
    f_y = f_cur[:, 1:2]
    pitch_mult = cp.where(f_y < 0,
                          1.0 - f_y * (DIVE_SPEED_MULT - 1.0),
                          1.0 - f_y * (1.0 - CLIMB_SPEED_MULT))
    max_spd_dyn = MAX_SPEED * pitch_mult

    # Clamp Speed (dynamic cruise cap)
    speed = cp.linalg.norm(vel_gpu, axis=1, keepdims=True)
    vel_gpu = cp.where(speed > max_spd_dyn, vel_gpu / speed * max_spd_dyn, vel_gpu)

    # 7. PREDATOR PANIC (applied after clamp -> overrides cruise speed).
    # Instinct flee baseline blended with the NN's learned evasive juke
    # (local frame -> world), gated by threat proximity.
    flee_gpu = cp.asarray(flee_np)
    threat_scale = cp.clip(1.0 - nearest_d_gpu / (PANIC_RADIUS * 1.5), 0.0, 1.0)[:, None]
    nn_evade_world = (out_gpu[:, 3:4] * f + out_gpu[:, 4:5] * r + out_gpu[:, 5:6] * u)
    evade_vec = 0.25 * flee_gpu + 0.75 * nn_evade_world * threat_scale
    vel_gpu += evade_vec * FORCE_FLEE * dt_scale
    # clamp raw velocity (dt_scale is applied later, only to position);
    # the panic ceiling also follows the pitch multiplier (dive = faster escape)
    panic_spd_dyn = PANIC_MAX_SPEED * pitch_mult
    speed = cp.linalg.norm(vel_gpu, axis=1, keepdims=True)
    vel_gpu = cp.where(speed > panic_spd_dyn, vel_gpu / speed * panic_spd_dyn, vel_gpu)

    # 8. CAUGHT birds respawn at the roost (orbiting — respawn follows it)
    if caught_np.any():
        caught_gpu = cp.asarray(caught_np)
        respawn = (cp.random.random((NUM_BIRDS, 3)).astype(cp.float32) * 10 - 5) + roost_center
        pos_gpu[caught_gpu] = respawn[caught_gpu]
        rv = cp.random.random((NUM_BIRDS, 3)).astype(cp.float32) - 0.5
        rv /= cp.linalg.norm(rv, axis=1, keepdims=True)
        vel_gpu[caught_gpu] = rv[caught_gpu] * MAX_SPEED
        # Darwinian tribe bookkeeping for the birds eaten this frame
        _tribe_respawn_check(np.where(caught_np)[0])

    # Floor Avoidance (terrain-aware: hills push birds up)
    pos_np2 = to_cpu(pos_gpu)
    ground_np = np.asarray(terrain.height_at(pos_np2[:, 0], pos_np2[:, 2]))
    clearance = pos_gpu[:, 1] - cp.array(ground_np, dtype=cp.float32)
    vel_gpu[:, 1] += cp.where(clearance < 3.0, 0.5, cp.where(clearance < 6.0, 0.15, 0)) * dt_scale

    # --- MIN_BIRD_SPEED floor: a starling never stalls to zero in mid-air
    # (birds keep flying even at the bottom of a climb). Placed AFTER the
    # floor-avoidance boost: that boost can cancel a diving bird's vertical
    # velocity, and re-enforcing here guarantees the final vel_gpu respects
    # the airspeed floor. ---
    speed = cp.linalg.norm(vel_gpu, axis=1, keepdims=True)
    vel_gpu = cp.where(speed < MIN_BIRD_SPEED,
                       vel_gpu / cp.maximum(speed, 1e-6) * MIN_BIRD_SPEED, vel_gpu)

    # Position integration (velocity is per-frame-at-60; dt_scale makes it
    # framerate-independent. Speed clamps above operate on raw vel_gpu.)
    pos_gpu += vel_gpu * dt_scale

    # 8b. Per-bird survival/cohesion score (CPU): surviving is worth 1/s,
    # sticking near the local neighbor centroid adds up to +0.5/s, scraping
    # the ground costs -1/s. Feeds the Moran-process parent choice.
    bird_survival_time += np.float32(time.dt)
    cohere_dist_np = to_cpu(cp.linalg.norm(cohere_force, axis=1))
    cohesion_bonus = np.clip(1.0 - cohere_dist_np / 15.0, 0.0, 1.0)
    clearance_np = to_cpu(clearance)
    bird_score += time.dt * (1.0 + 0.5 * cohesion_bonus - 1.0 * (clearance_np < 2.0))

    # 8c. Tribe casualty counters decay exponentially (framerate-independent:
    # multiplier = exp(-CASUALTY_DECAY * dt), exact for any dt). Old deaths
    # fade so a slow trickle can never execute a healthy top tribe minutes
    # later; a concentrated raid still pushes the counter over the threshold.
    if CASUALTY_DECAY > 0.0:
        tribe_casualties *= np.exp(np.float32(-CASUALTY_DECAY * min(time.dt, 0.05)))

    predators.update_pack_visuals()

# --- 4. VISUALS (Same optimized mesh) ---
class FlockVisuals(Entity):
    def __init__(self):
        super().__init__()
        self.model = Mesh(vertices=[], triangles=[], colors=[], mode='triangle', static=False)
        self.texture = 'white_cube'
        self.color = color.cyan
        self.double_sided = True
        self.bird_shape = np.array([[0, 0, 0.6], [-0.3, 0, -0.3], [0.3, 0, -0.3]], dtype=np.float32)

    def update(self):
        update_boids_gpu()
        positions = to_cpu(pos_gpu)
        velocities = to_cpu(vel_gpu)
        yaws = np.arctan2(velocities[:, 0], velocities[:, 2])
        c, s = np.cos(yaws), np.sin(yaws)
        verts = np.tile(self.bird_shape, (NUM_BIRDS, 1, 1))
        # x/z MUST be copies: they are views into verts, and the second
        # rotation line below would otherwise read the already-overwritten
        # column 0 (a subtle yaw corruption, not a true rotation).
        x, z = verts[:, :, 0].copy(), verts[:, :, 2].copy()
        verts[:, :, 0] = x * c[:, None] + z * s[:, None]
        verts[:, :, 2] = -x * s[:, None] + z * c[:, None]
        verts += positions[:, None, :]
        self.model.vertices = verts.reshape(-1, 3).tolist()
        self.model.generate()
