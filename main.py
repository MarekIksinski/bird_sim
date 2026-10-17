#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 22:43:26 2025

@author: marek
"""
from ursina import *
import numpy as np
import os
import sys
from panda3d.core import Quat as PQuat

# --- 0. GPU/CPU backend selection ---
# Use CuPy (GPU) if there's enough free VRAM; otherwise fall back to NumPy.
cp = None
_FORCE_CPU = False
# --cpu on the command line or FORCE_CPU=1 in the environment bypasses the
# CuPy VRAM probe completely (quick CPU testing without touching config).
_cfg_early = None
try:
    import tomllib as _tl
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml') if '__file__' in dir() else 'config.toml'
    if os.path.exists(_p):
        with open(_p, 'rb') as _fh:
            _cfg_early = _tl.load(_fh)
        _FORCE_CPU = bool(_cfg_early.get('game', {}).get('force_cpu', False))
except Exception:
    pass
if '--cpu' in sys.argv or os.environ.get('FORCE_CPU', '') == '1':
    _FORCE_CPU = True
    sys.argv = [a for a in sys.argv if a != '--cpu']   # keep ursina's arg parsing clean
if not _FORCE_CPU:
    try:
        import cupy as _cupy_probe
        # mem_info is (free, total) — index [0] is the free amount. (An old
        # version read [1] = TOTAL, which made the "> 512 MB free" check
        # almost always true and the log line misleading; the probe
        # allocation below is what actually gated GPU use.)
        _free_mb = _cupy_probe.cuda.Device(0).mem_info[0] // (1024 * 1024)
        # probe by actually allocating ~ what the sim needs — mem_info can be
        # stale/generous while other processes (llama.cpp) fill VRAM on demand
        _probe = _cupy_probe.zeros(64_000_000, dtype=_cupy_probe.float32)  # 256 MB
        _cupy_probe.cuda.Device(0).synchronize()
        del _probe
        _cupy_probe.get_default_memory_pool().free_all_blocks()
        if _free_mb > 512:
            cp = _cupy_probe
            print(f"[backend] CuPy GPU (free VRAM: {_free_mb} MB)")
        else:
            print(f"[backend] NumPy CPU (only {_free_mb} MB VRAM free)")
    except Exception as e:
        print(f"[backend] NumPy CPU (cupy unavailable: {type(e).__name__})")
else:
    print("[backend] NumPy CPU (forced by config: game.force_cpu)")

if cp is None:
    # Shim: pretend numpy is cupy. numpy supports everything the sim uses
    # (argsort/where/linalg/float32). Only .get() is missing (C type, can't
    # be monkey-patched), so all reads go through to_cpu() below. (No fake
    # cp.cuda attribute: `cp = np` makes that an in-place mutation of the
    # shared numpy module, visible to every other importer.)
    cp = np

def to_cpu(arr):
    """cupy array -> numpy via .get(); numpy array passes through."""
    return arr.get() if hasattr(arr, 'get') else arr

# --- 1. CONFIGURATION (Starling Physics) ---
# All tunables live in config.toml next to main.py; defaults below are used
# for anything missing there.
# tomllib is stdlib from 3.11; fall back to the tomli backport, then to
# "no config file" so older Pythons still run with built-in defaults.
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None
        print("[config] tomllib/tomli unavailable (Python < 3.11): using built-in defaults")
import os

def _load_cfg():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    cfg = {}
    if tomllib is not None and os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                cfg = tomllib.load(f)
        except Exception as e:
            print(f"[config] {os.path.basename(path)} failed to parse ({e}); using defaults")
    return cfg

_cfg = _load_cfg()

def _get(section, key, default):
    return _cfg.get(section, {}).get(key, default)

_f = _cfg.get('flock', {})
NUM_BIRDS = _get('flock', 'num_birds', 1000)
NEIGHBOR_COUNT = _get('flock', 'neighbor_count', 7)     # topological "magic number"
SEPARATION_DIST = _get('flock', 'separation_dist', 1.1)
ROOST_RADIUS = _get('flock', 'roost_radius', 55.0)      # flock turns back past this
FORCE_SEPARATE = _get('flock', 'force_separate', 0.7)
FORCE_ALIGN = _get('flock', 'force_align', 0.6)
FORCE_COHERE = _get('flock', 'force_cohere', 0.6)
FORCE_ROOST = _get('flock', 'force_roost', 0.05)
FORCE_CENTER = _get('flock', 'force_center', 0.2)
ROOST_ORBIT_RADIUS = _get('flock', 'roost_orbit_radius', 250.0)  # wide circle inside the map
ROOST_ORBIT_SPEED = _get('flock', 'roost_orbit_speed', 0.05)     # rad/s — slow and majestic
# Secondary wave: off-beat frequencies (x1.37 X, x0.79 Z) so the path never
# closes on itself — the roost traces an infinitely evolving sweep.
ORBIT_WOBBLE_RADIUS = _get('flock', 'orbit_wobble_radius', 0.4)  # fraction of main radius
ORBIT_WOBBLE_SPEED = _get('flock', 'orbit_wobble_speed', 1.0)    # scales the off-beat multipliers
# Vertical murmuration: roost altitude oscillates relative to the ground.
ROOST_BASE_ALTITUDE = _get('flock', 'roost_base_altitude', 25.0)  # m above terrain
VERTICAL_SPEED = _get('flock', 'vertical_speed', 0.02)      # rad/s — slow swoop
VERTICAL_AMPLITUDE = _get('flock', 'vertical_amplitude', 15.0)  # meters up/down
MAX_SPEED = _get('flock', 'max_speed', 2.2)
MAX_FORCE = _get('flock', 'max_force', 0.1)

# --- 1c. FLOCK NN (tribe co-evolution) ---
NUM_TRIBES = int(_get('flock_nn', 'num_tribes', 20))
if NUM_TRIBES < 1 or NUM_BIRDS % NUM_TRIBES != 0:
    # contiguous tribe blocks need even splitting; walk down to a divisor
    while NUM_TRIBES > 1 and NUM_BIRDS % NUM_TRIBES != 0:
        NUM_TRIBES -= 1
    print(f"[flock_nn] num_tribes adjusted to {NUM_TRIBES} so birds split evenly")
BIRDS_PER_TRIBE = NUM_BIRDS // NUM_TRIBES
STARLING_MUTATION_SIGMA = float(_get('flock_nn', 'mutation_sigma', 0.06))
TRIBE_REPLACE_DEATHS = int(_get('flock_nn', 'replace_deaths', 10))
# Casualty counter decay (per second, exponential). Steady-state counter =
# death_rate / CASUALTY_DECAY, so at 0.02/s a tribe only ever crosses the
# replace_deaths=10 threshold if it is dying at ~0.2 birds/s sustained
# (12/min — a real massacre), while a concentrated raid crosses it instantly
# and isolated deaths fade with a ~35s half-life. 0.0 disables decay.
CASUALTY_DECAY = float(_get('flock_nn', 'casualty_decay', 0.02))

# --- 1b. PREDATOR PACK (neuroevolution) ---
PRED_BOOST = _get('hawk', 'boost_speed', 4.6)           # throttle = +1
CRUISE_SPEED = _get('hawk', 'cruise_speed', 2.4)        # throttle = 0
SLOW_FACTOR = _get('hawk', 'slow_factor', 0.45)         # throttle = -1 -> cruise*this
YAW_RATE = _get('hawk', 'yaw_rate', 90.0)               # deg/s at output = +-1
PITCH_RATE = _get('hawk', 'pitch_rate', 60.0)           # deg/s at output = +-1
ROLL_SPEED = _get('hawk', 'roll_speed', 120.0)          # deg/s at output = +-1
PANIC_RADIUS = _get('hawk_vs_flock', 'panic_radius', 12.0)
FORCE_FLEE = _get('hawk_vs_flock', 'force_flee', 2.0)
PANIC_MAX_SPEED = _get('hawk_vs_flock', 'panic_max_speed', 3.4)
CATCH_RADIUS = _get('hawk_vs_flock', 'catch_radius', 1.2)
MAX_ALTITUDE = _get('hawk', 'max_altitude', 80.0)  # ceiling: keep map edges hidden

POPULATION_SIZE = int(_get('pack', 'population_size', 10))
GENERATION_SECONDS = float(_get('pack', 'generation_seconds', 30.0))
# 3-tiered speciation: each of the top-3 lineages gets its own mutation
# strength — exploitation (tiny) around the champion, exploration (heavy)
# around the wildcard. Replaces the old flat mutation_sigma.
MUTATION_TINY = float(_get('pack', 'mutation_tiny', 0.02))
MUTATION_MEDIUM = float(_get('pack', 'mutation_medium', 0.08))
MUTATION_HEAVY = float(_get('pack', 'mutation_heavy', 0.20))
if POPULATION_SIZE != 10:
    print(f"[pack] WARNING: tiered speciation hardcodes a 10-slot layout; "
          f"population_size = {POPULATION_SIZE} will break evolve_generation. "
          f"Forcing 10.")
    POPULATION_SIZE = 10
FITNESS_CATCH = float(_get('pack', 'fitness_catch', 1000.0))
PROXIMITY_RADIUS = float(_get('pack', 'proximity_radius', 150.0))
PROXIMITY_GAIN = float(_get('pack', 'proximity_gain', 3.0))
GROUND_PENALTY = float(_get('pack', 'ground_penalty', 2.0))
GROUND_CLEARANCE = float(_get('pack', 'ground_clearance', 1.2))
DOGFIGHT_RADIUS = float(_get('pack', 'dogfight_radius', 25.0))
DOGFIGHT_GAIN = float(_get('pack', 'dogfight_gain', 2.0))
BOUNDARY_PENALTY = float(_get('pack', 'boundary_penalty', 0.3))
BOUNDARY_SCRAPE = float(_get('pack', 'boundary_scrape_dist', 5.0))
PERSIST = bool(_get('pack', 'persist', True))
EVOLVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          _get('pack', 'save_folder', 'evolution'))

app = Ursina(vsync=False)

# --- 2. ENVIRONMENT ---
Sky(texture='sky_sunset')
scene.fog_color = color.rgb(20, 30, 20)
# This Ursina build ignores SCALAR fog_density (the exp branch is commented
# out in scene.py) — convert the config density to the linear (start,end)
# the shader actually uses: exp density d ~ fully fogs at ~3/d meters.
# NOTE: the fog_density setter only re-shaders entities that EXIST at call
# time (scene.py loops self.entities) — it is re-applied after all entities
# are built, just before app.run().
_fd = _get('terrain', 'fog_density', 0.0035)
_FOG_RANGE = (50, int(3.0 / _fd)) if 0 < _fd < 1 else ((50, 450) if _fd >= 1 else (10_000, 10_001))
scene.fog_density = _FOG_RANGE

# --- 2a. PROCEDURAL TERRAIN (value-noise fBm heightmap) ---
TERRAIN_SIZE = _get('terrain', 'size', 1000)        # field width/depth; RES auto-scales
TERRAIN_CELL = _get('terrain', 'cell', 1.2)         # meters per grid cell (lower = more detail)
TERRAIN_RES = int(TERRAIN_SIZE / TERRAIN_CELL)      # grid cells per side
TERRAIN_AMP = _get('terrain', 'amplitude', 10.0)    # max hill height
TERRAIN_SEED = _get('terrain', 'seed', 7)           # change for a different world
ROOST_CLEAR_RADIUS = _get('terrain', 'roost_clear_radius', 30.0)  # flat bowl for the flock

# Backdrop mountains — giant low-poly ring around the playable field.
# Visual only: never sampled by physics (height_at still uses `heightmap`).
BACKDROP_SIZE = 4000
BACKDROP_AMP = 180.0
BACKDROP_RES = 64   # tiny grid: cheap to render, deliberately coarse

def _smooth(t):
    return t * t * (3 - 2 * t)

def _make_heightmap(seed, res, size, amp):
    rng = np.random.default_rng(seed)
    uv = np.linspace(-1, 1, res + 1)
    xs, zs = np.meshgrid(uv * size / 2, uv * size / 2)
    h = np.zeros((res + 1, res + 1), dtype=np.float64)
    gain = 1.0; total = 0.0
    scale = size / 150.0   # keep hills the same physical size on bigger fields
    # Cap octaves so the noise lattice never far-exceeds the output grid:
    # finer octaves than the grid only alias, and the last lattice would be
    # gigabytes for the big backdrop (13k x 13k doubles).
    octaves = min(5 + int(np.log2(max(scale, 1))), int(np.log2(res)))
    for octave in range(octaves):
        cells = int((2 ** (octave + 1)) * scale) + 1   # lattice spanning the field
        g = rng.random((cells, cells))
        fx = (xs / size + 0.5) * (cells - 1)
        fz = (zs / size + 0.5) * (cells - 1)
        i = np.clip(np.floor(fx).astype(int), 0, cells - 2)
        j = np.clip(np.floor(fz).astype(int), 0, cells - 2)
        tx = _smooth(fx - i); tz = _smooth(fz - j)
        n = (g[i, j] * (1 - tx) * (1 - tz) + g[i + 1, j] * tx * (1 - tz) +
             g[i, j + 1] * (1 - tx) * tz + g[i + 1, j + 1] * tx * tz)
        h += n * gain
        total += gain; gain *= 0.5
    h = (h / total - 0.5) * 2 * amp
    # flatten the roost bowl so the flock keeps an open arena
    r = np.sqrt(xs**2 + zs**2)
    k = np.clip((r - ROOST_CLEAR_RADIUS) / 25.0, 0, 1)
    h *= _smooth(k)
    h = np.maximum(h, -1.0)  # valleys don't go under the waterline
    return h.astype(np.float32)

heightmap = _make_heightmap(TERRAIN_SEED, TERRAIN_RES, TERRAIN_SIZE, TERRAIN_AMP)
_cell = TERRAIN_SIZE / TERRAIN_RES

def height_at(x, z):
    """Bilinear sample of the heightmap (vectorized, used by physics too)."""
    u = (np.asarray(x, dtype=np.float64) + TERRAIN_SIZE / 2) / _cell
    v = (np.asarray(z, dtype=np.float64) + TERRAIN_SIZE / 2) / _cell
    u = np.clip(u, 0, TERRAIN_RES - 1e-6); v = np.clip(v, 0, TERRAIN_RES - 1e-6)
    i = u.astype(int); j = v.astype(int)
    tu = u - i; tv = v - j
    hm = heightmap
    return (hm[j, i] * (1 - tu) * (1 - tv) + hm[j, i + 1] * tu * (1 - tv) +
            hm[j + 1, i] * (1 - tu) * tv + hm[j + 1, i + 1] * tu * tv)

def build_terrain_mesh():
    verts = np.zeros((TERRAIN_RES + 1, TERRAIN_RES + 1, 3), dtype=np.float32)
    lin = np.linspace(-TERRAIN_SIZE / 2, TERRAIN_SIZE / 2, TERRAIN_RES + 1)
    verts[:, :, 0] = lin[None, :]
    verts[:, :, 1] = heightmap
    verts[:, :, 2] = lin[:, None]
    tri = []
    for j in range(TERRAIN_RES):
        for i in range(TERRAIN_RES):
            a = j * (TERRAIN_RES + 1) + i
            b = a + 1
            c = a + (TERRAIN_RES + 1)
            d = c + 1
            tri += [a, c, b, b, c, d]
    # flat-shaded vertex colors from height (low-poly look)
    cols = np.zeros((TERRAIN_RES + 1, TERRAIN_RES + 1, 4), dtype=np.float32)
    h = heightmap
    grass = np.array([0.10, 0.16, 0.09]); high = np.array([0.32, 0.33, 0.30])
    t = np.clip((h + 1) / (TERRAIN_AMP + 1), 0, 1)[:, :, None]
    rgb = grass * (1 - t) + high * t
    # darken steep slopes for definition
    dh = np.abs(np.gradient(h, axis=0)) + np.abs(np.gradient(h, axis=1))
    shade = (1 - np.clip(dh / 1.2, 0, 0.45))[:, :, None]
    cols[:, :, :3] = rgb * shade
    cols[:, :, 3] = 1.0
    m = Mesh(vertices=verts.reshape(-1, 3).tolist(),
             triangles=tri,
             colors=cols.reshape(-1, 4).tolist(),
             mode='triangle', static=True)
    m.generate()
    m.generate_normals(smooth=False)
    return m

terrain = Entity(model=build_terrain_mesh(), texture='white_cube')
terrain.double_sided = True  # no backface culling; hills never vanish at grazing angles

# --- 2b. BACKDROP MOUNTAINS (visual shell, no physics) ---
backdrop_hm = _make_heightmap(TERRAIN_SEED + 1, BACKDROP_RES, BACKDROP_SIZE, BACKDROP_AMP)
# Feather the center instead of a hard basin: the backdrop continues the PLAY
# surface ~3m below it out to the map edge (r=600), then blends up into the
# mountain ring over the next 100m (just outside the invisible walls at ~599).
# Overlap only exists outside the walls -> pure visual blend, no gameplay hit.
_blin = np.linspace(-BACKDROP_SIZE / 2, BACKDROP_SIZE / 2, BACKDROP_RES + 1)
_bx, _bz = np.meshgrid(_blin, _blin)
_br = np.sqrt(_bx**2 + _bz**2)
_play_floor = height_at(_bx, _bz) - 3.0   # height_at clamps coords: extends the edge
_blend = _smooth(np.clip((_br - TERRAIN_SIZE / 2) / 100.0, 0, 1))
backdrop_hm = _play_floor * (1 - _blend) + backdrop_hm * _blend

def build_backdrop_mesh():
    verts = np.zeros((BACKDROP_RES + 1, BACKDROP_RES + 1, 3), dtype=np.float32)
    lin = _blin
    verts[:, :, 0] = lin[None, :]
    verts[:, :, 1] = backdrop_hm
    verts[:, :, 2] = lin[:, None]
    tri = []
    for j in range(BACKDROP_RES):
        for i in range(BACKDROP_RES):
            a = j * (BACKDROP_RES + 1) + i
            b = a + 1
            c = a + (BACKDROP_RES + 1)
            d = c + 1
            tri += [a, c, b, b, c, d]
    cols = np.zeros((BACKDROP_RES + 1, BACKDROP_RES + 1, 4), dtype=np.float32)
    h = backdrop_hm
    rock = np.array([0.12, 0.14, 0.18]); snow = np.array([0.45, 0.48, 0.55])
    t = np.clip((h + 50) / (BACKDROP_AMP + 50), 0, 1)[:, :, None]
    rgb = rock * (1 - t) + snow * t
    # fake atmospheric scattering: darker + bluer the higher/farther
    cols[:, :, :3] = rgb * 0.6
    cols[:, :, 3] = 1.0
    m = Mesh(vertices=verts.reshape(-1, 3).tolist(),
             triangles=tri,
             colors=cols.reshape(-1, 4).tolist(),
             mode='triangle', static=True)
    m.generate()
    m.generate_normals(smooth=False)
    return m

backdrop = Entity(model=build_backdrop_mesh(), texture='white_cube')
backdrop.double_sided = True
# much thinner fog than the playfield so the range reads as a hazy silhouette
backdrop.set_shader_input('fog_start', 800)
backdrop.set_shader_input('fog_end', 5000)

# Roost Marker (Invisible center point)
roost_center = cp.array([0, ROOST_BASE_ALTITUDE, 0], dtype=cp.float32)

# Trees — planted ON the terrain, outside the roost bowl
for i in range(60):
    x, z = random.uniform(-TERRAIN_SIZE/2 + 10, TERRAIN_SIZE/2 - 10), random.uniform(-TERRAIN_SIZE/2 + 10, TERRAIN_SIZE/2 - 10)
    if (x*x + z*z) ** 0.5 < ROOST_CLEAR_RADIUS + 5:
        continue
    h = float(height_at(x, z))
    e = Entity(model='cube', color=color.dark_gray, scale=(1, random.uniform(5, 15), 1),
               position=(x, h, z))
    e.y += e.scale_y / 2

# --- 3. GPU PHYSICS (Topological) ---
# dtype=float32 is REQUIRED: an untyped cp.array([0, alt, 0]) is int-float64
# mixed -> the sum promotes pos_gpu to float64, and the whole N x N distance/
# argsort/neighbor chain then runs in double precision (2x the memory: 72MB
# vs 36MB per matrix at 3000 birds, and far slower on consumer GPUs).
pos_gpu = (cp.random.random((NUM_BIRDS, 3)).astype(cp.float32) * 40 - 20) + cp.array([0, ROOST_BASE_ALTITUDE, 0], dtype=cp.float32)
vel_gpu = cp.random.random((NUM_BIRDS, 3)).astype(cp.float32) - 0.5
vel_gpu = vel_gpu / cp.linalg.norm(vel_gpu, axis=1, keepdims=True) * MAX_SPEED

# --- 3a. FLOCK NEURAL NETS (tribe genomes, co-evolving vs the hawk pack) ---
# One MLP per tribe (NUM_TRIBES small nets, NOT one per bird): birds are
# CONTIGUOUS blocks (bird i -> tribe i // BIRDS_PER_TRIBE) so the forward
# pass reshapes to (T, BIRDS_PER_TRIBE, n) and einsums against tiny
# (T, ni, no) weights — fast on CPU cache AND on GPU, identical code paths.
STARLING_NET_SHAPE = (15, 16, 8, 6)
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

def _q_mul_b(a, b):
    """Batched quaternion multiply, [x,y,z,w] rows, (N,4) x (N,4) -> (N,4)."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz], axis=1)

def _q_from_aa_b(axes, deg):
    """Batched axis-angle quats: axes (N,3) unit-ish, deg (N,) -> (N,4) [x,y,z,w]."""
    r = np.radians(deg) * 0.5
    s = np.sin(r)[:, None]; c = np.cos(r)[:, None]
    n = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / np.maximum(n, 1e-9)
    return np.concatenate([axes * s, c], axis=1)

def _q_rot_b(q, v):
    """Batched rotate vectors v (N,3) by quaternions q (N,4)."""
    u = q[:, :3]; w = q[:, 3:4]
    uv = np.cross(u, v); uuv = np.cross(u, uv)
    return v + 2 * (w * uv + uuv)

def _slerp_np(a, b, t):
    """Shortest-arc slerp of unit quats [x,y,z,w]. Falls back to nlerp when
    the arc is tiny (sin(theta)~0) and renormalizes degenerate inputs."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    if na < 1e-9:
        return b / max(np.linalg.norm(b), 1e-9)
    a = a / na
    if np.dot(a, b) < 0.0:
        b = -b                          # shortest arc: never a 180deg snap
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 0.9995:                      # near-identical: lerp+renorm is stable
        q = a + (b - a) * t
    else:
        th = np.arccos(c); s = np.sin(th)
        q = a * np.sin((1 - t) * th) / s + b * np.sin(t * th) / s
    return q / max(np.linalg.norm(q), 1e-9)

def pack_axes():
    """local forward/right/up per hawk: each (N,3)."""
    f = _q_rot_b(pack_q, np.array([0.0, 0.0, 1.0]))   # nose
    r = _q_rot_b(pack_q, np.array([1.0, 0.0, 0.0]))   # right wing
    u = _q_rot_b(pack_q, np.array([0.0, 1.0, 0.0]))   # dorsal
    return f, r, u

def respawn_pack():
    """Respawn all hawks around the map edge, spread out, facing the roost."""
    global gen_elapsed, pack_q, pack_pos, pack_vel, pack_speed
    ang = np.linspace(0, 2 * np.pi, POPULATION_SIZE, endpoint=False) + pack_rng.uniform(-0.3, 0.3)
    rad = TERRAIN_SIZE * 0.42
    px = np.cos(ang) * rad; pz = np.sin(ang) * rad
    rc = to_cpu(roost_center)
    pack_pos[:, 0] = px
    pack_pos[:, 2] = pz
    pack_pos[:, 1] = height_at(px, pz) + 3.0
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

# --- persistence: keep the evolved pack between runs (folder from config) ---
_CKPT = os.path.join(EVOLVE_DIR, 'population.npz')
_HISTORY = os.path.join(EVOLVE_DIR, 'history.csv')
_STARLING_CKPT = os.path.join(EVOLVE_DIR, 'starling_tribes.npz')

def load_checkpoint():
    """Restore population/generation counters from the checkpoint if it
    matches the current config (population_size, genome length)."""
    global population, generation, total_catches
    global tribe_pop, starling_evolutions
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
        print(f"[persist] resumed at gen {generation} from {_CKPT}")
    except Exception as e:
        print(f"[persist] could not load checkpoint ({e}); starting fresh")
    # starling tribe genomes (separate file, own shape guard)
    if os.path.exists(_STARLING_CKPT):
        try:
            s = np.load(_STARLING_CKPT)
            tp = s['tribe_pop']
            if tp.shape != (NUM_TRIBES, STARLING_GENOME_LEN):
                print(f"[persist] starling checkpoint shape {tp.shape} != config "
                      f"({NUM_TRIBES},{STARLING_GENOME_LEN}); tribes start fresh")
                return
            tribe_pop = tp.astype(np.float32)
            starling_evolutions = int(s['starling_evolutions'])
            print(f"[persist] {starling_evolutions} starling tribe evolutions resumed")
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
                 total_catches=total_catches)
        np.savez(_STARLING_CKPT, tribe_pop=tribe_pop,
                 starling_evolutions=starling_evolutions)
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

load_checkpoint()
# a loaded starling checkpoint replaces tribe_pop -> re-upload cp layer tensors
_upload_tribe_layers()

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
    d2 = np.empty((POPULATION_SIZE, NUM_BIRDS), dtype=np.float32)
    step = 1024
    for s in range(0, NUM_BIRDS, step):
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
    clearance = pack_pos[:, 1] - height_at(pack_pos[:, 0], pack_pos[:, 2])
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
    ground = height_at(pack_pos[:, 0], pack_pos[:, 2])
    pack_pos[:, 1] = np.minimum(MAX_ALTITUDE,
                                np.maximum(pack_pos[:, 1], ground + 0.4))
    if not np.all(np.isfinite(pack_pos)):
        bad = np.where(~np.isfinite(pack_pos).all(axis=1))[0]
        print(f"[nan-guard] hawks {bad} went non-finite; q={pack_q[bad[0]]} spd={pack_speed[bad[0]]}")
        respawn_pack()

def evolve_generation():
    """Rank -> 3-tiered speciation: each top-3 lineage keeps an untouched
    clone plus offspring at a lineage-specific mutation strength.
    Tier 1 (champion, tiny): fine-tune the winner. Tier 2 (runner-up,
    medium): explore around second place. Tier 3 (wildcard, heavy):
    high-variance lottery tickets. Hardcoded 10-slot layout."""
    global population, fitness, gen_kills, generation, gen_elapsed, sel_idx
    order = np.argsort(-fitness)
    print(f"[evolution] gen {generation}: best fitness {fitness[order[0]]:.0f}, "
          f"kills best {gen_kills[order[0]]}, total {gen_kills.sum()}")
    # history row FIRST: fitness/gen_kills still hold THIS generation's
    # results (and `generation` is still the completed gen's number).
    append_history()
    new_pop = np.empty_like(population)
    # --- Tier 1: champion lineage (exploitation) ---
    new_pop[0] = population[order[0]]                       # untouched champion
    for i in (1, 2, 3):
        new_pop[i] = population[order[0]] + pack_rng.normal(
            0.0, MUTATION_TINY, size=GENOME_LEN)
    # --- Tier 2: runner-up lineage (medium exploration) ---
    new_pop[4] = population[order[1]]                       # untouched runner-up
    for i in (5, 6):
        new_pop[i] = population[order[1]] + pack_rng.normal(
            0.0, MUTATION_MEDIUM, size=GENOME_LEN)
    # --- Tier 3: wildcard lineage (heavy exploration) ---
    new_pop[7] = population[order[2]]                       # untouched wildcard
    for i in (8, 9):
        new_pop[i] = population[order[2]] + pack_rng.normal(
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
    clearance = pack_pos[:, 1] - height_at(pack_pos[:, 0], pack_pos[:, 2])
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

# --- 3d. GROUND PREY: RABBITS (pure NumPy 4-state machine) ---
# Forage(0) / Vigilance(1) / Freeze(2) / Flee(3). They react to the hawk
# pack but do not affect hawk fitness yet. Burrows are INVISIBLE: physics
# only, nothing rendered.
NUM_BURROWS = int(_get('rabbits', 'num_burrows', 20))
NUM_RABBITS = int(_get('rabbits', 'num_rabbits', 100))
RABBIT_FLEE_RADIUS = float(_get('rabbits', 'flee_radius', 25.0))      # hawk closer -> Flee
RABBIT_FREEZE_RADIUS = float(_get('rabbits', 'freeze_radius', 50.0))  # band between -> Freeze
RABBIT_RECOVER_RADIUS = float(_get('rabbits', 'recover_radius', 30.0))  # hawk farther -> end Flee
RABBIT_FORAGE_SPEED = float(_get('rabbits', 'forage_speed', 0.8))     # m/s (true SI, dt-scaled)
RABBIT_FLEE_SPEED = float(_get('rabbits', 'flee_speed', 6.0))         # m/s sprint
RABBIT_ESCAPE_RADIUS = float(_get('rabbits', 'escape_radius', 1.0))   # dive into burrow below this

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
rabbit_pos[:, 1] = height_at(_sx, _sz)

def update_rabbits():
    """One frame for the whole rabbit population: threat triggers -> state
    machine -> vectorized movement -> integration. Pure NumPy."""
    global rabbit_pos, rabbit_vel, rabbit_state, rabbit_timer
    # same dt-cap policy as the rest of the sim (no giant teleport on a hitch)
    dt = min(max(time.dt, 0.0), 0.05)

    # distance from each rabbit to the nearest hawk in the pack
    d = np.linalg.norm(rabbit_pos[:, None, :] - pack_pos[None, :, :], axis=2).min(axis=1)

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
            rabbit_pos[ei, 1] = height_at(new_b[:, 0], new_b[:, 1])
            rabbit_vel[ei] = 0.0
            rabbit_state[ei] = 2           # frozen in the burrow mouth
            rabbit_timer[ei] = 2.0

    # --- integration (true m/s: plain time.dt, no per-frame-60 scaling) ---
    rabbit_pos[:, [0, 2]] += rabbit_vel[:, [0, 2]] * dt
    rabbit_pos[:, 0] = np.clip(rabbit_pos[:, 0], -_rab_edge, _rab_edge)
    rabbit_pos[:, 2] = np.clip(rabbit_pos[:, 2], -_rab_edge, _rab_edge)
    rabbit_pos[:, 1] = height_at(rabbit_pos[:, 0], rabbit_pos[:, 2])

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

flock = FlockVisuals()

# --- 4b. RABBIT VISUALS (batched mesh, mirrors FlockVisuals) ---
class RabbitVisuals(Entity):
    def __init__(self):
        super().__init__()
        self.model = Mesh(vertices=[], triangles=[], colors=[], mode='triangle', static=False)
        self.texture = 'white_cube'
        self.color = color.rgb(140, 100, 70)   # small brown rabbits
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

rabbits = RabbitVisuals()

# Hop phase accumulator: time.time() is epoch-scale (~1.8e9) and
# time.time() * speed is computed in float32, whose ulp there is 128 —
# the sine argument stays frozen for minutes. Integrate phase per-rabbit
# in float32 instead (small accumulating values), which also keeps the
# gait continuous when speed changes instead of re-phasing on a jump.
rabbit_phase = np.zeros(NUM_RABBITS, dtype=np.float32)

# The hawk pack — small colored wedges (stretched cubes read as darts)
pack_entities = []
for i in range(POPULATION_SIZE):
    e = Entity(model='assets/hawk.glb', color=color.rgb(150, 70, 50))
    pack_entities.append(e)
GOLD = color.rgb(255, 200, 40)
MUTED = color.rgb(150, 70, 50)
BRIGHT = color.rgb(230, 50, 40)

hawk_info = Text('1/2/3 cameras  TAB cycle hawk  ESC quit  —  hawks evolve every 30s',
                 origin=(0, 0), y=-0.45, size=0.035, color=color.rgba(255, 255, 255, 120))
gen_text = Text('Gen: 1 | Time: 0.0s | Best Kills: 0', origin=(0, 0), y=-0.38, size=0.045,
                color=color.rgba(255, 220, 120, 200))
kill_counter = Text('catches: 0', origin=(0, 0), x=-0.9, y=-0.45, size=0.05, color=color.red)

# --- 5. CAMERA ---
# config start_camera: 1 fixed wide, 2 pack chase (best hawk), 3 first-person (selected)
curr_mode = max(0, min(2, int(_get('game', 'start_camera', 3)) - 1))
# Dynamic FOV warp: widens toward BOOST_FOV as the tracked hawk approaches boost speed.
BASE_FOV = _get('hawk', 'base_fov', 90.0)
BOOST_FOV = _get('hawk', 'boost_fov', 110.0)

def input(key):
    global curr_mode, sel_idx
    if key == '1':
        camera.parent = scene
        camera.position = (0, 30, -50)
        camera.look_at((0,0,0))
        curr_mode = 0
    if key == '2': curr_mode = 1
    if key == '3': curr_mode = 2
    if key == 'tab': sel_idx = (sel_idx + 1) % POPULATION_SIZE
    if key == 'escape': application.base.userExit()

def update():
    global _last_gen_line
    # Ursina's dt can spike huge after a heavy frame (CPU-fallback boids step)
    # — clamp it for ALL camera math so lerp can't extrapolate to inf.
    cdt = min(max(time.dt, 0.0), 0.1)
    if not np.all(np.isfinite([camera.position.x, camera.position.y, camera.position.z])):
        camera.position = Vec3(0, 35, -45)   # recover a poisoned camera
    # hide the selected hawk body in first-person
    for i, e in enumerate(pack_entities):
        e.visible = not (curr_mode == 2 and i == sel_idx)
    # --- speed FOV warp on the tracked hawk ---
    track = sel_idx if curr_mode == 2 else best_idx
    speed_ratio = max(0.0, float((pack_speed[track] - CRUISE_SPEED) / max(PRED_BOOST - CRUISE_SPEED, 1e-6)))
    target_fov = BASE_FOV + (BOOST_FOV - BASE_FOV) * speed_ratio
    camera.fov = lerp(float(camera.fov), float(target_fov), cdt * 4.0)
    if curr_mode == 1:
        # pack chase: follow the current generation's best genome.
        # The offset lives in the HAWK's body frame (behind -f, above +u) and
        # the camera quaternion SLERPs toward the hawk's own attitude, so the
        # camera rides through vertical climbs, loops and rolls. No look_at:
        # its up-vector singularity used to freeze the view (then snap) on a
        # steeply-climbing hawk.
        target_pos = pack_pos[best_idx]
        q_best = pack_q[best_idx]
        off_b = np.array([0.0, 2.5, -9.0])   # body frame (r, u, f): above+behind
        off = _q_rot_b(q_best[None, :], off_b[None, :])[0]
        wanted_pos = Vec3(*(target_pos + off))
        camera.position = lerp(camera.position, wanted_pos, min(cdt * 6, 1.0))
        # keep the chase camera out of the terrain: a hawk hugging a hill puts
        # the lerped camera inside the mesh -> whole frame becomes flat rock.
        cam_ground = float(height_at(camera.position.x, camera.position.z))
        if camera.position.y < cam_ground + 2.5:
            camera.position.y = cam_ground + 2.5
        # PQuat/LQuaternionf index as [0]=w(r), [1]=x(i), [2]=y(j), [3]=z(k)
        # (verified against getR/getI/getJ/getK) — _slerp_np uses (x,y,z,w),
        # so read with a [1,2,3,0] rotation and write back PQuat(w,x,y,z).
        q_tgt = np.array([q_best[0], q_best[1], q_best[2], q_best[3]])  # pack_q rows are (x,y,z,w)
        cq = camera.quaternion
        q_cur = np.array([cq[1], cq[2], cq[3], cq[0]])                  # (w,x,y,z) -> (x,y,z,w)
        q_out = _slerp_np(q_cur, q_tgt, min(cdt * 4.0, 1.0))
        camera.quaternion = PQuat(q_out[3], q_out[0], q_out[1], q_out[2])
    elif curr_mode == 2:
        # first-person on the TAB-selected hawk
        q = pack_q[sel_idx]
        u = _q_rot_b(q[None, :], np.array([0.0, 1.0, 0.0]))[0]
        camera.position = Vec3(*(pack_pos[sel_idx] + u * 0.3))
        camera.quaternion = PQuat(q[3], q[0], q[1], q[2])
    # HUD
    _line = (f'Gen: {generation} | Time: {GENERATION_SECONDS - gen_elapsed:.1f}s | '
             f'Best Kills: {int(gen_kills[best_idx])} | Flock Evo: {starling_evolutions}')
    if _line != _last_gen_line:
        _last_gen_line = _line
        gen_text.text = _line
    if kill_counter.text != f'catches: {total_catches}':
        kill_counter.text = f'catches: {total_catches}'
_last_gen_line = ''
camera.position = (0, 35, -45)

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
    global pos_gpu, vel_gpu, roost_center, gen_elapsed
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
    base_altitude = float(height_at(new_x, new_z)) + ROOST_BASE_ALTITUDE
    new_y = base_altitude + np.sin(time.time() * VERTICAL_SPEED) * VERTICAL_AMPLITUDE
    roost_center = cp.array([new_x, new_y, new_z], dtype=cp.float32)

    # read flock state once per frame (pack logic needs numpy; boids stay on backend)
    pos_np = to_cpu(pos_gpu)
    vel_np = to_cpu(vel_gpu)

    # --- predator pack step BEFORE the boids rules: panic/catch results feed in ---
    gen_elapsed += time.dt
    if gen_elapsed >= GENERATION_SECONDS:
        evolve_generation()
        pos_np = to_cpu(pos_gpu)  # respawns may have moved birds last frame; cheap refresh
    flee_np, caught_np, caught_by, nearest_dir_np, nearest_d_np = update_pack(pos_np, vel_np)

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
    ground_np_obs = np.asarray(height_at(pos_np[:, 0], pos_np[:, 2]))
    clearance_obs = pos_gpu[:, 1] - cp.array(ground_np_obs, dtype=cp.float32)

    cohere_dir = cohere_force / (cp.linalg.norm(cohere_force, axis=1, keepdims=True) + 1e-6)
    align_norm = avg_vel / MAX_SPEED
    sep_norm = sep_force / (cp.linalg.norm(sep_force, axis=1, keepdims=True) + 1.0)

    obs_gpu = cp.empty((NUM_BIRDS, 15), dtype=cp.float32)
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
    cp.clip(obs_gpu, -1.0, 1.0, out=obs_gpu)

    a = obs_gpu.reshape(NUM_TRIBES, BIRDS_PER_TRIBE, 15)
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

    # Clamp Speed
    speed = cp.linalg.norm(vel_gpu, axis=1, keepdims=True)
    vel_gpu = cp.where(speed > MAX_SPEED, vel_gpu / speed * MAX_SPEED, vel_gpu)

    # 7. PREDATOR PANIC (applied after clamp -> overrides cruise speed).
    # Instinct flee baseline blended with the NN's learned evasive juke
    # (local frame -> world), gated by threat proximity.
    flee_gpu = cp.asarray(flee_np)
    threat_scale = cp.clip(1.0 - nearest_d_gpu / (PANIC_RADIUS * 1.5), 0.0, 1.0)[:, None]
    nn_evade_world = (out_gpu[:, 3:4] * f + out_gpu[:, 4:5] * r + out_gpu[:, 5:6] * u)
    evade_vec = 0.25 * flee_gpu + 0.75 * nn_evade_world * threat_scale
    vel_gpu += evade_vec * FORCE_FLEE * dt_scale
    # clamp raw velocity (dt_scale is applied later, only to position)
    speed = cp.linalg.norm(vel_gpu, axis=1, keepdims=True)
    vel_gpu = cp.where(speed > PANIC_MAX_SPEED, vel_gpu / speed * PANIC_MAX_SPEED, vel_gpu)

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
    ground_np = np.asarray(height_at(pos_np2[:, 0], pos_np2[:, 2]))
    clearance = pos_gpu[:, 1] - cp.array(ground_np, dtype=cp.float32)
    vel_gpu[:, 1] += cp.where(clearance < 3.0, 0.5, cp.where(clearance < 6.0, 0.15, 0)) * dt_scale

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

    # --- hawk visuals + rank colors (top performer gold, runner-up bright) ---
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

# initial drop: hawks launch from the edge facing the roost
respawn_pack()

# Re-apply fog now that EVERY entity exists (setter only touches existing
# entities), then give the backdrop its thinner haze so the ring stays visible.
scene.fog_density = _FOG_RANGE
backdrop.set_shader_input('fog_start', 800)
backdrop.set_shader_input('fog_end', 5000)
app.run()
#EditorCamera() 
#app.run()
