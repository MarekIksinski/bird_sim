# -*- coding: utf-8 -*-
"""
config.py — configuration constants (layer 2: depends only on backend/stdlib).

All tunables live in config.toml next to main.py; defaults below are used
for anything missing there. (The --cpu/FORCE_CPU argv stripping lives in
backend.py, which main.py imports before Ursina is touched.)
"""
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

# --- terrain constants (used by terrain.py + physics) ---
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

# Fog: this Ursina build ignores SCALAR fog_density (the exp branch is
# commented out in scene.py) — convert the config density to the linear
# (start,end) the shader actually uses: exp density d ~ fully fogs at ~3/d
# meters. NOTE: the fog_density setter only re-shaders entities that EXIST
# at call time (scene.py loops self.entities) — it is re-applied in main.py
# after all entities are built, just before app.run().
_fd = _get('terrain', 'fog_density', 0.0035)
_FOG_RANGE = (50, int(3.0 / _fd)) if 0 < _fd < 1 else ((50, 450) if _fd >= 1 else (10_000, 10_001))

# --- camera/HUD constants ---
# config start_camera: 1 fixed wide, 2 pack chase (best hawk), 3 first-person (selected)
START_CAMERA = int(_get('game', 'start_camera', 3))
# Dynamic FOV warp: widens toward BOOST_FOV as the tracked hawk approaches boost speed.
BASE_FOV = _get('hawk', 'base_fov', 90.0)
BOOST_FOV = _get('hawk', 'boost_fov', 110.0)

# --- rabbit constants ---
NUM_BURROWS = int(_get('rabbits', 'num_burrows', 20))
NUM_RABBITS = int(_get('rabbits', 'num_rabbits', 100))
RABBIT_FLEE_RADIUS = float(_get('rabbits', 'flee_radius', 25.0))      # hawk closer -> Flee
RABBIT_FREEZE_RADIUS = float(_get('rabbits', 'freeze_radius', 50.0))  # band between -> Freeze
RABBIT_RECOVER_RADIUS = float(_get('rabbits', 'recover_radius', 30.0))  # hawk farther -> end Flee
RABBIT_FORAGE_SPEED = float(_get('rabbits', 'forage_speed', 0.8))     # m/s (true SI, dt-scaled)
RABBIT_FLEE_SPEED = float(_get('rabbits', 'flee_speed', 6.0))         # m/s sprint
RABBIT_ESCAPE_RADIUS = float(_get('rabbits', 'escape_radius', 1.0))   # dive into burrow below this
