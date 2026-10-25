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
# --- starling pitch-gravity energy exchange (per-frame-at-60 units) ---
# Diving birds accelerate along their path and get a HIGHER dynamic speed
# cap; climbing birds bleed speed and hit a LOWER cap; no bird ever stalls
# to zero in mid-air (MIN_BIRD_SPEED floor).
FLOCK_GRAVITY = float(_get('flock', 'gravity', 0.035))       # pitch acceleration along f_y
MIN_BIRD_SPEED = float(_get('flock', 'min_bird_speed', 0.8)) # minimum airspeed floor
DIVE_SPEED_MULT = float(_get('flock', 'dive_speed_mult', 1.35))   # max speed multiplier diving straight down
CLIMB_SPEED_MULT = float(_get('flock', 'climb_speed_mult', 0.65)) # max speed multiplier climbing straight up

# --- 1c. FLOCK NN (tribe co-evolution) ---
NUM_TRIBES = int(_get('flock_nn', 'num_tribes', 20))
if NUM_TRIBES < 1 or NUM_BIRDS % NUM_TRIBES != 0:
    # contiguous tribe blocks need even splitting; walk down to a divisor
    while NUM_TRIBES > 1 and NUM_BIRDS % NUM_TRIBES != 0:
        NUM_TRIBES -= 1
    print(f"[flock_nn] num_tribes adjusted to {NUM_TRIBES} so birds split evenly")
BIRDS_PER_TRIBE = NUM_BIRDS // NUM_TRIBES
STARLING_MUTATION_SIGMA = float(_get('flock_nn', 'mutation_sigma', 0.06))
TRIBE_REPLACE_DEATHS = int(_get('flock_nn', 'replace_deaths', 5))
# Casualty counter decay (per second, exponential). Steady-state counter =
# death_rate / CASUALTY_DECAY. At 0.02/s the ~35s half-life meant none of
# the tribes ever reached the threshold at the observed ~4 pack kills per
# 30s generation — the hawk pressure never executed anything and the tribe
# pool never moved. 0.0005/s gives a ~23-minute half-life: sustained hunting
# pressure reliably triggers evolution every few generations while ancient
# isolated deaths still slowly fade. 0.0 disables decay.
CASUALTY_DECAY = float(_get('flock_nn', 'casualty_decay', 0.0005))
# Hidden-layer widths of the starling tribe MLP: STARLING_NET_SHAPE =
# (21,) + hidden + (6,) in flock.py. Arbitrary depth/width; the
# checkpoint shape guard starts fresh tribes on any mismatch.
def _hidden_layers(section, key, default):
    v = _get(section, key, default)
    try:
        out = tuple(int(x) for x in v)
        assert len(out) > 0 and all(x > 0 for x in out)
    except (TypeError, ValueError, AssertionError):
        print(f"[config] {section}.{key} = {v!r} invalid "
              f"(need a non-empty list of positive ints); using {default}")
        out = tuple(default)
    return out
STARLING_HIDDEN_LAYERS = _hidden_layers('flock_nn', 'hidden_layers', [32, 16])

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
# --- aerodynamic gravity (potential <-> kinetic energy exchange) ---
# Hawks trade altitude for airspeed: pitching the nose down (f_y < 0)
# accelerates along the flight path, pulling up (f_y > 0) bleeds speed.
HAWK_GRAVITY = float(_get('hawk', 'gravity', 2.2))        # m/s^2-equivalent along pitch f_y
DIVE_MAX_SPEED = float(_get('hawk', 'dive_max_speed', 6.8))  # terminal velocity in a steep stoop (above PRED_BOOST)
MIN_FLIGHT_SPEED = float(_get('hawk', 'min_flight_speed', 0.9))  # airspeed floor: hawks never hover
STALL_SPEED = float(_get('hawk', 'stall_speed', 1.4))     # below this, a climbing nose gently drops toward the horizon
# --- aerodynamic drift (velocity decoupled from the instantaneous nose) ---
# The flight path vector lags behind the nose: high-speed dives carve
# sweeping arcs with momentum instead of snapping to the new heading.
LIFT_GRIP = float(_get('hawk', 'lift_grip', 5.5))     # 1/s rate v_dir aligns toward nose f (lower = more drift/swoop)
INDUCED_DRAG = float(_get('hawk', 'induced_drag', 1.8))  # airspeed bleed rate in high-AoA turns (f diverges from v_dir)
# --- BOOST STAMINA / exhaustion cooldown ---
# Hawks cannot hold boost forever: throttle > 0 burns a stamina tank at
# 1/BOOST_DURATION per second (throttle-scaled), throttle <= 0 refills at
# 1/BOOST_RECHARGE per second. Hitting 0.0 locks out positive throttle
# until the tank recovers to BOOST_COOLDOWN_RECOVER — the anti-feathering
# rule: with the default 1.0, a fully emptied tank must wait out the whole
# BOOST_RECHARGE, while letting off the throttle BEFORE 0.0 recharges
# freely with no lockout. Applies to the AI pack AND the player hawk.
BOOST_DURATION = float(_get('hawk', 'boost_duration', 1.0))          # s of full boost, 100% -> 0%
BOOST_RECHARGE = float(_get('hawk', 'boost_recharge', 3.0))          # s to refill 0% -> 100%
BOOST_COOLDOWN_RECOVER = float(_get('hawk', 'boost_cooldown_recover', 1.0))  # stamina to clear lockout (0..1)
# Hidden-layer widths of the hawk MLP: NET_SHAPE = (15,) + hidden + (4,)
# in predators.py. Arbitrary depth/width; the population.npz shape guard
# starts a fresh evolution on any mismatch (GENOME_LEN changes).
HAWK_HIDDEN_LAYERS = _hidden_layers('hawk', 'hidden_layers', [32, 16])

# --- 1d. PLAYER (spectator hawk) ---
# Separate flight feel + key bindings from the AI pack. Speed/rate values
# fall back to the [hawk] section, so the player mirrors the pack unless
# overridden in [player]. Keys are Ursina key names: a bare string or a
# list (e.g. ["w", "up arrow"]) — every mapping accepts multiple keys.
def _pget(key, default):
    return _cfg.get('player', {}).get(key, _get('hawk', key, default))
PLAYER_CRUISE_SPEED = float(_pget('cruise_speed', 2.4))
PLAYER_BOOST_SPEED = float(_pget('boost_speed', 4.6))
PLAYER_SLOW_FACTOR = float(_pget('slow_factor', 0.45))
PLAYER_YAW_RATE = float(_pget('yaw_rate', 90.0))
PLAYER_PITCH_RATE = float(_pget('pitch_rate', 60.0))
PLAYER_ROLL_RATE = float(_pget('roll_speed', 120.0))
# extra yaw per unit of bank: coordinated turns on keyboard (0 = raw yaw only)
PLAYER_AUTO_COORD = float(_get('player', 'auto_coord', 0.35))

def _pkeys(key, default):
    v = _get('player', key, default)
    return (v,) if isinstance(v, str) else tuple(v)

PLAYER_KEYS = {
    'pitch_down': _pkeys('pitch_down_keys', ('w', 'up arrow')),
    'pitch_up':   _pkeys('pitch_up_keys',   ('s', 'down arrow')),
    'roll_left':  _pkeys('roll_left_keys',  ('a', 'left arrow')),
    'roll_right': _pkeys('roll_right_keys', ('d', 'right arrow')),
    'yaw_left':   _pkeys('yaw_left_keys',   ('q',)),
    'yaw_right':  _pkeys('yaw_right_keys',  ('e',)),
    'boost':      _pkeys('boost_keys',      ('left shift',)),
    'slow':       _pkeys('slow_keys',       ('left control',)),
    'reset':      _pkeys('reset_keys',      ('r',)),
}

POPULATION_SIZE = int(_get('pack', 'population_size', 10))
GENERATION_SECONDS = float(_get('pack', 'generation_seconds', 30.0))
# 3-tiered speciation: each of the top-3 lineages gets its own mutation
# strength — exploitation (tiny) around the champion, exploration (heavy)
# around the wildcard. Replaces the old flat mutation_sigma.
MUTATION_TINY = float(_get('pack', 'mutation_tiny', 0.02))
MUTATION_MEDIUM = float(_get('pack', 'mutation_medium', 0.08))
MUTATION_HEAVY = float(_get('pack', 'mutation_heavy', 0.20))
# King of the Hill: the Tier-1 lineage breeds an ALL-TIME best genome
# ("Hall of Fame") instead of just this generation's winner. The King
# always flies in Slot 0, and its standing is re-evaluated live every
# generation with an EMA: keep HOF_DECAY_KEEP of the historical rating
# + (1-keep) of the fresh Slot-0 fitness. A fluke genome that only won
# once decays toward its true (mediocre) live performance and gets
# dethroned; a genuinely strong King keeps its crown.
HOF_DECAY_KEEP = float(_get('pack', 'hof_decay_keep', 0.7))
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
BACKDROP_RES = 32   # tiny grid: cheap to render, deliberately coarse

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
START_CAMERA = int(_get('game', 'start_camera', 4))
# Dynamic FOV warp: widens toward BOOST_FOV as the tracked hawk approaches boost speed.
BASE_FOV = _get('hawk', 'base_fov', 90.0)
BOOST_FOV = _get('hawk', 'boost_fov', 110.0)

# --- rabbit constants ---
NUM_BURROWS = int(_get('rabbits', 'num_burrows', 45))
NUM_RABBITS = int(_get('rabbits', 'num_rabbits', 400))
RABBIT_FLEE_RADIUS = float(_get('rabbits', 'flee_radius', 25.0))      # hawk closer -> Flee
RABBIT_FREEZE_RADIUS = float(_get('rabbits', 'freeze_radius', 50.0))  # band between -> Freeze
RABBIT_RECOVER_RADIUS = float(_get('rabbits', 'recover_radius', 30.0))  # hawk farther -> end Flee
RABBIT_FORAGE_SPEED = float(_get('rabbits', 'forage_speed', 0.8))     # m/s (true SI, dt-scaled)
RABBIT_FLEE_SPEED = float(_get('rabbits', 'flee_speed', 6.0))         # m/s sprint
RABBIT_ESCAPE_RADIUS = float(_get('rabbits', 'escape_radius', 1.0))   # dive into burrow below this

# --- fox pairs (ground predators, continuous Moran evolution) ---
NUM_FOXES = int(_get('foxes', 'num_foxes', 20))
NUM_PAIRS = int(_get('foxes', 'num_pairs', 10))
if NUM_PAIRS < 1 or NUM_FOXES % NUM_PAIRS != 0:
    # contiguous pair blocks must split exactly (same policy as NUM_TRIBES)
    _orig = NUM_PAIRS
    while NUM_PAIRS > 1 and NUM_FOXES % NUM_PAIRS != 0:
        NUM_PAIRS -= 1
    print(f"[foxes] num_pairs adjusted to {NUM_PAIRS} so foxes split evenly")
FOXES_PER_PAIR = NUM_FOXES // NUM_PAIRS
FOX_CRUISE_SPEED = float(_get('foxes', 'cruise_speed', 3.5))
FOX_BOOST_SPEED = float(_get('foxes', 'boost_speed', 7.8))
FOX_STALK_FACTOR = float(_get('foxes', 'stalk_factor', 0.35))
FOX_YAW_RATE = float(_get('foxes', 'yaw_rate', 140.0))
FOX_TURN_DRAG = float(_get('foxes', 'turn_drag', 1.2))
FOX_BOOST_DURATION = float(_get('foxes', 'boost_duration', 1.8))
FOX_BOOST_RECHARGE = float(_get('foxes', 'boost_recharge', 4.5))
FOX_BOOST_COOLDOWN_RECOVER = float(_get('foxes', 'boost_cooldown_recover', 1.0))
FOX_CATCH_RADIUS = float(_get('foxes', 'catch_radius', 1.2))
FOX_STEALTH_FLEE_RADIUS = float(_get('foxes', 'stealth_flee_radius', 7.5))
FOX_STEALTH_FREEZE_RADIUS = float(_get('foxes', 'stealth_freeze_radius', 22.0))
FOX_SPRINT_FLEE_RADIUS = float(_get('foxes', 'sprint_flee_radius', 50.0))
FOX_PROXIMITY_RADIUS = float(_get('foxes', 'proximity_radius', 80.0))
FOX_STARVE_SECONDS = float(_get('foxes', 'starve_seconds', 180.0))
FOX_STALK_HUNGER_RELIEF = float(_get('foxes', 'stalk_hunger_relief', 0.4))
FOX_MUTATION_SIGMA = float(_get('foxes', 'mutation_sigma', 0.07))
FOX_HIDDEN_LAYERS = _hidden_layers('foxes', 'hidden_layers', [24, 12])
FOX_WOLF_PANIC_RADIUS = float(_get('foxes', 'wolf_panic_radius', 45.0))

# --- wolf pairs (apex ground predators; howl channel + pack-share hunting) ---
NUM_WOLVES = int(_get('wolves', 'num_wolves', 8))
NUM_WOLF_PAIRS = int(_get('wolves', 'num_pairs', 4))
if NUM_WOLF_PAIRS < 1 or NUM_WOLVES % NUM_WOLF_PAIRS != 0:
    # contiguous pair blocks must split exactly (same policy as NUM_PAIRS)
    while NUM_WOLF_PAIRS > 1 and NUM_WOLVES % NUM_WOLF_PAIRS != 0:
        NUM_WOLF_PAIRS -= 1
    print(f"[wolves] num_pairs adjusted to {NUM_WOLF_PAIRS} so wolves split evenly")
WOLVES_PER_PAIR = NUM_WOLVES // NUM_WOLF_PAIRS
WOLF_CRUISE_SPEED = float(_get('wolves', 'cruise_speed', 4.0))
WOLF_BOOST_SPEED = float(_get('wolves', 'boost_speed', 7.3))
WOLF_STALK_FACTOR = float(_get('wolves', 'stalk_factor', 0.45))
WOLF_YAW_RATE = float(_get('wolves', 'yaw_rate', 95.0))
WOLF_TURN_DRAG = float(_get('wolves', 'turn_drag', 1.0))
WOLF_BOOST_DURATION = float(_get('wolves', 'boost_duration', 4.5))
WOLF_BOOST_RECHARGE = float(_get('wolves', 'boost_recharge', 4.0))
WOLF_BOOST_COOLDOWN_RECOVER = float(_get('wolves', 'boost_cooldown_recover', 1.0))
WOLF_CATCH_RADIUS = float(_get('wolves', 'catch_radius', 1.5))
WOLF_FLEE_TRIGGER_RADIUS = float(_get('wolves', 'flee_trigger_radius', 55.0))
WOLF_PROXIMITY_RADIUS = float(_get('wolves', 'proximity_radius', 120.0))
WOLF_HOWL_RADIUS = float(_get('wolves', 'howl_radius', 250.0))
WOLF_ASSIST_RADIUS = float(_get('wolves', 'assist_radius', 35.0))
WOLF_ASSIST_SHARE = float(_get('wolves', 'assist_share', 0.65))
WOLF_HOWL_COOP_BONUS = float(_get('wolves', 'howl_coop_bonus', 25.0))
WOLF_STARVE_SECONDS = float(_get('wolves', 'starve_seconds', 180.0))
WOLF_MUTATION_SIGMA = float(_get('wolves', 'mutation_sigma', 0.07))
WOLF_HIDDEN_LAYERS = _hidden_layers('wolves', 'hidden_layers', [32, 16])

# --- 1e. COLORS ([colors] section; 0..255 per channel, a defaults to 255) ---
# Values are plain tuples (r,g,b,a) 0..255 so config.py stays Ursina-free;
# consumers convert with color.rgba32(*COLOR_X). Accepts "#rrggbb",
# "#rrggbbaa", [r,g,b] or [r,g,b,a]. NEVER color.rgb(): in this ursina
# build rgb() is 0..1-scale and 0..255 values clamp to WHITE.
def _color(section, key, default):
    v = _get(section, key, default)
    if isinstance(v, str):
        s = v.strip().lstrip('#')
        if len(s) not in (6, 8) or any(c not in '0123456789abcdefABCDEF' for c in s):
            print(f"[config] [colors] {key}={v!r} is not #rrggbb/#rrggbbaa; using default")
            v = default
        else:
            v = [int(s[i:i + 2], 16) for i in range(0, len(s), 2)]
    try:
        vals = [max(0, min(255, int(c))) for c in v]
        assert len(vals) in (3, 4)
    except (TypeError, ValueError, AssertionError):
        print(f"[config] [colors] {key}={v!r} is not [r,g,b(,a)] 0..255; using default")
        vals = list(default)
    if len(vals) == 3:
        vals.append(255)
    return tuple(vals)

COLOR_STARLING   = _color('colors', 'starling',   (0, 255, 255))
COLOR_RABBIT     = _color('colors', 'rabbit',     (140, 100, 70))
COLOR_FOX        = _color('colors', 'fox',        (215, 85, 25))
COLOR_HAWK_GOLD  = _color('colors', 'hawk_gold',  (255, 200, 40))    # pack rank 1
COLOR_HAWK_BRIGHT = _color('colors', 'hawk_bright', (230, 50, 40))   # pack rank 2
COLOR_HAWK_MUTED = _color('colors', 'hawk_muted', (150, 70, 50))     # pack rank 3+ / spawn
COLOR_PLAYER     = _color('colors', 'player',     (80, 220, 255))    # spectator hawk
COLOR_FOG        = _color('colors', 'fog',        (20, 30, 20))
COLOR_TREE_TRUNK = _color('colors', 'tree_trunk', (90, 60, 40))
COLOR_TREE_CROWN = _color('colors', 'tree_crown', (40, 90, 45))
COLOR_BUSH       = _color('colors', 'bush',       (60, 110, 50))
COLOR_ROCK       = _color('colors', 'rock',       (100, 105, 115))
COLOR_HUD_INFO   = _color('colors', 'hud_info',   (255, 255, 255, 120))
COLOR_HUD_GEN    = _color('colors', 'hud_gen',    (255, 220, 120, 200))
COLOR_HUD_KILLS  = _color('colors', 'hud_kills',  (255, 0, 0))
COLOR_DUST       = _color('colors', 'dust',       (200, 176, 144, 150))

# --- 1f. VISUALS ([visuals]: triplanar terrain + sun shadows + sprint dust) ---
TRIPLANAR_TOP_TEXTURE = str(_get('visuals', 'triplanar_top_texture', 'grass'))
TRIPLANAR_SIDE_TEXTURE = str(_get('visuals', 'triplanar_side_texture', 'rock_procedural'))
TRIPLANAR_TEXTURE_SCALE = float(_get('visuals', 'triplanar_texture_scale', 0.05))
# side scale accepts one float or an [x, y] pair (non-uniform tiling)
_ss = _get('visuals', 'triplanar_side_texture_scale', 0.02)
if isinstance(_ss, (list, tuple)):
    TRIPLANAR_SIDE_TEXTURE_SCALE_X = float(_ss[0])
    TRIPLANAR_SIDE_TEXTURE_SCALE_Y = float(_ss[1])
else:
    TRIPLANAR_SIDE_TEXTURE_SCALE_X = TRIPLANAR_SIDE_TEXTURE_SCALE_Y = float(_ss)
TRIPLANAR_BLEND_ZONE = float(_get('visuals', 'triplanar_blend_zone', 0.5))
TRIPLANAR_GRASS_BLEND_START = float(_get('visuals', 'triplanar_grass_blend_start', 0.25))
TRIPLANAR_GRASS_BLEND_END = float(_get('visuals', 'triplanar_grass_blend_end', 0.75))
SHADOW_MAP_SIZE = int(_get('visuals', 'shadow_map_size', 2048))
SHADOW_BIAS = float(_get('visuals', 'shadow_bias', 0.005))
SPRINT_DUST_FACTOR = float(_get('visuals', 'sprint_dust_factor', 1.5))
DUST_INTERVAL = float(_get('visuals', 'dust_interval', 0.12))
DUST_LIFETIME = float(_get('visuals', 'dust_lifetime', 0.4))
FOX_DUST_POOL = int(_get('visuals', 'fox_dust_pool', 96))
WOLF_DUST_POOL = int(_get('visuals', 'wolf_dust_pool', 48))
# wolf coat + howl highlight live in [wolves] (not [colors]): section-local
# tunables alongside the rest of the wolf block (parsed after _color exists).
COLOR_WOLF      = _color('wolves', 'color',      (130, 140, 155))
COLOR_WOLF_HOWL = _color('wolves', 'howl_color', (220, 240, 255))
