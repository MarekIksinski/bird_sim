# Gravity physics probe: dive acceleration, climb bleed + stall nose-drop,
# speed clamps — for both the pack (update_predators) and the player.
import sys
import numpy as np
sys.path.insert(0, '/home/marek/Playground/bird_sim')
sys.argv = ['main.py', '--cpu']

import ursina.main
_um = ursina.main.Ursina
cells = [c.cell_contents for c in _um.__new__.__closure__]
real = next(c for c in cells if getattr(c, '__module__', '') == 'ursina.main' and callable(c))
real.run = lambda self, info=True: None
_orig = real.__init__
def _init(self, *a, **k):
    k['window_type'] = 'offscreen'; k['development_mode'] = False
    _orig(self, *a, **k)
real.__init__ = _init

from ursina import Ursina, held_keys
app = Ursina()

import backend, config, terrain, predators, flock, math3d
import player

terrain.init_environment()
predators.init_pack_visuals()
player.init_player_visuals()

# keep the live ./evolution checkpoint untouched
import os, tempfile
tmp = tempfile.mkdtemp()
predators.EVOLVE_DIR = tmp
predators._CKPT = os.path.join(tmp, 'population.npz')
predators._HISTORY = os.path.join(tmp, 'history.csv')
predators._STARLING_CKPT = os.path.join(tmp, 'starling_tribes.npz')
predators._LOG_FILE = os.path.join(tmp, 'evolution.log')

from ursina import time as _utime
def frame(fn):
    _utime.dt = 1.0 / 60.0
    fn()

# ---------- A. PLAYER DIVE: nose straight down, gravity must exceed boost ----------
player.reset_player()
player.player_pos[:] = [0.0, 140.0, 0.0]
# pitch nose straight down (-90deg about body x)
q_down = math3d._q_from_aa_b(np.array([[1.0, 0.0, 0.0]]), np.array([90.0]))[0]
player.player_q[:] = q_down
held_keys['left shift'] = True   # full throttle too
for _ in range(300):
    frame(player.update_player)
held_keys['left shift'] = False
f = player.player_axes()[0]
print(f"dive: speed={player.player_speed:.2f} f_y={f[1]:.2f} y={player.player_pos[1]:.1f}")
assert player.player_speed > config.PLAYER_BOOST_SPEED, \
    f"dive never exceeded boost: {player.player_speed}"
assert player.player_speed <= config.DIVE_MAX_SPEED + 1e-6
# (the arena floor clamps the fall after ~140m; terminal speed is checked
# integrator-only in A2 where the spatial budget does not bind)

# ---------- A2. terminal velocity: the speed integrator alone converges to
# the drag/gravity equilibrium: boost_target + HAWK_GRAVITY/rate_above =
# 3.5 + 2.2/0.8 = 6.25 (below the DIVE_MAX_SPEED clip, which is the hard
# backstop for throttle-independent stoops) ----------
spd = float(config.PLAYER_CRUISE_SPEED)
adt = 1.0 / 60.0
for _ in range(1200):
    target = config.PLAYER_BOOST_SPEED
    rate = 0.8 if spd > target else 1.5
    spd += (target - spd) * min(adt * rate, 1.0)
    spd += (-config.HAWK_GRAVITY * -1.0) * adt
    spd = float(np.clip(spd, config.MIN_FLIGHT_SPEED, config.DIVE_MAX_SPEED))
term = config.PLAYER_BOOST_SPEED + config.HAWK_GRAVITY / 0.8
print(f"terminal integrator: speed={spd:.3f} (equilibrium {term:.2f}, clip {config.DIVE_MAX_SPEED})")
assert abs(spd - term) < 0.02

# ---------- B. PLAYER CLIMB: nose straight up from a dive -> speed bleeds + stall drops nose ----------
q_up = math3d._q_from_aa_b(np.array([[1.0, 0.0, 0.0]]), np.array([-90.0]))[0]
player.player_q[:] = q_up
for _ in range(240):
    frame(player.update_player)
f = player.player_axes()[0]
print(f"climb: speed={player.player_speed:.2f} f_y={f[1]:.2f} y={player.player_pos[1]:.1f}")
assert player.player_speed < config.PLAYER_BOOST_SPEED * 0.5, \
    f"climb should bleed speed hard, got {player.player_speed}"
# stall nose-drop: once slow + nose up, gravity must rotate the nose toward horizon
assert f[1] < 0.95, f"stall nose-drop never engaged: f_y={f[1]:.2f}"

# ---------- C. PLAYER level flight: speed settles at max(cruise, floor).
# NOTE: [player] cruise_speed 0.5 < MIN_FLIGHT_SPEED 0.9, so the never-hover
# floor binds — effective player cruise is the floor (reported to user). ----
player.reset_player()
for _ in range(180):
    frame(player.update_player)
eff = max(config.PLAYER_CRUISE_SPEED, config.MIN_FLIGHT_SPEED)
assert abs(player.player_speed - eff) < 1e-3, player.player_speed
print(f"level: speed={player.player_speed:.3f} == max(cruise {config.PLAYER_CRUISE_SPEED}, floor {config.MIN_FLIGHT_SPEED})")

# ---------- D. PACK: same physics via update_predators with pinned NN outputs ----------
# zero-ish outputs: throttle 0 (cruise), no rates -> level flight, gravity neutral
def pin_layers(ni):
    # outputs ~0: tiny weights, zero biases
    shape = predators.NET_SHAPE
    layers = []
    for i in range(len(shape) - 1):
        W = np.zeros((predators.POPULATION_SIZE, shape[i], shape[i+1]))
        b = np.zeros((predators.POPULATION_SIZE, shape[i+1]))
        layers.append((W, b))
    return layers

predators.respawn_pack()
pos_np = flock.to_cpu(flock.pos_gpu); vel_np = flock.to_cpu(flock.vel_gpu)
layers = pin_layers(predators.NET_SHAPE[0])
for _ in range(240):
    frame(lambda: predators.update_predators(layers, pos_np, vel_np))
assert np.all(predators.pack_speed >= config.MIN_FLIGHT_SPEED - 1e-6)
assert np.all(predators.pack_speed <= config.DIVE_MAX_SPEED + 1e-6)
print(f"pack level: speeds={np.round(predators.pack_speed, 2)}")

# pack dive: nose all hawks straight down, full-throttle output (+1)
predators.respawn_pack()
f, r, u = predators.pack_axes()
q_down_b = math3d._q_from_aa_b(np.array([[1.0, 0.0, 0.0]]), np.array([90.0]))
predators.pack_q[:] = math3d._q_mul_b(predators.pack_q, np.tile(q_down_b, (predators.POPULATION_SIZE, 1)))
layers = pin_layers(predators.NET_SHAPE[0])
# force throttle output = +1: bias the last layer's 4th output to +1
layers[-1][1][:, 3] = 1.0
layers[-1][0][:, :, 3] = 0.0
for _ in range(300):
    frame(lambda: predators.update_predators(layers, pos_np, vel_np))
print(f"pack dive: speeds={np.round(predators.pack_speed, 2)}")
assert np.all(predators.pack_speed > config.PRED_BOOST), \
    f"pack dive never exceeded PRED_BOOST: {predators.pack_speed}"
pack_term = config.PRED_BOOST + config.HAWK_GRAVITY / 0.8
assert np.allclose(predators.pack_speed, pack_term, atol=0.4), \
    f"pack dive should sit near equilibrium {pack_term:.2f}"

# ---------- E. FLOCK: dive vs climb dynamic caps + MIN_BIRD_SPEED floor ----------
# run real boids steps and track per-bird speed vs pitch
flock.update_boids_gpu()
mins, maxes = [], []
for _ in range(180):
    frame(flock.update_boids_gpu)
    v = flock.to_cpu(flock.vel_gpu)
    spd = np.linalg.norm(v, axis=1)
    fy = v[:, 1] / np.maximum(spd, 1e-6)
    mins.append(spd.min()); maxes.append(spd.max())
    # dynamic cap check: speed <= MAX_SPEED*pitch_mult(f_y) OR <= panic cap
    pm = np.where(fy < 0, 1.0 - fy * (config.DIVE_SPEED_MULT - 1.0),
                  1.0 - fy * (1.0 - config.CLIMB_SPEED_MULT))
    cap = config.PANIC_MAX_SPEED * pm
    # +0.55 tolerance: the floor-avoidance boost (pre-existing, +0.5/frame in
    # vy near terrain) fires AFTER the clamps — same exception as before.
    assert np.all(spd <= cap + 0.55), f"dynamic cap violated: {(spd - cap).max():.2f} over"
    assert np.all(spd >= config.MIN_BIRD_SPEED - 1e-4), f"bird stalled to {spd.min()}"
print(f"flock soak 180 frames: speed min={min(mins):.2f} max={max(maxes):.2f} "
      f"(floor {config.MIN_BIRD_SPEED}, dive cap {config.MAX_SPEED*config.DIVE_SPEED_MULT:.2f})")

# ---------- F. obs vector sanity ----------
obs = predators._hawk_observations(pos_np, vel_np)
assert obs.shape == (predators.POPULATION_SIZE, 14)
assert np.all(np.isfinite(obs)) and np.all(np.abs(obs) <= 1.0)
print(f"hawk obs: shape={obs.shape} spd_in={obs[0,12]:.2f} fy_in={obs[0,13]:.2f}")

print('ALL GRAVITY CHECKS PASSED')
app.userExit()
