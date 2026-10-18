# Offscreen harness: aerodynamic drift, lift bite, induced turn drag.
# 1) straight flight: aoa_slip == 0, terminal dive speed unchanged (6.25 equilibrium)
# 2) 90deg pull-out of a fast dive: v_dir lags f for several frames (swoop arc,
#    no instant corner) and aoa_slip > 0 bleeds speed during the high-G pull
# 3) respawn velocity along the nose; pack form behaves identically
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

# ---------- 1. respawn velocity rides the nose ----------
player.reset_player()
f0 = player.player_axes()[0]
assert np.linalg.norm(player.player_vel) > 1e-3
assert np.dot(player.player_vel / np.linalg.norm(player.player_vel), f0) > 0.999
predators.respawn_pack()
fp, _, _ = predators.pack_axes()
for i in range(predators.POPULATION_SIZE):
    d = predators.pack_vel[i] / np.linalg.norm(predators.pack_vel[i])
    assert np.dot(d, fp[i]) > 0.999, (i, d, fp[i])
print("PASS respawn: pack_vel / player_vel initialized along the nose")

# ---------- 2. straight dive: aoa_slip stays 0, terminal speed unchanged ----------
player.reset_player()
player.player_pos[:] = [0.0, 140.0, 0.0]
q_down = math3d._q_from_aa_b(np.array([[1.0, 0.0, 0.0]]), np.array([90.0]))[0]
player.player_q[:] = q_down
held_keys[config.PLAYER_KEYS['boost'][0]] = True
max_slip = 0.0
for i in range(360):
    player.player_stamina = 1.0; player.player_exhausted = False  # drift test, not stamina test
    frame(player.update_player)
    if i < 60:
        continue   # skip the intentional nose-snap alignment transient
    f = player.player_axes()[0]
    v = player.player_vel / max(np.linalg.norm(player.player_vel), 1e-6)
    max_slip = max(max_slip, 1.0 - np.dot(f, v))
held_keys[config.PLAYER_KEYS['boost'][0]] = False
term_ref = config.PLAYER_BOOST_SPEED + config.HAWK_GRAVITY / 0.8  # 6.25 (commit 1cdbd25)
print(f"straight dive: speed={player.player_speed:.3f} ref={term_ref:.2f} max_slip={max_slip:.4f}")
assert max_slip < 0.02, f"straight dive developed slip: {max_slip}"
assert abs(player.player_speed - term_ref) < 0.15, \
    f"terminal dive speed CHANGED from 1cdbd25: {player.player_speed} vs {term_ref}"

# ---------- 3. 90deg pull-out: v_dir lags f, swoop arc, induced drag bleeds ----------
# (still nose-down at ~6.2 speed) — point the nose UP 90deg from the dive
# (i.e. 45deg above horizon): the classic stoop pull-out, not a reversal.
player.player_q[:] = math3d._q_from_aa_b(np.array([[1.0, 0.0, 0.0]]), np.array([-135.0]))[0]
speed_at_pull = player.player_speed
slips, lags = [], []
y_min = player.player_pos[1]
for _ in range(45):
    frame(player.update_player)
    f = player.player_axes()[0]
    v = player.player_vel / max(np.linalg.norm(player.player_vel), 1e-6)
    slips.append(1.0 - np.dot(f, v))
    lags.append(np.degrees(np.arccos(np.clip(np.dot(f, v), -1, 1))))
    y_min = min(y_min, player.player_pos[1])
lag0, lag10, lag30 = lags[0], lags[10], lags[30]
print(f"pull-out: speed {speed_at_pull:.2f} -> {player.player_speed:.2f}, "
      f"lag f->v_dir: {lag0:.0f}deg @f1, {lag10:.0f}deg @f11, {lag30:.0f}deg @f31; "
      f"max slip={max(slips):.2f}; y dipped to {y_min:.1f}")
assert lag0 > 60, f"nose snap: v_dir jumped with f (lag={lag0:.0f}deg)"
assert lag10 < lag0 - 10, f"v_dir never caught up to f (lag {lag0:.0f}->{lag10:.0f})"
assert max(slips) > 0.5, f"aoa_slip never engaged: {max(slips)}"
# induced drag: speed must bleed meaningfully more than pure gravity alone
assert player.player_speed < speed_at_pull - 0.8, \
    f"no induced-drag bleed in pull-out: {speed_at_pull:.2f} -> {player.player_speed:.2f}"
# swoop: trajectory keeps descending after the nose points up (overshoot arc)
assert y_min < player.player_pos[1], "no swoop overshoot — instant corner"

# ---------- 4. pack form: pinned NN, same behaviour vectorized ----------
def pin_layers():
    shape = predators.NET_SHAPE
    layers = []
    for i in range(len(shape) - 1):
        W = np.zeros((predators.POPULATION_SIZE, shape[i], shape[i+1]))
        b = np.zeros((predators.POPULATION_SIZE, shape[i+1]))
        layers.append((W, b))
    return layers

predators.respawn_pack()
predators.pack_pos[:, 1] = 120.0
predators.pack_speed[:] = 6.0
f, _, _ = predators.pack_axes()
predators.pack_vel[:] = np.array([0.0, 0.0, -1.0]) * predators.pack_speed[:, None]
layers = pin_layers()
q_down_b = math3d._q_from_aa_b(np.array([[1.0, 0.0, 0.0]]), np.array([90.0]))
predators.pack_q[:] = math3d._q_mul_b(predators.pack_q, np.tile(q_down_b, (predators.POPULATION_SIZE, 1)))
pos_np = flock.to_cpu(flock.pos_gpu); vel_np = flock.to_cpu(flock.vel_gpu)
# level nose, velocity still pointing down = pure AoA slip -> strong bleed
f2, _, _ = predators.pack_axes()
speed_before = predators.pack_speed.copy()
predators.update_predators(layers, pos_np, vel_np)
slip = 1.0 - np.einsum('ij,ij->i', f2, predators.pack_vel / np.linalg.norm(predators.pack_vel, axis=1, keepdims=True))
bleed = speed_before - predators.pack_speed
print(f"pack AoA slip frame: slip={slip.mean():.2f} bleed/frame={bleed.mean():.3f}")
assert np.all(slip > 0.5) and np.all(bleed > 0.05), "induced drag dead in pack form"
assert np.all(np.isfinite(predators.pack_pos)) and np.all(predators.pack_pos[:, 1] < 130.0)

print('ALL DRIFT CHECKS PASSED')
app.userExit()
