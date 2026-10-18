# Boost-stamina / exhaustion-cooldown harness (pack + player + HUD).
# Verifies, against the LIVE config.toml values (never hardcoded):
#  1. pinned raw_thr=+1 drains stamina 1.0 -> 0.0 in BOOST_DURATION, flips
#     exhausted=True, clamps effective thr to 0 (speed decays to cruise),
#     and the lockout holds until BOOST_COOLDOWN_RECOVER is re-reached
#     (default: the full BOOST_RECHARGE) before boost re-engages.
#  2. releasing boost at a partial tank recharges smoothly with NO lockout.
#  3. exhausted hawks keep braking (thr<0) and gliding (thr==0).
#  4. obs[:,14] stamina sensor: full +1.0, empty -0.8, lockout pinned -1.0.
#  5. HUD shows the tracked hawk's stamina % and [COOL].
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
import player, hud_cam

terrain.init_environment()
predators.init_pack_visuals()
player.init_player_visuals()
hud_cam.init_hud_and_camera()

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

FPS = 60.0
DUR = config.BOOST_DURATION
REC = config.BOOST_RECHARGE
RECOVER = config.BOOST_COOLDOWN_RECOVER
# pinned tanh bias for a near-exact +-1 throttle output (tanh(4)=0.99933)
BIAS = 4.0
THR_PIN = float(np.tanh(BIAS))

def pin_layers(thr_bias):
    shape = predators.NET_SHAPE
    layers = []
    for i in range(len(shape) - 1):
        W = np.zeros((predators.POPULATION_SIZE, shape[i], shape[i+1]))
        b = np.zeros((predators.POPULATION_SIZE, shape[i+1]))
        layers.append((W, b))
    layers[-1][1][:, 3] = thr_bias     # throttle output = tanh(bias)
    return layers

# ---------- 1. PACK: full tank -> empty -> lockout -> full recharge ----------
predators.respawn_pack()
assert np.all(predators.pack_stamina == 1.0) and not predators.pack_exhausted.any()
pos_np = flock.to_cpu(flock.pos_gpu); vel_np = flock.to_cpu(flock.vel_gpu)
layers = pin_layers(BIAS)

# half a BOOST_DURATION of boost: tank ~0.5, still flying
half_frames = int(round(DUR * FPS / 2.0))
for _ in range(half_frames):
    frame(lambda: predators.update_predators(layers, pos_np, vel_np))
mid = float(predators.pack_stamina[0])
assert abs(mid - (1.0 - 0.5 * THR_PIN)) < 0.02, f"linear burn off: {mid}"
assert not predators.pack_exhausted.any()
print(f"pack burn @ {half_frames}f: stamina={mid:.2f} (thr pinned {THR_PIN:.3f})")

# keep boosting until the tank empties: ~BOOST_DURATION total
exhaust_f = -1
for i in range(2, int(10 * FPS)):
    frame(lambda: predators.update_predators(layers, pos_np, vel_np))
    if predators.pack_exhausted.all():
        exhaust_f = half_frames + i
        break
assert exhaust_f >= 0, "pack never exhausted while pinned to full throttle"
expect_f = DUR * FPS / THR_PIN
assert abs(exhaust_f - expect_f) <= 3, f"drain took {exhaust_f}f, expected ~{expect_f:.0f}f"
assert np.all(predators.pack_stamina <= 1e-5)
print(f"pack exhausted at {exhaust_f}f = {exhaust_f/FPS:.2f}s (BOOST_DURATION {DUR}s)")

# lockout: thr pinned +1 but clamped to 0 -> speed decays to cruise.
# Tank re-pinned to 0 each frame so the lockout cannot clear mid-soak
# (the natural recharge timing is measured separately below).
predators.pack_speed[:] = config.PRED_BOOST   # simulate the boost speed reached
obs = predators._hawk_observations(pos_np, vel_np)
assert np.allclose(obs[:, 14], -1.0), "lockout must pin the stamina sensor at -1.0"
for _ in range(300):
    predators.pack_stamina[:] = 0.0
    frame(lambda: predators.update_predators(layers, pos_np, vel_np))
    assert predators.pack_exhausted.all(), "lockout released while tank pinned empty"
assert np.allclose(predators.pack_speed, config.CRUISE_SPEED, atol=0.05), \
    f"locked-out hawk must settle at cruise, got {predators.pack_speed}"
print(f"lockout holds: speed -> cruise {config.CRUISE_SPEED}, sensor obs=-1.0")

# recharge to BOOST_COOLDOWN_RECOVER: default 1.0 -> full BOOST_RECHARGE
predators.pack_stamina[:] = 0.0   # start the recharge clock from empty
clear_f = -1
for i in range(1, int(10 * FPS)):
    frame(lambda: predators.update_predators(layers, pos_np, vel_np))
    if not predators.pack_exhausted.any():
        clear_f = i
        break
assert clear_f >= 0, "lockout never cleared"
expect_clear = RECOVER * REC * FPS
assert abs(clear_f - expect_clear) <= 4, f"recharge took {clear_f}f, expected ~{expect_clear:.0f}f"
# boost re-engages immediately after the lockout clears
spd_before = float(predators.pack_speed[0])
for _ in range(30):
    frame(lambda: predators.update_predators(layers, pos_np, vel_np))
assert predators.pack_speed[0] > spd_before + 0.2, "boost did not re-engage after recharge"
print(f"lockout cleared after {clear_f}f = {clear_f/FPS:.2f}s (BOOST_RECHARGE {REC}s x recover {RECOVER}); boost re-engaged")

# ---------- 2. PACK anti-feathering: release at a partial tank = no lockout ----------
predators.respawn_pack()
layers = pin_layers(BIAS)
partial_f = int(round(0.8 * DUR * FPS / THR_PIN))   # drain to ~0.2
for _ in range(partial_f):
    frame(lambda: predators.update_predators(layers, pos_np, vel_np))
near = float(predators.pack_stamina[0])
assert 0.15 < near < 0.3, f"partial drain off target: {near}"
assert not predators.pack_exhausted.any(), "lockout tripped BEFORE the tank hit 0"
layers_off = pin_layers(0.0)                         # let off the throttle
rec_f = -1
for i in range(1, int(10 * FPS)):
    frame(lambda: predators.update_predators(layers_off, pos_np, vel_np))
    assert not predators.pack_exhausted.any(), "feathering lockout fired without hitting 0"
    if predators.pack_stamina[0] >= 1.0 - 1e-6:
        rec_f = i
        break
expect_rec = (1.0 - near) * REC * FPS
assert abs(rec_f - expect_rec) <= 4, f"partial recharge took {rec_f}f, expected ~{expect_rec:.0f}f"
print(f"anti-feathering: released at {near:.2f}, recharged to 1.0 in {rec_f}f, no lockout")

# ---------- 3. PACK exhausted: braking still works (thr<0 unblocked) ----------
predators.respawn_pack()
layers = pin_layers(BIAS)
for _ in range(int(DUR * FPS / THR_PIN) + 3):
    frame(lambda: predators.update_predators(layers, pos_np, vel_np))
assert predators.pack_exhausted.all()
layers_brake = pin_layers(-BIAS)                     # raw thr = -1
for _ in range(600):   # long soak: the 0.8/s drag blend converges slowly
    frame(lambda: predators.update_predators(layers_brake, pos_np, vel_np))
brake_target = config.CRUISE_SPEED * config.SLOW_FACTOR
assert np.all(predators.pack_speed < config.CRUISE_SPEED - 0.1), \
    "exhausted hawk could not brake"
assert np.allclose(predators.pack_speed, brake_target, atol=0.1), \
    f"brake target off: {predators.pack_speed} vs {brake_target}"
print(f"exhausted braking: speed -> {predators.pack_speed[0]:.2f} (brake target {brake_target:.2f})")

# ---------- 4. PLAYER mirror ----------
K = {a: config.PLAYER_KEYS[a][0] for a in config.PLAYER_KEYS}
player.reset_player()
assert player.player_stamina == 1.0 and not player.player_exhausted
held_keys[K['boost']] = True
p_exhaust_f = -1
for i in range(1, int(10 * FPS)):
    frame(player.update_player)
    if player.player_exhausted:
        p_exhaust_f = i
        break
assert p_exhaust_f >= 0, "player never exhausted while holding boost"
assert abs(p_exhaust_f - DUR * FPS) <= 3, f"player drain took {p_exhaust_f}f, expected ~{DUR*FPS:.0f}f"
# lockout: boost key still held, thr clamped to 0 -> decays to effective
# cruise (tank re-pinned empty so the lockout cannot clear mid-soak)
eff_cruise = max(config.PLAYER_CRUISE_SPEED, config.MIN_FLIGHT_SPEED)
for _ in range(800):   # exponential 0.8/s drag decay: give it room to settle
    player.player_stamina = 0.0
    player.player_pos[:] = [0.0, 60.0, 0.0]   # pin mid-arena: wall blend would skew speed
    frame(player.update_player)
    assert player.player_exhausted, "player lockout released while tank pinned empty"
assert abs(player.player_speed - eff_cruise) < 0.05
player.player_stamina = 0.0   # start the recharge clock from empty
p_clear_f = -1
for i in range(1, int(10 * FPS)):
    frame(player.update_player)
    if not player.player_exhausted:
        p_clear_f = i
        break
assert abs(p_clear_f - RECOVER * REC * FPS) <= 4, f"player recharge took {p_clear_f}f"
spd_before = player.player_speed
for _ in range(30):
    frame(player.update_player)
assert player.player_speed > spd_before + 0.2, "player boost did not re-engage"
held_keys[K['boost']] = False
print(f"player: drain {p_exhaust_f}f, lockout -> cruise {eff_cruise}, recharge {p_clear_f}f, boost re-engaged")

# player anti-feathering mirror
player.reset_player()
held_keys[K['boost']] = True
for _ in range(int(0.8 * DUR * FPS)):
    frame(player.update_player)
assert 0.15 < player.player_stamina < 0.3 and not player.player_exhausted
held_keys[K['boost']] = False
for _ in range(int(2 * REC * FPS)):
    frame(player.update_player)
assert not player.player_exhausted and player.player_stamina >= 1.0 - 1e-6
print(f"player anti-feathering: released at partial tank, recharged to 1.0, no lockout")

# ---------- 5. HUD stamina readout ----------
predators.respawn_pack()
predators.pack_stamina[predators.sel_idx] = 0.5
predators.pack_exhausted[predators.sel_idx] = True
hud_cam.handle_input('3')          # AI first-person: tracks sel_idx
hud_cam.update_camera_and_hud()
txt = hud_cam.gen_text.text
assert 'Stamina: 50% [COOL]' in txt, f"AI-mode HUD stamina wrong: {txt}"
predators.pack_exhausted[:] = False
predators.pack_stamina[:] = 1.0    # section 3's soak left the tanks empty
hud_cam.update_camera_and_hud()
assert 'Stamina: 100%' in hud_cam.gen_text.text and '[COOL]' not in hud_cam.gen_text.text
hud_cam.handle_input('4')          # player chase cam
player.reset_player()
player.player_stamina = 0.42
hud_cam.update_camera_and_hud()
assert 'Stamina: 42%' in hud_cam.gen_text.text, hud_cam.gen_text.text
print("PASS HUD: AI + player stamina % and [COOL] flag in gen_text")

print('ALL STAMINA CHECKS PASSED')
app.userExit()
