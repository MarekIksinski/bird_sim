# -*- coding: utf-8 -*-
"""verify_fox_cams.py — offscreen check of fox camera modes 5/6.
Boots Ursina headless, steps real frames via taskMgr.step() (the sim chain
runs inside FlockVisuals.update -> predators.update_pack -> rabbits/foxes),
then checks:
  - fox_quat() unit norm + pitches UP on rising ground
  - update_foxes() keeps best_fox_idx in range on the best-rate pair
  - keys '6'/'7' select modes 5/6; TAB is fox-aware in fox cams, hawk-aware
    elsewhere
  - chase cam never buries itself in terrain; FP cam stays ahead of the
    0.85 m nose tip; HUD stamina line renders
Run: .venv/bin/python -u tools/verify_fox_cams.py
"""
import os, sys
os.environ['FORCE_CPU'] = '1'
sys.path.insert(0, '/home/marek/Playground/bird_sim')
os.chdir('/home/marek/Playground/bird_sim')

import numpy as np
from direct.showbase.ShowBase import taskMgr
from ursina import Ursina, camera, application

app = Ursina(window_type='offscreen', development_mode=False)
import config, terrain, flock, rabbits, foxes, predators, player, hud_cam

terrain.init_environment()
flock_viz = flock.FlockVisuals()
rabbit_viz = rabbits.RabbitVisuals()
fox_viz = foxes.FoxVisuals()
predators.init_pack_visuals()
player.init_player_visuals()
hud_cam.init_hud_and_camera()
predators.respawn_pack()

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f'{"OK " if cond else "FAIL"} {name}')

def step(n=1):
    for _ in range(n):
        taskMgr.step()
        # main.py wires update_camera_and_hud via its global update();
        # offscreen we call it explicitly after each frame
        hud_cam.update_camera_and_hud()

# --- fox_quat sanity: face +Z, see how it pitches to the local slope ---
foxes.fox_pos[0] = [0.0, terrain.height_at(0, 0), 0.0]
foxes.fox_yaw[0] = 0.0
for _ in range(40):
    foxes.fox_quat(0)
q = foxes.fox_quat(0)
check('fox_quat unit norm', abs(np.linalg.norm(q) - 1.0) < 1e-5)
fwd = np.array([2 * (q[0] * q[2] + q[3] * q[1]),
                2 * (q[1] * q[2] - q[3] * q[0]),
                1.0 - 2 * (q[0] * q[0] + q[1] * q[1])])   # R @ (0,0,1)
h0, h1 = terrain.height_at(0, 0), terrain.height_at(0, 1.0)
slope = h1 - h0
if slope > 0.5:
    check(f'pitch UP on rising ground (slope {slope:+.2f}m)', fwd[1] > 0.02)
elif slope < -0.5:
    check(f'pitch DOWN on falling ground (slope {slope:+.2f}m)', fwd[1] < -0.02)
else:
    check('flat ground: fwd ~ horizontal', abs(fwd[1]) < 0.05)

# --- run real frames, then check best_fox_idx bookkeeping ---
step(120)
pair_rate = foxes.pair_score / np.maximum(foxes.pair_survival_time, 10.0)
bf = foxes.best_fox_idx
check('best_fox_idx in range', 0 <= bf < config.NUM_FOXES)
check('best_fox_idx on argmax-rate pair',
      foxes.fox_pair_np[bf] == int(np.argmax(pair_rate)))

def urs_input(key):
    hud_cam.handle_input(key)

# --- key bindings + TAB awareness (direct handler call) ---
urs_input('6')
check("key '6' -> mode 5", hud_cam.curr_mode == 5)
urs_input('tab')
check('TAB in fox cam cycles fox, not hawk',
      foxes.sel_fox_idx == 1 and foxes.sel_used and predators.sel_idx == 0)
urs_input('tab')
check('TAB cycles fox again', foxes.sel_fox_idx == 2)
urs_input('7')
check("key '7' -> mode 6", hud_cam.curr_mode == 6)
urs_input('tab')
check('TAB in mode 6 cycles fox', foxes.sel_fox_idx == 3)
urs_input('2')
urs_input('tab')
check('TAB in hawk cam cycles hawk, fox index untouched',
      predators.sel_idx == 1 and foxes.sel_fox_idx == 3)

# --- drive chase cam 200 frames: camera must never bury in terrain ---
urs_input('6')
worst_clear = 1e9
for _ in range(150):
    step()
    cam_pos = np.array([camera.position.x, camera.position.y, camera.position.z])
    g = terrain.height_at(cam_pos[0], cam_pos[2])
    worst_clear = min(worst_clear, cam_pos[1] - g)
check('chase cam never buried (>= +1.0m)', worst_clear >= 1.0)
print(f'    chase cam worst ground clearance: {worst_clear:.2f} m')

# --- drive FP cam 200 frames: camera stays outside the nose-tip sphere ---
urs_input('7')
min_d = 1e9
for _ in range(150):
    step()
    trk = foxes.sel_fox_idx
    cam_pos = np.array([camera.position.x, camera.position.y, camera.position.z])
    d = np.linalg.norm(cam_pos - foxes.fox_pos[trk])
    min_d = min(min_d, d)
check('FP cam ahead of 0.85m nose tip (no clip)', min_d >= 0.85)
print(f'    FP cam min distance to fox origin: {min_d:.2f} m')

# HUD stamina line renders in fox modes
check('HUD line has Stamina', 'Stamina:' in hud_cam.gen_text.text)
check('HUD help text mentions fox cams', '6/7 Fox cams' in hud_cam.hawk_info.text)

application.base.userExit()
print('ALL FOX CAM CHECKS PASSED' if ok else 'FOX CAM CHECKS FAILED')
sys.exit(0 if ok else 1)
