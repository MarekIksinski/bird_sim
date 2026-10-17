# -*- coding: utf-8 -*-
"""
hud_cam.py — camera modes, HUD text, keyboard handling (top layer below
main.py: imports config/math3d/terrain/predators/flock/rabbits).

main.py wires this into Ursina by defining thin global input()/update()
callbacks that delegate to handle_input() and update_camera_and_hud().
"""
import numpy as np
from ursina import (Vec3, camera, color, application, lerp, Text,
                    time, scene)
from panda3d.core import Quat as PQuat

from config import (POPULATION_SIZE, START_CAMERA, BASE_FOV, BOOST_FOV,
                    GENERATION_SECONDS, CRUISE_SPEED, PRED_BOOST)
from math3d import _slerp_np, _q_rot_b
import terrain
import predators
import flock

hawk_info = None
gen_text = None
kill_counter = None
_last_gen_line = ''

# --- 5. CAMERA ---
# config start_camera: 1 fixed wide, 2 pack chase (best hawk), 3 first-person (selected)
curr_mode = max(0, min(2, START_CAMERA - 1))

def init_hud_and_camera():
    """Create the HUD Text objects and set the initial camera pose.
    Call only AFTER app = Ursina(...) exists."""
    global hawk_info, gen_text, kill_counter
    hawk_info = Text('1/2/3 cameras  TAB cycle hawk  ESC quit  —  hawks evolve every 30s',
                     origin=(0, 0), y=-0.45, size=0.035, color=color.rgba(255, 255, 255, 120))
    gen_text = Text('Gen: 1 | Time: 0.0s | Best Kills: 0', origin=(0, 0), y=-0.38, size=0.045,
                    color=color.rgba(255, 220, 120, 200))
    kill_counter = Text('catches: 0', origin=(0, 0), x=-0.9, y=-0.45, size=0.05, color=color.red)
    camera.position = (0, 35, -45)

def handle_input(key):
    global curr_mode
    if key == '1':
        camera.parent = scene
        camera.position = (0, 30, -50)
        camera.look_at((0,0,0))
        curr_mode = 0
    if key == '2': curr_mode = 1
    if key == '3': curr_mode = 2
    if key == 'tab': predators.sel_idx = (predators.sel_idx + 1) % POPULATION_SIZE
    if key == 'escape': application.base.userExit()

def update_camera_and_hud():
    global _last_gen_line
    # Ursina's dt can spike huge after a heavy frame (CPU-fallback boids step)
    # — clamp it for ALL camera math so lerp can't extrapolate to inf.
    cdt = min(max(time.dt, 0.0), 0.1)
    if not np.all(np.isfinite([camera.position.x, camera.position.y, camera.position.z])):
        camera.position = Vec3(0, 35, -45)   # recover a poisoned camera
    # hide the selected hawk body in first-person
    for i, e in enumerate(predators.pack_entities):
        e.visible = not (curr_mode == 2 and i == predators.sel_idx)
    # --- speed FOV warp on the tracked hawk ---
    track = predators.sel_idx if curr_mode == 2 else predators.best_idx
    speed_ratio = max(0.0, float((predators.pack_speed[track] - CRUISE_SPEED) / max(PRED_BOOST - CRUISE_SPEED, 1e-6)))
    target_fov = BASE_FOV + (BOOST_FOV - BASE_FOV) * speed_ratio
    camera.fov = lerp(float(camera.fov), float(target_fov), cdt * 4.0)
    if curr_mode == 1:
        # pack chase: follow the current generation's best genome.
        # The offset lives in the HAWK's body frame (behind -f, above +u) and
        # the camera quaternion SLERPs toward the hawk's own attitude, so the
        # camera rides through vertical climbs, loops and rolls. No look_at:
        # its up-vector singularity used to freeze the view (then snap) on a
        # steeply-climbing hawk.
        target_pos = predators.pack_pos[predators.best_idx]
        q_best = predators.pack_q[predators.best_idx]
        off_b = np.array([0.0, 2.5, -9.0])   # body frame (r, u, f): above+behind
        off = _q_rot_b(q_best[None, :], off_b[None, :])[0]
        wanted_pos = Vec3(*(target_pos + off))
        camera.position = lerp(camera.position, wanted_pos, min(cdt * 6, 1.0))
        # keep the chase camera out of the terrain: a hawk hugging a hill puts
        # the lerped camera inside the mesh -> whole frame becomes flat rock.
        cam_ground = float(terrain.height_at(camera.position.x, camera.position.z))
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
        q = predators.pack_q[predators.sel_idx]
        u = _q_rot_b(q[None, :], np.array([0.0, 1.0, 0.0]))[0]
        camera.position = Vec3(*(predators.pack_pos[predators.sel_idx] + u * 0.3))
        camera.quaternion = PQuat(q[3], q[0], q[1], q[2])
    # HUD
    _line = (f'Gen: {predators.generation} | Time: {GENERATION_SECONDS - predators.gen_elapsed:.1f}s | '
             f'Best Kills: {int(predators.gen_kills[predators.best_idx])} | Flock Evo: {flock.starling_evolutions}')
    if _line != _last_gen_line:
        _last_gen_line = _line
        gen_text.text = _line
    if kill_counter.text != f'catches: {predators.total_catches}':
        kill_counter.text = f'catches: {predators.total_catches}'
