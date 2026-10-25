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
                    GENERATION_SECONDS, CRUISE_SPEED, PRED_BOOST,
                    COLOR_HUD_INFO, COLOR_HUD_GEN, COLOR_HUD_KILLS)
import config
from math3d import _slerp_np, _q_rot_b
import terrain
import predators
import flock
import foxes
import wolves
import player

hawk_info = None
gen_text = None
kill_counter = None
_last_gen_line = ''

# --- 5. CAMERA ---
# config start_camera: 1 fixed wide, 2 pack chase (best hawk), 3 first-person
# (selected), 4 player chase cam, 5 player first-person cam,
# 6 fox chase cam, 7 fox first-person cam,
# 8 wolf chase cam, 9 wolf first-person cam
curr_mode = max(0, min(8, START_CAMERA - 1))

def init_hud_and_camera():
    """Create the HUD Text objects and set the initial camera pose.
    Call only AFTER app = Ursina(...) exists."""
    global hawk_info, gen_text, kill_counter
    hawk_info = Text('1-3 Hawk cams  4/5 Player cams  6/7 Fox cams  8/9 Wolf cams  WASD/QE/Shift/Ctrl fly  TAB cycle  ESC quit',
                     origin=(0, 0), y=-0.45, size=0.035, color=color.rgba32(*COLOR_HUD_INFO))
    gen_text = Text('Gen: 1 | Time: 0.0s | Best Kills: 0', origin=(0, 0), y=-0.38, size=0.045,
                    color=color.rgba32(*COLOR_HUD_GEN))
    kill_counter = Text('catches: 0', origin=(0, 0), x=-0.9, y=-0.45, size=0.05,
                        color=color.rgba32(*COLOR_HUD_KILLS))
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
    if key == '4': curr_mode = 3   # player chase cam
    if key == '5': curr_mode = 4   # player first-person cam
    if key == '6': curr_mode = 5   # fox chase cam
    if key == '7': curr_mode = 6   # fox first-person cam
    if key == '8': curr_mode = 7   # wolf chase cam
    if key == '9': curr_mode = 8   # wolf first-person cam
    if key in config.PLAYER_KEYS['reset']:
        player.reset_player()
    if key == 'tab':
        # context-aware: in the fox cams TAB cycles the tracked fox,
        # in the wolf cams the tracked wolf, everywhere else the AI hawk
        if curr_mode in (5, 6):
            foxes.sel_fox_idx = (foxes.sel_fox_idx + 1) % config.NUM_FOXES
            foxes.sel_used = True
        elif curr_mode in (7, 8):
            wolves.sel_wolf_idx = (wolves.sel_wolf_idx + 1) % config.NUM_WOLVES
            wolves.sel_used = True
        else:
            predators.sel_idx = (predators.sel_idx + 1) % POPULATION_SIZE
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
    # hide the player body in the player first-person cam
    if player.player_entity is not None:
        player.player_entity.visible = (curr_mode != 4)
    # the fox tracked by the fox cams (mode 5: best, or TAB-cycled once used;
    # mode 6: always the TAB-cycled fox)
    if curr_mode == 6:
        _fox_trk = foxes.sel_fox_idx
    elif curr_mode == 5:
        _fox_trk = foxes.sel_fox_idx if foxes.sel_used else foxes.best_fox_idx
    else:
        _fox_trk = foxes.best_fox_idx
    # the wolf tracked by the wolf cams (mode 7: best, or TAB-cycled once
    # used; mode 8: always the TAB-cycled wolf) — mirrors the fox cams
    if curr_mode == 8:
        _wolf_trk = wolves.sel_wolf_idx
    elif curr_mode == 7:
        _wolf_trk = wolves.sel_wolf_idx if wolves.sel_used else wolves.best_wolf_idx
    else:
        _wolf_trk = wolves.best_wolf_idx
    # --- speed FOV warp on the tracked hawk/fox/wolf ---
    if curr_mode in (3, 4):
        _spd = player.player_speed
        _cruise, _boost = config.PLAYER_CRUISE_SPEED, config.PLAYER_BOOST_SPEED
    elif curr_mode in (5, 6):
        _spd = float(foxes.fox_speed[_fox_trk])
        _cruise, _boost = config.FOX_CRUISE_SPEED, config.FOX_BOOST_SPEED
    elif curr_mode in (7, 8):
        _spd = float(wolves.wolf_speed[_wolf_trk])
        _cruise, _boost = config.WOLF_CRUISE_SPEED, config.WOLF_BOOST_SPEED
    else:
        track = predators.sel_idx if curr_mode == 2 else predators.best_idx
        _spd = float(predators.pack_speed[track])
        _cruise, _boost = CRUISE_SPEED, PRED_BOOST
    # speed_ratio scales 0 at cruise -> 1 at boost -> up to 1.35 in a steep
    # stoop (pack/player speed can now exceed PRED_BOOST up to DIVE_MAX_SPEED):
    # dives get an extra FOV rush without blowing out the projection.
    speed_ratio = min(1.35, max(0.0, (_spd - _cruise) / max(_boost - _cruise, 1e-6)))
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
            camera.position = Vec3(camera.position.x, cam_ground + 2.5, camera.position.z)
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
    elif curr_mode == 3:
        # player chase cam: identical to the pack chase cam (mode 1) but
        # tracking the player hawk — body-frame offset, terrain clearance
        # clamp, and quaternion SLERPs (no look_at -> no up-singularity).
        target_pos = player.player_pos
        q_best = player.player_q
        off_b = np.array([0.0, 2.5, -9.0])   # body frame (r, u, f): above+behind
        off = _q_rot_b(q_best[None, :], off_b[None, :])[0]
        wanted_pos = Vec3(*(target_pos + off))
        camera.position = lerp(camera.position, wanted_pos, min(cdt * 6, 1.0))
        # keep the chase camera out of the terrain (reassign the WHOLE Vec3:
        # camera.position returns a fresh copy, so .y = would be lost)
        cam_ground = float(terrain.height_at(camera.position.x, camera.position.z))
        if camera.position.y < cam_ground + 2.5:
            camera.position = Vec3(camera.position.x, cam_ground + 2.5, camera.position.z)
        q_tgt = np.array([q_best[0], q_best[1], q_best[2], q_best[3]])  # (x,y,z,w)
        cq = camera.quaternion
        q_cur = np.array([cq[1], cq[2], cq[3], cq[0]])                  # (w,x,y,z) -> (x,y,z,w)
        q_out = _slerp_np(q_cur, q_tgt, min(cdt * 4.0, 1.0))
        camera.quaternion = PQuat(q_out[3], q_out[0], q_out[1], q_out[2])
    elif curr_mode == 4:
        # player first-person cam (body hidden above)
        q = player.player_q
        u = _q_rot_b(q[None, :], np.array([0.0, 1.0, 0.0]))[0]
        camera.position = Vec3(*(player.player_pos + u * 0.3))
        camera.quaternion = PQuat(q[3], q[0], q[1], q[2])
    elif curr_mode == 5:
        # fox chase cam: same shape as mode 3 but tracking a ground fox with
        # the terrain-slope-tilted quat from foxes.fox_quat()
        target_pos = foxes.fox_pos[_fox_trk].copy()
        target_pos[1] += (abs(np.sin(foxes.fox_hop_phase[_fox_trk]))
                          * min(float(foxes.fox_speed[_fox_trk]) / foxes.FOX_BOOST_SPEED, 1.0) * 0.5)
        q_fox = foxes.fox_quat(_fox_trk)
        off_b = np.array([0.0, 1.8, -5.5])   # body frame (r, u, f): above+behind
        off = _q_rot_b(q_fox[None, :], off_b[None, :])[0]
        wanted_pos = Vec3(*(target_pos + off))
        camera.position = lerp(camera.position, wanted_pos, min(cdt * 6, 1.0))
        # keep the chase camera out of the terrain (reassign the WHOLE
        # Vec3: camera.position returns a fresh copy, so .y = would be lost)
        cam_ground = float(terrain.height_at(camera.position.x, camera.position.z))
        if camera.position.y < cam_ground + 1.2:
            camera.position = Vec3(camera.position.x, cam_ground + 1.2, camera.position.z)
        cq = camera.quaternion
        q_cur = np.array([cq[1], cq[2], cq[3], cq[0]])                  # (w,x,y,z) -> (x,y,z,w)
        q_out = _slerp_np(q_cur, q_fox, min(cdt * 4.0, 1.0))
        camera.quaternion = PQuat(q_out[3], q_out[0], q_out[1], q_out[2])
    elif curr_mode == 6:
        # fox first-person cam: sits 0.9m AHEAD of the nose tip (dart nose is
        # at z=0.85) so the camera never clips inside the batched fox mesh
        pos = foxes.fox_pos[_fox_trk].copy()
        pos[1] += (abs(np.sin(foxes.fox_hop_phase[_fox_trk]))
                   * min(float(foxes.fox_speed[_fox_trk]) / foxes.FOX_BOOST_SPEED, 1.0) * 0.5)
        q_fox = foxes.fox_quat(_fox_trk)
        fwd = _q_rot_b(q_fox[None, :], np.array([0.0, 0.0, 1.0]))[0]
        up = _q_rot_b(q_fox[None, :], np.array([0.0, 1.0, 0.0]))[0]
        camera.position = Vec3(*(pos + fwd * 0.9 + up * 0.45))
        cq = camera.quaternion
        q_cur = np.array([cq[1], cq[2], cq[3], cq[0]])                  # (w,x,y,z) -> (x,y,z,w)
        q_out = _slerp_np(q_cur, q_fox, min(cdt * 10.0, 1.0))
        camera.quaternion = PQuat(q_out[3], q_out[0], q_out[1], q_out[2])
    elif curr_mode == 7:
        # wolf chase cam: same shape as the fox chase cam (mode 5) but
        # tracking a ground wolf with the terrain-slope-tilted wolf_quat()
        target_pos = wolves.wolf_pos[_wolf_trk].copy()
        target_pos[1] += (abs(np.sin(wolves.wolf_hop_phase[_wolf_trk]))
                          * min(float(wolves.wolf_speed[_wolf_trk]) / wolves.WOLF_BOOST_SPEED, 1.0) * 0.6)
        q_wolf = wolves.wolf_quat(_wolf_trk)
        off_b = np.array([0.0, 2.2, -6.5])   # body frame (r, u, f): above+behind
        off = _q_rot_b(q_wolf[None, :], off_b[None, :])[0]
        wanted_pos = Vec3(*(target_pos + off))
        camera.position = lerp(camera.position, wanted_pos, min(cdt * 6, 1.0))
        # keep the chase camera out of the terrain (reassign the WHOLE
        # Vec3: camera.position returns a fresh copy, so .y = would be lost)
        cam_ground = float(terrain.height_at(camera.position.x, camera.position.z))
        if camera.position.y < cam_ground + 1.4:
            camera.position = Vec3(camera.position.x, cam_ground + 1.4, camera.position.z)
        cq = camera.quaternion
        q_cur = np.array([cq[1], cq[2], cq[3], cq[0]])                  # (w,x,y,z) -> (x,y,z,w)
        q_out = _slerp_np(q_cur, q_wolf, min(cdt * 4.0, 1.0))
        camera.quaternion = PQuat(q_out[3], q_out[0], q_out[1], q_out[2])
    elif curr_mode == 8:
        # wolf first-person cam: sits 1.4m AHEAD of the nose tip (dart nose
        # is at z=1.3) so the camera never clips inside the batched wolf mesh
        pos = wolves.wolf_pos[_wolf_trk].copy()
        pos[1] += (abs(np.sin(wolves.wolf_hop_phase[_wolf_trk]))
                   * min(float(wolves.wolf_speed[_wolf_trk]) / wolves.WOLF_BOOST_SPEED, 1.0) * 0.6)
        q_wolf = wolves.wolf_quat(_wolf_trk)
        fwd = _q_rot_b(q_wolf[None, :], np.array([0.0, 0.0, 1.0]))[0]
        up = _q_rot_b(q_wolf[None, :], np.array([0.0, 1.0, 0.0]))[0]
        camera.position = Vec3(*(pos + fwd * 1.4 + up * 0.55))
        cq = camera.quaternion
        q_cur = np.array([cq[1], cq[2], cq[3], cq[0]])                  # (w,x,y,z) -> (x,y,z,w)
        q_out = _slerp_np(q_cur, q_wolf, min(cdt * 10.0, 1.0))
        camera.quaternion = PQuat(q_out[3], q_out[0], q_out[1], q_out[2])
    # HUD
    # stamina readout for the currently tracked hawk: player in modes 3/4,
    # otherwise the chase/FP AI hawk (sel_idx in FP mode 2, best hawk in 0/1).
    # [COOL] flags the exhaustion lockout (boost blocked until recharged).
    if curr_mode in (3, 4):
        _stam, _cool = float(player.player_stamina), player.player_exhausted
    elif curr_mode in (5, 6):
        _stam = float(foxes.fox_stamina[_fox_trk])
        _cool = bool(foxes.fox_exhausted[_fox_trk])
    elif curr_mode in (7, 8):
        _stam = float(wolves.wolf_stamina[_wolf_trk])
        _cool = bool(wolves.wolf_exhausted[_wolf_trk])
    else:
        _trk = predators.sel_idx if curr_mode == 2 else predators.best_idx
        _stam, _cool = float(predators.pack_stamina[_trk]), bool(predators.pack_exhausted[_trk])
    _line = (f'Gen: {predators.generation} | Time: {GENERATION_SECONDS - predators.gen_elapsed:.1f}s | '
             f'Best Kills: {int(predators.gen_kills[predators.best_idx])} | Flock Evo: {flock.starling_evolutions} | '
             f'Fox Evo: {foxes.fox_evolutions} | Wolf Evo: {wolves.wolf_evolutions} | '
             f'Stamina: {int(round(_stam * 100))}%' + (' [COOL]' if _cool else ''))
    if _line != _last_gen_line:
        _last_gen_line = _line
        gen_text.text = _line
    _catch_line = (f'catches: {predators.total_catches} (you: {player.player_catches}) '
                   f'| foxes: {foxes.fox_total_catches} '
                   f'| wolves: {wolves.wolf_fox_catches}f+{wolves.wolf_rabbit_catches}r '
                   f'(coop: {wolves.wolf_coop_kills})')
    if kill_counter.text != _catch_line:
        kill_counter.text = _catch_line
