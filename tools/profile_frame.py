#!/usr/bin/env python3
"""profile_frame.py — where does the frame budget actually go?

Boots the REAL pipeline (sim_step + the full render pass from main.py)
offscreen and times every stage over N frames:
  physics: daynight / player / boids / rabbits / foxes / wolves
  render:  player / flock / rabbit / fox / wolf visuals / pack / hud
Then runs cProfile over the same loop and prints the top cumulative
functions. GPU path by default (same as live ./run.sh); FORCE_CPU=1
mirrors the CPU fallback. Run: .venv/bin/python -u tools/profile_frame.py [N]
"""
import os, sys, time as _t, cProfile, pstats, io
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

import numpy as np
from ursina import Ursina
app = Ursina(window_type='offscreen')
from ursina.lights import DirectionalLight
from ursina import Vec2, scene
import config, terrain, predators, flock, rabbits, foxes, wolves, player, hud_cam, simstep, daynight

terrain.init_environment()
flock_viz = flock.FlockVisuals()
rabbit_viz = rabbits.RabbitVisuals()
fox_viz = foxes.FoxVisuals()
wolf_viz = wolves.WolfVisuals()
predators.init_pack_visuals()
player.init_player_visuals()
hud_cam.init_hud_and_camera()
predators.respawn_pack()

lit = __import__('ursina.shaders', fromlist=['lit_with_shadows_shader']).lit_with_shadows_shader
sun = DirectionalLight(shadow_map_resolution=Vec2(config.SHADOW_MAP_SIZE, config.SHADOW_MAP_SIZE), shadows=True)
sun.look_at(Vec3 := __import__('ursina').Vec3(-1, -1.2, -1))
terrain.backdrop.enabled = False
sun.update_bounds()
terrain.backdrop.enabled = True

def physics():
    simstep.sim_step(simstep.FIXED_DT)

def render():
    player.update_player_visuals()
    flock_viz.render()
    rabbit_viz.render()
    fox_viz.render()
    wolf_viz.render()
    predators.update_pack_visuals()
    hud_cam.update_camera_and_hud()

# warmup
for _ in range(5):
    physics(); render()

# --- per-stage timing ---
import simstep as _s
stages = {}
def stage(name, fn):
    t0 = _t.perf_counter()
    fn()
    stages.setdefault(name, []).append(_t.perf_counter() - t0)

# finer physics split: re-time the chain pieces directly
import types
def phys_parts():
    stage('daynight', lambda: daynight.update_daynight(simstep.FIXED_DT))
    stage('player', lambda: player.update_player())
    stage('boids', lambda: flock.update_boids_gpu())
    stage('rabbits', lambda: rabbits.update_rabbits())
    stage('foxes', lambda: foxes.update_foxes())
    stage('wolves', lambda: wolves.update_wolves())

def render_parts():
    stage('vis_player', lambda: player.update_player_visuals())
    stage('vis_flock', lambda: flock_viz.render())
    stage('vis_rabbit', lambda: rabbit_viz.render())
    stage('vis_fox', lambda: fox_viz.render())
    stage('vis_wolf', lambda: wolf_viz.render())
    stage('vis_pack', lambda: predators.update_pack_visuals())
    stage('hud', lambda: hud_cam.update_camera_and_hud())

for _ in range(N):
    phys_parts()
    render_parts()

print(f'\n=== per-stage mean ms over {N} frames ===')
total = 0.0
for k, v in sorted(stages.items(), key=lambda kv: -np.mean(kv[1])):
    m = np.mean(v) * 1000
    total += m
    print(f'{k:12s} {m:8.2f} ms')
print(f'{"TOTAL":12s} {total:8.2f} ms   -> {1000/total if total else 0:.1f} fps budget')

# --- cProfile over the full chain ---
pr = cProfile.Profile()
pr.enable()
for _ in range(N):
    simstep.sim_step(simstep.FIXED_DT)
    render()
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(35)
print('\n=== cProfile top 35 (cumulative) ===')
print(s.getvalue())
