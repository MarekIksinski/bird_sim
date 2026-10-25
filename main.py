#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 22:43:26 2025

@author: marek

main.py — entrypoint orchestrator. Module hierarchy (bottom -> top):
    backend.py   --cpu/FORCE_CPU handling, CuPy probe, cp/np/to_cpu
    config.py    config.toml -> all uppercase constants
    math3d.py    pure NumPy batched quaternion utilities
    terrain.py   heightmaps + height_at (import-time), entities on init
    predators.py hawk pack state, neuroevolution, flight, checkpoints
    flock.py     starling state, tribe NNs, boids step, FlockVisuals
    rabbits.py   ground prey FSM + RabbitVisuals
    foxes.py     ground predator pairs, Moran pair evolution, FoxVisuals
    wolves.py    apex predator pairs, howl channel, pack-share hunting, WolfVisuals
    hud_cam.py   camera modes, HUD text, keyboard

State is ALWAYS reached through the module namespace (predators.best_idx,
flock.pos_gpu, ...) — several of these names are REBOUND during updates and
evolution, so `from module import var` would silently go stale.
"""
# backend first: strips --cpu from sys.argv and runs the CuPy VRAM probe
# BEFORE Ursina sees the command line or anything allocates arrays.
import backend
from config import _FOG_RANGE

from ursina import Ursina, scene

import terrain
import predators
import flock
import rabbits
import foxes
import wolves
import player
import hud_cam

app = Ursina(vsync=False)

# --- persistence: resume the evolved pack + starling tribes + fox pairs
# + wolf pairs, then rebuild the cp-resident tribe layer tensors from
# whatever was loaded ---
predators.load_checkpoint()
flock._upload_tribe_layers()
foxes._upload_pair_layers()
wolves._upload_pair_layers()

# --- environment + entities (all Ursina object creation happens here,
# after Ursina() exists) ---
terrain.init_environment()
flock_viz = flock.FlockVisuals()
rabbit_viz = rabbits.RabbitVisuals()
fox_viz = foxes.FoxVisuals()
wolf_viz = wolves.WolfVisuals()
predators.init_pack_visuals()
player.init_player_visuals()
hud_cam.init_hud_and_camera()

# initial drop: hawks launch from the edge facing the roost
predators.respawn_pack()

# Re-apply fog now that EVERY entity exists (setter only touches existing
# entities), then give the backdrop its thinner haze so the ring stays visible.
scene.fog_density = _FOG_RANGE
terrain.backdrop.set_shader_input('fog_start', 800)
terrain.backdrop.set_shader_input('fog_end', 5000)

# --- Ursina global callbacks (delegated; Ursina polls these by name) ---
def input(key):
    hud_cam.handle_input(key)

def update():
    player.update_player()
    hud_cam.update_camera_and_hud()

app.run()
#EditorCamera
#app.run()
