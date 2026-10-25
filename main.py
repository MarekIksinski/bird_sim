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
from config import _FOG_RANGE, SHADOW_BIAS, SHADOW_MAP_SIZE

from ursina import Ursina, Entity, Vec2, Vec3, scene
from ursina.lights import DirectionalLight
from ursina.shaders import lit_with_shadows_shader

import terrain
import predators
import flock
import rabbits
import foxes
import wolves
import player
import hud_cam
app = Ursina(vsync=False)

# --- advanced lighting: every entity built AFTER this point gets the
# shadow-casting lit shader by default (Text/Sky pin their own shaders in
# their constructors, so the HUD and skydome are unaffected). The terrain
# OVERRIDES this with triplanar_shader (terrain.init_environment) and
# re-adds the shadow inputs there. ---
Entity.default_shader = lit_with_shadows_shader

# --- dynamic shadow-casting sun. look_at() re-fits the shadow lens bounds
# to the scene; terrain.init_environment() calls sun.update_bounds() again
# once every entity exists so the lens covers the whole arena. shadow_bias
# is raised from the shader default (0.001) to kill shadow acne on the
# low-poly hills — patched into the shader's default_input BEFORE any
# entity is built, because ursina copies default_input onto each entity at
# shader-assignment time (a later render-level input would be overridden).
# ---
lit_with_shadows_shader.default_input['shadow_bias'] = SHADOW_BIAS
terrain.triplanar_terrain_shader.default_input['shadow_bias'] = SHADOW_BIAS
sun = DirectionalLight(shadow_map_resolution=Vec2(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE),
                       shadows=True)
sun.look_at(Vec3(-1, -1.2, -1))

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
# the triplanar terrain is NOT touched by the fog setter's default_input
# check unless its shader declares fog_start/fog_end — it does (terrain.py),
# but re-assert here so it can never keep the stock (10,100) defaults.
terrain.terrain.set_shader_input('fog_start', _FOG_RANGE[0])
terrain.terrain.set_shader_input('fog_end', _FOG_RANGE[1])

# Re-fit the sun's shadow lens now that the whole scene (terrain, props,
# predators) exists — look_at() above fitted it to a near-empty scene.
sun.update_bounds()

# --- Ursina global callbacks (delegated; Ursina polls these by name) ---
def input(key):
    hud_cam.handle_input(key)

def update():
    player.update_player()
    hud_cam.update_camera_and_hud()

app.run()
#EditorCamera
#app.run()
