# -*- coding: utf-8 -*-
"""
simstep.py — one simulation step, decoupled from rendering (layer: above
all physics modules, below main.py's visual wiring).

sim_step(dt) advances the WHOLE simulation chain by dt simulated seconds:
daynight clock -> player -> boids (which drive predators.update_pack) ->
rabbits -> foxes -> wolves. It sets the shared Ursina Clock's dt because
every physics module reads time.dt from its own namespace (same Clock
instance), so the harness pattern `flock.time = Clock()` still works for
harnesses that call the step functions directly.

Visuals NEVER drive physics anymore: the Visuals classes expose render()
(mesh rebuild only) and main.py's update() calls sim_step() then the
renders. Turbo training (todo Task 6) = call sim_step(fixed_dt) K times
per rendered frame; physics/fitness advance identically per simulated
second at any K.

Order mirrors the pre-refactor live order: player physics, then the
entity-update chain flock -> rabbits -> foxes -> wolves.
"""
from ursina import time as _clock

import config
import daynight
import flock
import rabbits
import foxes
import wolves
import player

FIXED_DT = 1.0 / 60.0

# --- idle turbo state (todo Task 6 + turbo button Task 10) --------------
# turbo_active is READ by the HUD badge/button; written only by begin_frame.
# Three-way control (turbo_mode), cycled by the on-screen TURBO button:
#   auto -> turbo once idle (no key event) for turbo_idle_seconds AND
#           the camera is the fixed one (curr_mode 0)
#   on   -> turbo always, regardless of idle/camera (spectator FF)
#   off  -> never turbo
# The 't' key toggles between forced on and auto (from 'off' it goes on).
# Any keypress drops auto-turbo instantly (mode stays; 'on' keeps running).
turbo_active = False
turbo_mode = 'auto'      # 'auto' | 'on' | 'off'
_last_activity = None   # anchored on the first begin_frame (ursina clock
                        # may not be live at import)
_accum = 0.0


def note_activity():
    """Call from the input handler: any key event marks the player active."""
    global _last_activity, turbo_active
    _last_activity = _clock.time()
    if turbo_mode == 'auto':
        turbo_active = False


def cycle_turbo_mode():
    """TURBO button click: auto -> on -> off -> auto. Mouse clicks are NOT
    key events, so this deliberately does not touch the idle timer."""
    global turbo_mode
    turbo_mode = {'auto': 'on', 'on': 'off', 'off': 'auto'}[turbo_mode]


def toggle_turbo():
    """'t' key: spectator fast-forward, works in any camera, ignores idle."""
    global turbo_mode
    turbo_mode = 'on' if turbo_mode != 'on' else 'auto'


def begin_frame(wall_dt, curr_mode):
    """Called once per rendered frame BEFORE stepping. Advances the
    fixed-dt accumulator at 1x or TURBO_FACTOR x and returns the number
    of sim_step(FIXED_DT) substeps to run. Physics per simulated second
    is identical at any speed — turbo only changes wall-clock rate."""
    global _accum, turbo_active, _last_activity
    if _last_activity is None:
        _last_activity = _clock.time()   # anchor idle timing to first frame
    wall_dt = min(max(wall_dt, 0.0), 0.1)   # Ursina frame-spike clamp
    if turbo_mode == 'auto':
        idle = (_clock.time() - _last_activity) >= config.TURBO_IDLE_SECONDS
        turbo_active = idle and curr_mode == 0
    else:
        turbo_active = (turbo_mode == 'on')
    speed = float(config.TURBO_FACTOR) if turbo_active else 1.0
    _accum += wall_dt * speed
    # deficit cap: a heavy frame can never bank unbounded catch-up work
    _accum = min(_accum, config.TURBO_MAX_CATCHUP)
    n = int(_accum / FIXED_DT)
    _accum -= n * FIXED_DT
    # 1x SPIRAL GUARD: at 1x a slow frame (startup shaders, first mesh
    # rebuild) must NEVER bank multi-substep catch-up — 6 heavy steps
    # make the NEXT frame slow too, and the sim locks itself into a
    # spiral of death at a fraction of realtime. One step max, drop the
    # remainder: at 1x losing sim-time beats never catching up.
    if not turbo_active and n > 1:
        n = 1
        _accum = 0.0
    return n


def sim_step(dt):
    _clock.dt = dt
    daynight.update_daynight(dt)
    player.update_player()
    flock.update_boids_gpu()
    rabbits.update_rabbits()
    foxes.update_foxes()
    wolves.update_wolves()
