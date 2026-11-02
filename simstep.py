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

# --- idle turbo state (todo Task 6) -------------------------------------
# turbo_active is READ by the HUD badge; written only by begin_frame.
# Three-way control per the agreed design:
#   auto   -> turbo once idle (no key event) for turbo_idle_seconds AND
#             the camera is the fixed one (curr_mode 0)
#   forced -> 't' toggle: on regardless of idle/camera (spectator FF)
#   any keypress drops auto-turbo instantly (forced survives, 't' off)
turbo_active = False
_turbo_forced = False
_last_activity = 0.0
_accum = 0.0


def note_activity():
    """Call from the input handler: any key event marks the player active."""
    global _last_activity, turbo_active
    _last_activity = _clock.time()
    if not _turbo_forced:
        turbo_active = False


def toggle_turbo():
    """'t' key: spectator fast-forward, works in any camera, ignores idle."""
    global _turbo_forced, turbo_active
    _turbo_forced = not _turbo_forced
    turbo_active = _turbo_forced
    if not _turbo_forced:
        turbo_active = False


def begin_frame(wall_dt, curr_mode):
    """Called once per rendered frame BEFORE stepping. Advances the
    fixed-dt accumulator at 1x or TURBO_FACTOR x and returns the number
    of sim_step(FIXED_DT) substeps to run. Physics per simulated second
    is identical at any speed — turbo only changes wall-clock rate."""
    global _accum, turbo_active
    wall_dt = min(max(wall_dt, 0.0), 0.1)   # Ursina frame-spike clamp
    if not _turbo_forced:
        idle = (_clock.time() - _last_activity) >= config.TURBO_IDLE_SECONDS
        turbo_active = idle and curr_mode == 0
    speed = float(config.TURBO_FACTOR) if turbo_active else 1.0
    _accum += wall_dt * speed
    # deficit cap: a heavy frame can never bank unbounded catch-up work
    _accum = min(_accum, config.TURBO_MAX_CATCHUP)
    n = int(_accum / FIXED_DT)
    _accum -= n * FIXED_DT
    return n


def sim_step(dt):
    _clock.dt = dt
    daynight.update_daynight(dt)
    player.update_player()
    flock.update_boids_gpu()
    rabbits.update_rabbits()
    foxes.update_foxes()
    wolves.update_wolves()
