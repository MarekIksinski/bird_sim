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

import daynight
import flock
import rabbits
import foxes
import wolves
import player


def sim_step(dt):
    _clock.dt = dt
    daynight.update_daynight(dt)
    player.update_player()
    flock.update_boids_gpu()
    rabbits.update_rabbits()
    foxes.update_foxes()
    wolves.update_wolves()
