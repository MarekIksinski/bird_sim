# -*- coding: utf-8 -*-
"""
daynight.py — the day/night clock + sim-time source (layer above config,
below everything that simulates).

sim_elapsed  total SIMULATED seconds. This is the only allowed time source
             for physics oscillators (roost orbit etc.) — NEVER Ursina's
             time.time(): it is epoch-scale (~1.8e9) and float32 math on it
             quantizes/frozen-freezes any sine (see rabbit_phase note in
             rabbits.py). sim_elapsed accumulates capped dt, so turbo
             substepping and normal frames advance it identically per
             simulated second.
day_phase    seconds into the current day/night cycle, [0, CYCLE_SECONDS).
day_number   completed cycles since session start (not checkpointed:
             every session starts at dawn, day 0).
light_level  0..1: smoothstep dawn ramp at cycle start, flat day,
             smoothstep dusk ramp, flat 0 night.
sun_height   placeholder elevation proxy for the future visual sun:
             +1 at noon, -1 at midnight (cos over the cycle).

Config [daynight]: day_seconds, night_seconds, transition_seconds.
"""
import math
from config import DAY_SECONDS, NIGHT_SECONDS, TRANSITION_SECONDS

CYCLE_SECONDS = DAY_SECONDS + NIGHT_SECONDS

sim_elapsed = 0.0
day_phase = 0.0
day_number = 0
light_level = 1.0
sun_height = 0.0


def _smoothstep(a, b, x):
    if b <= a:
        return 1.0 if x >= b else 0.0
    t = (x - a) / (b - a)
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _compute_light(t):
    T = TRANSITION_SECONDS
    if t < T:                            # dawn ramp 0 -> 1
        return _smoothstep(0.0, T, t)
    if t < DAY_SECONDS - T:              # full day
        return 1.0
    if t < DAY_SECONDS:                  # dusk ramp 1 -> 0
        return 1.0 - _smoothstep(DAY_SECONDS - T, DAY_SECONDS, t)
    return 0.0                           # night


def update_daynight(dt):
    """Advance the clock. dt-capped like every other sim step, so a
    framerate hitch never jumps the day."""
    global sim_elapsed, day_phase, day_number, light_level, sun_height
    dt = min(max(dt, 0.0), 0.05)
    sim_elapsed += dt
    day_phase += dt
    if day_phase >= CYCLE_SECONDS:
        day_phase -= CYCLE_SECONDS
        day_number += 1
    light_level = _compute_light(day_phase)
    # +1 at local noon (middle of the day half), -1 at midnight
    sun_height = math.cos(2.0 * math.pi * (day_phase - 0.5 * DAY_SECONDS)
                          / CYCLE_SECONDS)


def light_obs():
    """NN sensor encoding (same style as the stamina sensor):
    +1.0 high noon, -1.0 midnight — tanh-friendly."""
    return -1.0 + 2.0 * light_level


def reset_day():
    """Harness helper: rewind to dawn of day 0."""
    global sim_elapsed, day_phase, day_number, light_level, sun_height
    sim_elapsed = 0.0
    day_phase = 0.0
    day_number = 0
    light_level = _compute_light(0.0)
    sun_height = math.cos(2.0 * math.pi * (0.0 - 0.5 * DAY_SECONDS) / CYCLE_SECONDS)
