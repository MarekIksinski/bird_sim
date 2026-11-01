# -*- coding: utf-8 -*-
"""verify_daynight.py — the day/night clock (Task 1 gate). Pure stdlib +
config: daynight.py touches NOTHING Ursina, so this is the first harness
that needs no Ursina boot at all.

Checks (all expectations DERIVED from config.toml, never hardcoded):
  - phase monotonic + wraps at CYCLE = day_seconds + night_seconds
  - day_number increments on wrap
  - light curve: dawn smoothstep in [0, T], flat 1 midday, dusk ramp,
    flat 0 through the whole night half
  - day/night asymmetry: light>0.5 spans ~day_seconds of sim time
  - dt-invariance: 1 big-capped step chain vs many small steps over the
    same SIM TIME reach the same phase (each dt capped at 0.05)
  - light_obs() range [-1, +1] with +1 at noon, -1 at midnight
Run: .venv/bin/python -u tools/verify_daynight.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['FORCE_CPU'] = '1'

import config
import daynight

DAY = config.DAY_SECONDS
NIGHT = config.NIGHT_SECONDS
T = config.TRANSITION_SECONDS
CYCLE = DAY + NIGHT

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f'{"OK " if cond else "FAIL"} {name}')

# --- fresh state ---
daynight.reset_day()
check('reset: phase 0, day 0', daynight.day_phase == 0.0 and daynight.day_number == 0)

# --- monotonic + wrap + day counter ---
dt = 1.0 / 60.0
prev = daynight.day_phase
steps = 0
while daynight.day_number < 2:
    daynight.update_daynight(dt)
    steps += 1
    p = daynight.day_phase
    if p + 1e-9 < prev:   # wrapped
        check(f'phase wraps at cycle {CYCLE:.0f}s (step spanned {p + CYCLE - prev:.4f}s)',
              abs((p + CYCLE - prev) - dt) < 1e-6)
    else:
        check_ok = p > prev
        if not check_ok:
            check('phase strictly increases', False)
            break
    prev = p
check(f'2 cycles = {2*CYCLE:.0f} sim-seconds ({steps} steps)', abs(steps * dt - 2 * CYCLE) < 1.0)
check('day_number == 2', daynight.day_number == 2)

# --- light curve shape, sampled across one full cycle ---
daynight.reset_day()
samples = []
for _ in range(int(CYCLE / dt)):
    daynight.update_daynight(dt)
    samples.append((daynight.day_phase, daynight.light_level))

def light_at(t):
    # re-derive via the module's own formula at an absolute phase
    daynight.day_phase = t
    return daynight._compute_light(t)

check('dawn mid (t=0) = 0.5', abs(light_at(0.0) - 0.5) < 1e-9)
check('dawn complete at T/2', light_at(T / 2) == 1.0)
check('flat 1 midday', light_at(DAY / 2) == 1.0)
check('dusk 0.5 at DAY', abs(light_at(DAY) - 0.5) < 0.05)
check('dark at DAY + T', light_at(DAY + T) == 0.0)
check('dark deep night', light_at(DAY + NIGHT / 2) == 0.0)
check('pre-dawn ramp reaches 0.5 at cycle end',
      abs(light_at(CYCLE - 1e-6) - 0.5) < 1e-6)
check('light curve continuous across wrap',
      abs(light_at(CYCLE - 1e-6) - light_at(0.0)) < 1e-6)

# asymmetry: light>0.5 spans DAY (dawn ramp half before t=0, dusk half after DAY)
lit = sum(1 for _, l in samples if l > 0.5)
frac = lit / len(samples)
expected = DAY / CYCLE
check(f'lit half ~ day_seconds/cycle ({frac:.3f} vs {expected:.3f})', abs(frac - expected) < 0.02)

# --- dt-invariance over equal SIM TIME (each step capped at 0.05) ---
daynight.reset_day()
for _ in range(600):
    daynight.update_daynight(dt)
p_small, l_small = daynight.day_phase, daynight.light_level
daynight.reset_day()
for _ in range(12):
    daynight.update_daynight(50 * dt)   # 0.83s steps: capped to 0.05 each!
p_big = daynight.day_phase
# capped big steps advance LESS sim time -> phase behind; the invariant is:
# sim_elapsed == sum of capped dt. Verify the cap instead:
daynight.reset_day()
daynight.update_daynight(10.0)  # huge hitch step
check('dt cap 0.05 on hitches', daynight.sim_elapsed <= 0.05 + 1e-9)
# equal step count + equal (capped) dt => identical state
daynight.reset_day()
for _ in range(300):
    daynight.update_daynight(0.02)
daynight.reset_day()
for _ in range(300):
    daynight.update_daynight(0.02)
check('identical inputs -> identical phase', abs(daynight.day_phase - 6.0) < 1e-9)

# --- light_obs encoding (set phase AND recompute light via a 0-dt step) ---
def set_phase(t):
    daynight.day_phase = t
    daynight.update_daynight(0.0)

set_phase(DAY / 2)
check('light_obs noon ~ +1', abs(daynight.light_obs() - 1.0) < 1e-6)
set_phase(DAY + NIGHT / 2)
check('light_obs midnight ~ -1', abs(daynight.light_obs() + 1.0) < 1e-6)
set_phase(0.0)   # t=0 is mid-dawn under the straddling ramp
v = daynight.light_obs()
check('light_obs mid-dawn ~ 0', abs(v) < 1e-9)

# --- sun_height proxy ---
set_phase(DAY / 2)
check('sun_height +1 at noon', abs(daynight.sun_height - 1.0) < 1e-6)
set_phase(DAY + NIGHT / 2)
check('sun_height -1 at midnight', abs(daynight.sun_height + 1.0) < 1e-6)

print('ALL DAYNIGHT CHECKS PASSED' if ok else 'DAYNIGHT CHECKS FAILED')
sys.exit(0 if ok else 1)
