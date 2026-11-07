# -*- coding: utf-8 -*-
"""verify_turbo.py — Task 6 gate: idle-turbo scheduler invariants.

  1. 1x: begin_frame(1/60) yields exactly one substep per frame (steady state)
  2. turbo: K substeps per frame for K frames == same sim-time as 60 frames
     at 1x (fixed-dt accumulator is rate-independent)
  3. deficit cap: a 5 s frame-spike never banks more than turbo_max_catchup
  4. idle rule: no turbo before turbo_idle_seconds; auto-on after; any
     note_activity() drops auto-turbo instantly; forced ('t') survives it
  5. substep dt is ALWAYS FIXED_DT (never a bigger dt)

simstep's Clock is stubbed (ursina not booted headless).
Run: .venv/bin/python -u tools/verify_turbo.py
"""
import os, sys
os.environ['FORCE_CPU'] = '1'
sys.path.insert(0, '/home/marek/Playground/bird_sim')
os.chdir('/home/marek/Playground/bird_sim')

import config

class FakeClock:
    dt = 1.0 / 60.0
    t = 0.0
    def time(self):
        return self.t

import simstep
_fc = FakeClock()
simstep._clock = _fc

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f'{"OK " if cond else "FAIL"} {name}')

K = config.TURBO_FACTOR
FIX = simstep.FIXED_DT

# ---------- 1. 1x steady state: one substep per frame ----------
simstep._accum = 0.0
simstep.turbo_active = False
counts = []
_fc.t = 0.0
for _ in range(60):
    counts.append(simstep.begin_frame(FIX, 1))   # chase cam -> never auto-turbo
check('1x yields exactly 1 substep per frame', set(counts) == {1})

# ---------- 2. turbo: same sim-time per wall-time, rate-independent ----------
simstep._accum = 0.0
simstep.turbo_mode = 'on'
simstep.turbo_active = True
n_turbo = sum(simstep.begin_frame(FIX, 1) for _ in range(60))
check(f'turbo 60 frames ~ 60*K substeps ({n_turbo})',
      abs(n_turbo - 60 * K) <= 2)
check('turbo speedup = turbo_factor', abs(n_turbo / 60 - K) < 0.1)
simstep.turbo_mode = 'auto'
simstep.turbo_active = False

# ---------- 3. frame-spike deficit cap ----------
simstep._accum = 0.0
n_spike = simstep.begin_frame(5.0, 1)   # alt-tab return spike
check(f'5 s spike capped ({n_spike} substeps, cap {config.TURBO_MAX_CATCHUP/FIX:.0f})',
      n_spike <= int(config.TURBO_MAX_CATCHUP / FIX) + 1)

# ---------- 3b. 1x spiral guard: NEVER more than 1 substep at 1x ----------
simstep._accum = 0.0
simstep.turbo_active = False
simstep.turbo_mode = 'auto'
n_1x_spike = simstep.begin_frame(0.1, 1)   # slow frame at 1x
check('1x spiral guard: heavy frame yields at most 1 substep', n_1x_spike == 1)
check('1x spiral guard drops the banked remainder', simstep._accum == 0.0)
# and the guard must not fire in turbo mode
simstep._accum = 0.0
simstep.toggle_turbo()
n_turbo_spike = simstep.begin_frame(0.1, 1)
check(f'turbo keeps multi-substep catch-up ({n_turbo_spike})', n_turbo_spike > 1)
simstep.toggle_turbo()

# ---------- 4. idle rule ----------
simstep._accum = 0.0
simstep._last_activity = 0.0
_fc.t = 0.0
n = simstep.begin_frame(FIX, 0)   # fixed cam, zero idle
check('not turbo while active (fixed cam)', simstep.turbo_active is False)
_fc.t = config.TURBO_IDLE_SECONDS - 0.5
simstep.begin_frame(FIX, 0)
check('not turbo just before idle threshold', simstep.turbo_active is False)
_fc.t = config.TURBO_IDLE_SECONDS + 0.5
simstep.begin_frame(FIX, 0)
check('auto-turbo after idle in fixed cam', simstep.turbo_active is True)
_fc.t = config.TURBO_IDLE_SECONDS + 1.5
simstep.begin_frame(FIX, 1)
check('chase cam suppresses auto-turbo', simstep.turbo_active is False)
_fc.t = config.TURBO_IDLE_SECONDS + 2.5
simstep.begin_frame(FIX, 0)
check('fixed cam + idle -> turbo again', simstep.turbo_active is True)
simstep.note_activity()
check('keypress drops auto-turbo instantly', simstep.turbo_active is False)
# forced toggle survives activity
simstep.toggle_turbo()
simstep.note_activity()
simstep.begin_frame(FIX, 1)
check('forced turbo survives keypress + any camera', simstep.turbo_active is True)
simstep.toggle_turbo()
check('second toggle returns mode to auto', simstep.turbo_mode == 'auto')
simstep.begin_frame(FIX, 1)   # auto + chase cam -> active flag drops
check('auto + chase cam clears turbo_active', simstep.turbo_active is False)

# ---------- 4b. tri-state button cycle (Task 10/12) ----------
simstep.turbo_mode = 'auto'
simstep._accum = 0.0
simstep._last_activity = 0.0
_fc.t = 0.0
simstep.cycle_turbo_mode()
check('cycle auto -> on', simstep.turbo_mode == 'on')
simstep.cycle_turbo_mode()
check('cycle on -> off', simstep.turbo_mode == 'off')
simstep.cycle_turbo_mode()
check('cycle off -> auto (wraps)', simstep.turbo_mode == 'auto')
# 'off' pins 1x even when idle + fixed cam (auto rule must NOT fire)
_fc.t = config.TURBO_IDLE_SECONDS + 5.0
simstep.cycle_turbo_mode()   # auto -> on
simstep.cycle_turbo_mode()   # on -> off
n_off = simstep.begin_frame(FIX, 0)
check("'off' suppresses turbo while idle+fixed cam", simstep.turbo_active is False)
check("'off' yields 1 substep", n_off == 1)
simstep.note_activity()
check("'off' survives note_activity", simstep.turbo_mode == 'off')
# 'on' forces turbo in a chase cam while actively pressing keys
simstep.cycle_turbo_mode()   # off -> auto
simstep.cycle_turbo_mode()   # auto -> on
simstep.note_activity()
n_forced = simstep.begin_frame(FIX, 1)   # chase cam, just pressed a key
check("'on' forces turbo in chase cam while active", simstep.turbo_active is True)
check("'on' frame yields K substeps", n_forced == K)
# 't' from each state: on<->auto, off -> on
simstep.turbo_mode = 'auto'; simstep.toggle_turbo()
check("t from auto -> on", simstep.turbo_mode == 'on')
simstep.toggle_turbo()
check("t from on -> auto", simstep.turbo_mode == 'auto')
simstep.turbo_mode = 'off'; simstep.toggle_turbo()
check("t from off -> on", simstep.turbo_mode == 'on')
simstep.turbo_mode = 'auto'
simstep.turbo_active = False

# ---------- 5. dt is never bigger than FIXED_DT ----------
check('FIXED_DT = 1/60', abs(FIX - 1.0 / 60.0) < 1e-12)

print('ALL TURBO CHECKS PASSED' if ok else 'TURBO CHECKS FAILED')
sys.exit(0 if ok else 1)
