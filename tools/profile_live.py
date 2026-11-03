#!/usr/bin/env python3
"""profile_live.py — cProfile the REAL windowed game for N seconds.

Runs main.py in-process under cProfile with a live X window (same code
path as ./run.sh, GPU backend included), lets it render for
PROF_SECONDS (default 20), then userExits and prints the cumulative
profile. py-spy can't attach on this box (ptrace_scope=1), so this is
the live-path profiler. Run: DISPLAY=:0 .venv/bin/python -u tools/profile_live.py [seconds]
"""
import os, sys, time, signal, cProfile, pstats, io
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.argv[0] = os.path.join(ROOT, 'main.py')   # ursina derives asset_folder from argv[0]
SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

def _stop(signum, frame):
    from ursina import application as _app   # 8.3: ursina exports the INSTANCE
    try:
        _app.base.userExit()
    except Exception as e:
        print('[profile_live] stop failed:', e)

signal.signal(signal.SIGALRM, _stop)
signal.setitimer(signal.ITIMER_REAL, SECS)

pr = cProfile.Profile()
pr.enable()
try:
    with open(os.path.join(ROOT, 'main.py')) as fh:
        code = compile(fh.read(), os.path.join(ROOT, 'main.py'), 'exec')
    # ursina's _update calls __main__.update() — exec into the REAL __main__
    # module dict, otherwise the game loop never runs and we profile a
    # frozen scene (draw-only).
    import __main__ as _mm
    exec(code, _mm.__dict__)
except SystemExit:
    pass   # userExit from the alarm handler
finally:
    pr.disable()

s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(30)
print('\n=== LIVE profile (tottime, top 30) ===')
print(s.getvalue())
# update() + sim_step cumulative for the per-frame budget split
s2 = io.StringIO()
pstats.Stats(pr, stream=s2).sort_stats('cumulative').print_stats('update|sim_step|render|taskMgr|frame')
print('\n=== cumulative: frame/update/sim_step/render ===')
print(s2.getvalue())
