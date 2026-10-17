#!/usr/bin/env python3
"""verify_refactor.py — offscreen gate for the modular refactor.

Imports the real modules via main.py (with Ursina.run patched out and the
window offscreen), steps taskMgr frames, and exercises: boids+tribe NN,
evolve_generation + checkpoint save (incl. the starling hook), camera
input modes, respawn, HUD. Uses a TEMP save folder so the live ./evolution
state is untouched. (hawk.glb may warn "missing model" because ursina
derives asset_folder from this script's dir — harmless for physics checks.)
"""
import os, sys, tempfile, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['FORCE_CPU'] = '1'   # numpy shim, no VRAM probe
sys.argv = [sys.argv[0]]

import ursina.main

# Ursina is @singleton-decorated: ursina.main.Ursina is a proxy whose
# __new__ closure holds the REAL class. Patch the real class:
# run() -> no-op, __init__ -> offscreen + no dev-mode watchdog.
_cls = None
for cell in (getattr(ursina.main.Ursina.__new__, '__closure__', None) or ()):
    try:
        v = cell.cell_contents
    except ValueError:
        continue
    if isinstance(v, type) and v.__name__ == 'Ursina' and v.__module__ == 'ursina.main':
        _cls = v
        break
assert _cls is not None, "could not reach the real Ursina class"
_cls.run = lambda self, info=True: None
_orig_init = _cls.__init__
def _init(self, *a, **kw):
    kw['development_mode'] = False
    kw['window_type'] = 'offscreen'
    return _orig_init(self, *a, **kw)
_cls.__init__ = _init

TMP = tempfile.mkdtemp(prefix='birdsim_verify_')
import config
config.EVOLVE_DIR = TMP          # predators imports config AFTER this
import main   # noqa: E402  (runs full init incl. load_checkpoint, entities)
import predators, flock, rabbits, hud_cam, terrain   # noqa: E402

app = main.app
assert app is not None
print("STEP1: modules imported, entities built OK")

# entity sanity: terrain/backdrop/hawks/visuals exist
assert terrain.terrain is not None and terrain.backdrop is not None
assert len(predators.pack_entities) == config.POPULATION_SIZE
assert isinstance(main.flock_viz, flock.FlockVisuals)
assert isinstance(main.rabbit_viz, rabbits.RabbitVisuals)
print("STEP2: entities OK; fog:", __import__("ursina").scene.fog_density)

# step real frames through ursina's task loop (update() + entity updates)
for _ in range(60):
    app.taskMgr.step()
import numpy as np
print("STEP3: 60 frames OK; flock pos finite:",
      bool(np.all(np.isfinite(flock.pos_gpu))))

# force a generation boundary -> evolve + history + BOTH checkpoints
predators.gen_elapsed = config.GENERATION_SECONDS + 1
predators.fitness = np.linspace(1, 10, config.POPULATION_SIZE)
for _ in range(5):
    app.taskMgr.step()
print("STEP4: generation advanced to", predators.generation)
for f in ('population.npz', 'history.csv', 'starling_tribes.npz'):
    p = os.path.join(TMP, f)
    print("  ckpt:", f, "exists:", os.path.exists(p))
s = np.load(os.path.join(TMP, 'starling_tribes.npz'))
assert s['tribe_pop'].shape == flock.tribe_pop.shape
assert np.array_equal(s['tribe_pop'], flock.tribe_pop)
print("STEP5: starling checkpoint round-trip OK (shape", s['tribe_pop'].shape, ")")

# reload path: zero tribe_pop, load_checkpoint must restore it
backup = flock.tribe_pop.copy()
flock.tribe_pop = np.zeros_like(flock.tribe_pop)
predators.load_checkpoint()
assert np.array_equal(flock.tribe_pop, backup), "starling checkpoint reload failed"
flock._upload_tribe_layers()
print("STEP6: load_checkpoint -> tribe reload + _upload_tribe_layers OK")

# camera modes + input handler
for k in ('1', '2', '3', 'tab'):
    main.input(k)
for _ in range(30):
    app.taskMgr.step()
print("STEP7: camera modes 1/2/3 + tab OK; curr_mode =", hud_cam.curr_mode,
      "sel_idx =", predators.sel_idx)

# rabbit soak
for _ in range(120):
    app.taskMgr.step()
assert np.all(np.isfinite(rabbits.rabbit_pos))
print("STEP8: 120 more frames; rabbits finite, states:",
      sorted(set(rabbits.rabbit_state.tolist())))

app.userExit()
shutil.rmtree(TMP, ignore_errors=True)
print("ALL_OK")
