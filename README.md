# bird_sim

GPU/NumPy starling murmuration sim with an autonomous hawk pack driven by
neuroevolution, plus an autonomous ground-prey layer (rabbits).

Built with [Ursina](https://www.ursinaengine.org/) + CuPy (auto-falls back to
NumPy when VRAM is tight).

## What's in here

- **Starling flock** — topological boids (7 nearest neighbours) over a
  procedural fBm heightmap, with a migrating roost that sweeps the map on an
  never-closing orbit + vertical wave.
- **Flock neural nets** — one small MLP per tribe (birds split into
  contiguous blocks). Tribes co-evolve against the hawks via a Moran
  process: tribes that take sustained casualties get replaced by mutated
  copies of the current top performers.
- **Hawk pack (neuroevolution)** — a population of hawks, each flown by an
  MLP (12 observations -> 4 controls: yaw/pitch/roll rates + throttle).
  True 6-DOF quaternion flight, arena/ceiling deflection, dense shaped
  fitness (proximity ramp + dogfight alignment bonus + catches, minus
  ground/wall/ceiling scrapes). Between generations, a **3-tiered
  speciation** scheme breeds the top 3: champion lineage with tiny mutation
  (exploitation), runner-up with medium, wildcard with heavy (exploration).
- **Rabbit prey layer** — 100 rabbits on a pure-NumPy 4-state machine
  (Forage / Vigilance / Freeze / Flee) that react to the nearest hawk and
  sprint to invisible burrows, diving underground when cornered.

## Running

```bash
./run.sh          # creates a py3.12 venv via uv, exports CUDA_PATH, launches
./run.sh --cpu    # skip the CuPy probe (or set game.force_cpu in config.toml)
```

All tunables live in `config.toml` (flock, hawk, pack, rabbits, terrain,
game). Evolution checkpoints persist in the folder set by
`[pack] save_folder`.

## Controls

| Key | Action |
|-----|--------|
| 1 / 2 / 3 | fixed wide / pack-chase / first-person camera |
| TAB | cycle the tracked hawk |
| ESC | quit |

## Files / architecture

Strict bottom-up layering (each module only imports layers below it):

```
backend.py    --cpu / FORCE_CPU handling, CuPy VRAM probe, cp / np / to_cpu
config.py     config.toml -> all uppercase constants + derived values
math3d.py     pure NumPy batched quaternion math (_q_mul_b, _slerp_np, ...)
terrain.py    heightmaps + height_at at import; entities via init_environment()
predators.py  hawk pack state, neuroevolution, flight physics, checkpoints
flock.py      starling state, tribe NNs, update_boids_gpu, FlockVisuals
rabbits.py    ground-prey FSM, update_rabbits, RabbitVisuals
hud_cam.py    camera modes, HUD text, keyboard handling
main.py       entrypoint orchestrator (init order + global input/update)
```

- `main.py` imports `backend` before Ursina sees the command line (`--cpu`
  is stripped there), then does: `Ursina(vsync=False)` -> `load_checkpoint()`
  + `_upload_tribe_layers()` -> environment + `FlockVisuals` +
  `RabbitVisuals` + hawk entities + HUD -> `respawn_pack()` -> fog re-apply
  -> `app.run()`. No Ursina entity is ever created at module import time.
- `predators` <-> `flock` reference each other only through lazy imports
  inside functions (the starling checkpoint hooks in
  `predators.load/save_checkpoint` read/write `flock.tribe_pop`).
- Cross-module state is always reached via the module namespace
  (`predators.best_idx`, `flock.pos_gpu`, ...) — `from module import var`
  would go stale when evolution/updates rebind those names.
- `config.toml` — live tunables.
- `assets/hawk.glb` — hawk model (decimated GLB; Ursina can't load the raw
  STL, so the original 14 MB `hawk.stl` is gitignored).
- `tools/verify_refactor.py` — offscreen harness that imports the real
  modules, steps Ursina frames, and checks evolution, checkpoints (incl.
  the starling round-trip), cameras and rabbits end to end.
  Run: `DISPLAY=:0 .venv/bin/python tools/verify_refactor.py`
