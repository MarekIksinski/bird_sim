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

## Files

- `main.py` — thin entrypoint orchestrator (Ursina init order + global
  `input`/`update` callbacks + fog re-apply).
- `backend.py` / `config.py` / `math3d.py` — CPU/GPU backend probe,
  config.toml constants, batched quaternion math.
- `terrain.py` / `predators.py` / `flock.py` / `rabbits.py` / `hud_cam.py`
  — terrain + backdrop, hawk pack NN + speciation + checkpoints, starling
  boids + tribe NNs + FlockVisuals, rabbit FSM + RabbitVisuals, camera/HUD.
- `config.toml` — live tunables.
- `assets/hawk.glb` — hawk model (decimated GLB; Ursina can't load the raw
  STL, so the original 14 MB `hawk.stl` is gitignored).

Import order matters: `main.py` imports `backend` before Ursina sees the
command line (`--cpu` is stripped there), and all Ursina entities are built
inside `init_*()` functions / classes instantiated after `Ursina()`.
Cross-module state is always reached via the module namespace
(`predators.best_idx`, `flock.pos_gpu`, ...) — `from module import var`
would go stale when evolution rebinds those names.
