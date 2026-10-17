# bird_sim

GPU/NumPy starling murmuration sim with an autonomous hawk pack driven by
neuroevolution, plus an autonomous ground-prey layer (rabbits).

Built with [Ursina](https://www.ursinaengine.org/) + CuPy (auto-falls back to
NumPy when VRAM is tight).

## What's in here

- **Starling flock** — topological boids (7 nearest neighbours) over a
  procedural fBm heightmap, with a migrating roost that sweeps the map on an
  never-closing orbit + vertical wave. Pitch-gravity energy exchange: diving
  birds accelerate along their path (dynamic speed cap up to 1.35x), climbing
  birds bleed speed (down to 0.65x), and a minimum airspeed floor keeps every
  bird flying — no hovering mid-air stalls.
- **Flock neural nets** — one small MLP per tribe (birds split into
  contiguous blocks; 17 observations incl. airspeed + vertical pitch so
  tribes can sense and exploit gravity). Tribes co-evolve against the hawks
  via a Moran process: tribes that take sustained casualties get replaced by
  mutated copies of the current top performers.
- **Hawk pack (neuroevolution)** — a population of hawks, each flown by an
  MLP (14 observations -> 4 controls: yaw/pitch/roll rates + throttle;
  observations include airspeed and nose pitch). Aerodynamic gravity trades
  altitude for airspeed: a stoop accelerates past boost toward the dive
  ceiling, a climb bleeds speed, and a nose-up hawk below the stall speed
  drops its nose back toward the horizon. The flight path also **drifts**:
  velocity lags behind the nose (speed-modulated lift bite), so high-speed
  dives carve sweeping arcs, and sharp high-AoA turns bleed airspeed via
  induced drag.
  True 6-DOF quaternion flight, arena/ceiling deflection, dense shaped
  fitness (proximity ramp + dogfight alignment bonus + catches, minus
  ground/wall/ceiling scrapes). Between generations, a **3-tiered
  speciation** scheme breeds the top 3: champion lineage with tiny mutation
  (exploitation), runner-up with medium, wildcard with heavy (exploration).
- **Rabbit prey layer** — 100 rabbits on a pure-NumPy 4-state machine
  (Forage / Vigilance / Freeze / Flee) that react to the nearest hawk and
  sprint to invisible burrows, diving underground when cornered.
- **Spectator hawk (player)** — a keyboard-piloted hawk flying the same
  6-DOF quaternion physics, wall/ceiling deflection, gravity stoop/climb
  energy exchange, stall nose-drop and ground clearance as the AI pack, but
  a pure ghost: starlings and rabbits never see it, it cannot catch birds,
  and generation resets never touch it. Its flight feel (speeds, rates, key
  bindings) is configured separately in `[player]`.

## Running

```bash
./run.sh          # creates a py3.12 venv via uv, exports CUDA_PATH, launches
./run.sh --cpu    # skip the CuPy probe (or set game.force_cpu in config.toml)
```

All tunables live in `config.toml` (flock, hawk, player, pack, rabbits,
terrain, game). Gravity/dive knobs: `[hawk]` `gravity`, `dive_max_speed`,
`min_flight_speed`, `stall_speed`, `lift_grip`, `induced_drag` (all shared
by the player) and `[flock]`
`gravity`, `min_bird_speed`, `dive_speed_mult`, `climb_speed_mult`.
The `[player]` section overrides the spectator hawk's
speeds/rates (falling back to `[hawk]` values when a line is removed) and
its key bindings (`<action>_keys` = a key name or a list of them).
Evolution checkpoints persist in the folder set by `[pack] save_folder`.

## Controls

Camera modes: `1` fixed wide, `2` pack-chase (best hawk), `3` AI
first-person, `4` player chase, `5` player first-person.

| Key | Action |
|-----|--------|
| 1 / 2 / 3 / 4 / 5 | switch camera mode (see above) |
| TAB | cycle the tracked AI hawk |
| R | respawn the player hawk (remappable: `[player] reset_keys`) |
| ESC | quit |

Player flight (default bindings, all remappable in `[player]`):

| Key | Action |
|-----|--------|
| W / S | pitch nose down / up |
| A / D | roll left / right (banks into a coordinated turn) |
| Q / E | yaw left / right |
| Shift / Ctrl | throttle boost / slow (release = cruise) |

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
player.py     spectator hawk state + keyboard flight (ghost; never feeds
              predators/flock/rabbits)
hud_cam.py    camera modes, HUD text, keyboard handling
main.py       entrypoint orchestrator (init order + global input/update)
```

- `main.py` imports `backend` before Ursina sees the command line (`--cpu`
  is stripped there), then does: `Ursina(vsync=False)` -> `load_checkpoint()`
  + `_upload_tribe_layers()` -> environment + `FlockVisuals` +
  `RabbitVisuals` + hawk entities + player hawk + HUD -> `respawn_pack()`
  -> fog re-apply -> `app.run()`. No Ursina entity is ever created at module
  import time.
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
- `tools/verify_player.py` — offscreen harness for the spectator hawk:
  spawn, straight flight, pitch/roll signs, throttle, wall/ceiling/ground
  clamps, dt-spike guard, camera modes 1-5, respawn, and render-truth
  quaternion sync. Run: `DISPLAY=:0 .venv/bin/python tools/verify_player.py`
- `tools/verify_player_cfg.py` — checks the `[player]` config section:
  speed/rate overrides, `[hawk]` fallbacks, and string/list key bindings.
  Run: `.venv/bin/python tools/verify_player_cfg.py`
- `tools/verify_gravity_cfg.py` — checks the gravity constants parse, the
  14/17-input net shapes, and that fresh populations match the new genome
  lengths. Run: `.venv/bin/python tools/verify_gravity_cfg.py`
- `tools/verify_gravity.py` — offscreen harness for the gravity physics:
  player + pack dive acceleration past boost, climb speed bleed, stall
  nose-drop recovery, the never-hover speed floor, the flock's dynamic
  dive/climb speed caps and MIN_BIRD_SPEED floor over a 180-frame soak.
  Run: `DISPLAY=:0 .venv/bin/python tools/verify_gravity.py`
- `tools/verify_drift.py` — offscreen harness for aerodynamic drift:
  spawn velocity along the nose, zero AoA slip + unchanged terminal speed
  in a straight dive, v_dir lag / swoop arc / induced-drag bleed in a
  90deg pull-out of a fast stoop, and the same in the vectorized pack form.
  Run: `DISPLAY=:0 .venv/bin/python tools/verify_drift.py`
