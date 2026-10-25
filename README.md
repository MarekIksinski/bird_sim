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
  contiguous blocks; 21 observations incl. airspeed + vertical pitch +
  arena awareness (wall repulsion on the body axes, nearest-wall
  proximity, ceiling clearance) so
  tribes can sense and exploit gravity and steer inward at the boundary).
  Hidden widths are configurable:
  `[flock_nn] hidden_layers = [32, 16]` -> shape (21,32,16,6); any
  depth/width works, and changing it starts the tribes fresh (shape
  guard on `starling_tribes.npz`). Tribes co-evolve against the hawks
  via a Moran process: a tribe that loses 5 birds (counter decays with a
  ~23-minute half-life, so sustained hunting pressure executes tribes every
  few generations) is replaced by a mutated copy of a top parent. Parents
  are ranked by **score rate** (points per second alive, not raw cumulative
  score, so a freshly evolved high-performer isn't locked out by an old
  tribe's age-accrued total) and must be mature (mean survival >= 10s).
  Bird score rewards **close-call evasion**: surviving inside a hawk's
  panic radius pays up to +2.5/s on top of survival + cohesion, so jukes
  that beat a dive earn far more than sitting on the unattacked side of
  the flock. Every replacement lands in `evolution/evolution.log`.
- **Hawk pack (neuroevolution)** — a population of hawks, each flown by an
  MLP (15 observations -> 4 controls: yaw/pitch/roll rates + throttle;
  observations include airspeed, nose pitch, and boost stamina). Hidden
  widths are configurable: `[hawk] hidden_layers = [32, 16]` -> shape
  (15,32,16,4); any depth/width works, and changing it changes
  GENOME_LEN, so `population.npz`'s shape guard starts a fresh
  evolution. Boost
  runs on a **stamina tank**: throttle > 0 drains it (full boost lasts
  `boost_duration` seconds), gliding/braking refills it, and hitting empty
  locks positive throttle out until the tank recovers to
  `boost_cooldown_recover` — letting off before empty recharges freely
  with no lockout (anti-feathering). Aerodynamic gravity trades
  altitude for airspeed: a stoop accelerates past boost toward the dive
  ceiling, a climb bleeds speed, and a nose-up hawk below the stall speed
  drops its nose back toward the horizon. The flight path also **drifts**:
  velocity lags behind the nose (speed-modulated lift bite), so high-speed
  dives carve sweeping arcs, and sharp high-AoA turns bleed airspeed via
  induced drag.
  True 6-DOF quaternion flight, arena/ceiling deflection, plus a terrain
  **pull-up reflex** (a hawk under 8m ground clearance with a nose-down
  attitude gets its nose rotated up at a ground-proximity rate — no
  nose-dive ground-lock). Dense shaped fitness is **dt-scaled**
  (framerate-independent) and **aim-gated**: the proximity ramp only pays
  while the nose points at the nearest bird, dogfight inside 25m scales
  with closing speed, and holding near-max yaw rate without closing on a
  bird costs a spin-lock penalty — passive circling can no longer out-score
  attacking. Between generations, a **3-tiered speciation** scheme breeds
  the King of the Hill (Hall of Fame) with tiny mutation (exploitation),
  plus two challenger lineages at medium/heavy mutation — Tier-2/3 parents
  are picked from OUTSIDE the King's Tier-1 slots when possible (lineage
  diversity), and slot 9 is an **immigrant wildcard** (50/50 King x
  runner-up crossover + heavy mutation, or a fully fresh genome whenever a
  generation scored zero kills). All mutation offspring get **L2 weight
  decay (0.985) + a [-3.5, 3.5] clip** so tanh never saturates from random-
  walk weight drift; saturated checkpoints load back clipped into tanh's
  linear region. Hawks respawn at hunting altitude (terrain+35m, never
  more than 15m below the roost) instead of skimming the deck.
- **Rabbit prey layer** — 400 rabbits on a pure-NumPy 4-state machine
  (Forage / Vigilance / Freeze / Flee) that react to the nearest hawk and
  sprint to invisible burrows (45 of them), diving underground when cornered.
- **Fox pairs (ground predators, continuous evolution)** — 20 foxes in 10
  pairs, each pair sharing one 15-input MLP (prey direction/velocity/state
  in the body frame, the prey's escape route to its nearest burrow, own
  speed + stamina sensors, wall repulsion, and the nearest-WOLF direction +
  threat proximity for apex-predator evasion). Speed-modulated stealth: a
  creeping fox only triggers rabbit Flee at 7.5m / Freeze at 22m, a
  sprinting fox at the full hawk thresholds (50m/100m) — the raw distance
  is rescaled into the hawk-threshold space so the rabbit FSM works
  unchanged. Touch (1.2m) = catch: the rabbit respawns Frozen at a random
  burrow and the pair is credited. Evolution is a **Moran process on
  starvation** (no generation reset): each check replaces only the SINGLE
  most-starved pair (hunger clocks start staggered 0-20s and re-stagger on
  replacement, so fresh pairs never starve out in a synchronized wave),
  swapped for a mutated copy of a top score-RATE parent, checkpointed to
  `evolution/fox_pairs.npz` with its own shape guard. Boost runs on the
  same stamina tank/lockout design as the hawks. Foxes hunted by wolves get
  a survival bonus for staying alive inside the wolf panic radius; a fox
  eaten by a wolf respawns safe and its pair is punished (+35s starvation,
  -25 score), pushing bad-at-evasion fox pairs toward replacement.
  NOTE: all entity colors use `color.rgba32(r,g,b)` — in this ursina build
  `color.rgb()` is 0..1-scale (an `rgba` alias), so `rgb(215,85,25)` stores
  a 215x color scale that clamps to WHITE.
- **Wolf pairs (apex predators — the cooperation experiment)** — 8 wolves
  in 4 pairs, each pair sharing one 16-input / 3-output MLP. The third
  output is a HOWL: a vocal signal in [-1,1] every other wolf within 250m
  hears (the packmate sensor is loudness-weighted, so a howling wolf draws
  attention from farther away). Kills inside a 35m assist radius are
  **shared across pair boundaries** (0.65 of score + hunger relief to
  every nearby wolf's pair), with a +25 howl-coop bonus when someone was
  actively howling — so cooperative pack hunting and vocal communication
  can emerge between DIFFERENT pairs. Wolves eat foxes (+120, full
  starvation reset) and rabbits (+45 snack); they are endurance cursors
  (4.5s sprint tank = 2.5x a fox's) with a wider turning circle (95 deg/s
  vs the foxes' 140), so a lone wolf cannot simply run down a boosting
  fox — pincer angles pay. Howling wolves light up ice-blue on screen.
  Moran starvation replacement like the foxes; `evolution/wolf_pairs.npz`.
  Rabbits detect an approaching wolf at 55m and flee to burrows.
- **Spectator hawk (player)** — a keyboard-piloted hawk flying the same
  6-DOF quaternion physics, wall/ceiling deflection, gravity stoop/climb
  energy exchange, stall nose-drop and ground clearance as the AI pack. It
  is a LIVE predator: starlings panic, evade and get caught by it (caught
  birds respawn at the roost and drive starling tribe evolution), and
  rabbits freeze/flee from it — but player catches only count in the HUD
  (`catches: N (you: M)`), never in the AI pack's neuroevolution fitness.
  Its flight feel (speeds, rates, key
  bindings) is configured separately in `[player]`.
- **Visual fidelity layer** — the terrain renders with a **triplanar
  shader** (grass on flat ground, **natural rock** on slopes — a
  seamless, procedural grey-brown rock texture generated by
  `tools/gen_rock_texture.py` into `assets/textures/` — cross-faded by
  the world normal with a wide face-overlap zone and a smoothstep on
  `N.y`, so gentle slopes stay grassy and no hard grid lines appear;
  derived from ursina's built-in `triplanar_shader` with its GLSL
  compile bug fixed and fog + shadow lighting spliced in; textures,
  scales and blend band in `[visuals]`). A **dynamic
  shadow-casting sun** (`DirectionalLight`, shadow map + bias in
  `[visuals]`) lights every entity through `lit_with_shadows_shader`, and
  **sprint dust trails**: foxes/wolves above 1.5x cruise kick up fading
  ground pips behind them from a fixed recycled entity pool (rate,
  lifetime, pool sizes in `[visuals]`, pip color in `[colors] dust`).

## Running

```bash
./run.sh          # creates a py3.12 venv via uv, exports CUDA_PATH, launches
./run.sh --cpu    # skip the CuPy probe (or set game.force_cpu in config.toml)
```

Evolution checkpoints persist in the folder set by `[pack] save_folder`.
Alongside the checkpoints that folder also holds `evolution.log` (every
session start, enriched per-generation summary — best/mean fitness, HoF
crownings with Slot-0 live score, peak dive speed, flock evolutions — and
starling tribe replacements with parent lineage, timestamped and flushed
live for `tail -f`) and `history.csv` (one rich row per generation for
charting). Everything respects `[pack] persist = false`.

All tunables live in `config.toml` (flock, flock_nn, hawk, player, pack,
rabbits, foxes, wolves, terrain, visuals, colors, game). Gravity/dive knobs: `[hawk]` `gravity`, `dive_max_speed`,
`min_flight_speed`, `stall_speed`, `lift_grip`, `induced_drag` (all shared
by the player) and `[flock]`
`gravity`, `min_bird_speed`, `dive_speed_mult`, `climb_speed_mult`.
Boost-stamina knobs: `[hawk]` `boost_duration`, `boost_recharge`,
`boost_cooldown_recover` (also shared with the player).
Network width knobs: `[hawk] hidden_layers`, `[flock_nn] hidden_layers`,
`[foxes] hidden_layers` and `[wolves] hidden_layers`
(lists of hidden layer widths; changing either starts that evolution fresh).
The `[player]` section overrides the spectator hawk's
speeds/rates (falling back to `[hawk]` values when a line is removed) and
its key bindings (`<action>_keys` = a key name or a list of them).
Evolution checkpoints persist in the folder set by `[pack] save_folder`.

## Controls

Camera modes: `1` fixed wide, `2` pack-chase (best hawk), `3` AI
first-person, `4` player chase, `5` player first-person, `6` fox chase,
`7` fox first-person, `8` wolf chase, `9` wolf first-person.

| Key | Action |
|-----|--------|
| 1 … 9 | switch camera mode (see above) |
| TAB | cycle the tracked entity: the AI hawk, or the fox in the fox cams, or the wolf in the wolf cams |
| R | respawn the player hawk (remappable: `[player] reset_keys`) |
| ESC | quit |

Player flight (default bindings, all remappable in `[player]`):

| Key | Action |
|-----|--------|
| W / S | pitch nose down / up |
| A / D | roll left / right (banks into a coordinated turn) |
| Q / E | yaw left / right |
| Shift / Ctrl | throttle boost / slow (release = cruise; boost runs on the stamina tank — empty = locked out until recharged) |

## Files / architecture

Strict bottom-up layering (each module only imports layers below it):

```
backend.py    --cpu / FORCE_CPU handling, CuPy VRAM probe, cp / np / to_cpu
config.py     config.toml -> all uppercase constants + derived values
math3d.py     pure NumPy batched quaternion math (_q_mul_b, _slerp_np, ...)
terrain.py    heightmaps + height_at at import; sky/triplanar terrain/
              backdrop + props (80 Tree.glb model trees, 120
              valley bushes, 100 slope/backdrop rocks) via init_environment()
predators.py  hawk pack state, neuroevolution, flight physics, checkpoints
flock.py      starling state, tribe NNs, update_boids_gpu, FlockVisuals
rabbits.py    ground-prey FSM, update_rabbits, RabbitVisuals
foxes.py      ground predator pairs: pair MLPs, Moran starvation
              replacement, stealth threat scaling (via rabbits), FoxVisuals
wolves.py     apex predator pairs: 16-in/3-out MLP with the HOWL vocal
              channel, inter-pair assist-share rewards, fox+rabbit
              catches, Moran starvation replacement, WolfVisuals
              (howling wolves light up via per-vertex colors)
player.py     spectator hawk state + keyboard flight (live predator:
              fed to predators/rabbits as an extra threat; catches only
              bump player.player_catches)
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
- `tools/verify_hof.py` — King-of-the-Hill speciation gate: crowning/EMA,
  lineage-diverse Tier-2/3 parents (outside Tier 1), slot-9 immigrant
  (crossover + zero-kill fresh fallback), weight-decay clip on mutation
  and saturated-checkpoint recovery on load.
  Run: `.venv/bin/python tools/verify_hof.py`
- `tools/verify_player.py` — offscreen harness for the spectator hawk:
  spawn, straight flight, pitch/roll signs, throttle, wall/ceiling/ground
  clamps, dt-spike guard, camera modes 1-5, respawn, and render-truth
  quaternion sync. Run: `DISPLAY=:0 .venv/bin/python tools/verify_player.py`
- `tools/verify_player_cfg.py` — checks the `[player]` config section:
  speed/rate overrides, `[hawk]` fallbacks, and string/list key bindings.
  Run: `.venv/bin/python tools/verify_player_cfg.py`
- `tools/verify_player_predation.py` — the player hawk as a LIVE predator:
  panic/flee vector from the player alone, catch counting split (player
  catches never touch gen_kills/fitness/total_catches), tribe casualty
  bookkeeping for player-caught birds, rabbit freeze/flee reaction to the
  player, and the pack-rows-only hawk_dmin source guard.
  Run: `.venv/bin/python tools/verify_player_predation.py`
- `tools/verify_gravity_cfg.py` — checks the gravity + stamina constants
  parse, the config-derived net shapes, and that fresh populations match the
  new genome lengths. Run: `.venv/bin/python tools/verify_gravity_cfg.py`
- `tools/verify_hidden_layers.py` — configurable MLP hidden widths:
  default shapes (15,32,16,4)/(21,32,16,6) match config.toml, batched
  forwards emit (POP,4)/(NUM_BIRDS,6), genome unpack round-trips,
  arbitrary depth/width via a variant config.toml subprocess
  ([8,4,8,2] and [64]), and old-shape checkpoints are rejected -> fresh.
  Run: `.venv/bin/python tools/verify_hidden_layers.py`
- `tools/verify_arena.py` — arena containment gate: birds re-flung at max
  panic speed every 60 frames stay inside the hard clamp (|x|,|z| <=
  TERRAIN_SIZE/2-5, y <= max_altitude-2), the soft wall push bends vx
  inward and the soft ceiling push bends vy down, the wall sensor math
  gives strong inward repulsion at the boundary, and the boundary-scrape
  penalty makes a wall-hugging bird score below a mid-arena one.
  Run: `FORCE_CPU=1 .venv/bin/python tools/verify_arena.py` (~2 min on CPU)
- `tools/verify_foxes.py` — fox pair layer gate: every `[foxes]` key parsed
  from config.toml, exact pair split + FOX_NET_SHAPE/genome match, wall
  soak + stamina drain/lockout/recharge, speed-modulated stealth thresholds
  (creep vs sprint), catch -> burrow respawn + pair credit, starvation
  Moran replacement with rate-ranking and [fox_nn] log line, batched
  einsum-vs-loop forward match, fox_pairs.npz round-trip + shape guard.
  Run: `FORCE_CPU=1 .venv/bin/python tools/verify_foxes.py`
- `tools/verify_wolves.py` — apex wolf pair gate: every `[wolves]` key
  parsed from config.toml, exact pair split + WOLF_NET_SHAPE (16-in /
  3-out) / genome match, wall soak + endurance stamina drain/lockout,
  loudness-weighted howl packmate pick + signal publication, fox catch
  (+120, register_fox_eaten respawn + pair punishment), rabbit snack
  (+45, 0.6x relief, burrow respawn), inter-PAIR assist share + howl coop
  bonus + wolf_coop_kills, the 55m rabbit wolf-threat rescale, starvation
  Moran with [wolf_nn] log, wolf_pairs.npz round-trip + shape guard.
  Run: `FORCE_CPU=1 .venv/bin/python tools/verify_wolves.py`
- `tools/verify_fox_cams.py` — offscreen harness for the fox cameras:
  fox_quat unit-norm + slope tilt, key 6/7 mode switches, context-aware
  TAB (fox idx in fox cams, hawk idx elsewhere), chase-cam terrain
  clearance, FP cam ahead of the nose tip, HUD stamina line.
  Run: `DISPLAY=:0 .venv/bin/python tools/verify_fox_cams.py`
- `tools/verify_colors.py` — reads every entity/HUD color back offscreen
  and compares it to `[colors]` (all rgba32-scaled).
  Run: `DISPLAY=:0 .venv/bin/python tools/verify_colors.py`
- `tools/verify_stamina.py` — offscreen harness for the boost-stamina
  cycle: pinned full throttle drains the tank in `boost_duration`, the
  exhaustion lockout clamps throttle to 0 and holds until
  `boost_cooldown_recover` recharges (full `boost_recharge`), releasing
  at a partial tank recharges with no lockout (anti-feathering), braking
  stays available while exhausted, the obs[:,14] stamina sensor
  (+1 full / -0.8 empty / -1 locked), and the HUD % + [COOL] readout —
  for pack and player. Run: `DISPLAY=:0 .venv/bin/python tools/verify_stamina.py`
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
- `tools/verify_logging.py` — checks evolution.log telemetry: session
  separator, enriched [evolution] lines (CROWNED/HELD + slot0 live +
  max_spd + flock_evos), the rich history.csv schema (incl. old-schema
  rewrite), tribe-replacement lineage logging, and the PERSIST=False
  no-filesystem path. Run: `.venv/bin/python tools/verify_logging.py`
- `tools/verify_tribe_evolution.py` — starling co-evolution gate:
  `[flock_nn]` config values derived from config.toml + decay half-life,
  net-shape guard, 5-catch tribe execution via `_tribe_respawn_check`
  (counter bump + `[flock_nn]` log line), rate-based parent ranking
  (age-bias regression), the 10s maturity gate + fallback pool, the
  close-call evasion bonus formula, and the starling checkpoint round-trip.
  Run: `.venv/bin/python tools/verify_tribe_evolution.py`
- `tools/verify_terrain_backdrop.py` — ground-surface gate: `height_at()`
  = max(playfield, backdrop) with bit-identical mesh hashes (visual shell
  unchanged), play-wins-at-center / backdrop-wins-in-the-four arena
  corners, scalar vs array call parity, and the intact play-3 feathering
  at backdrop lattice vertices. Run: `.venv/bin/python tools/verify_terrain_backdrop.py`
- `tools/verify_props.py` — props gate: exact counts (80 GLB model trees —
  falls back to 80 procedural trunk+crown pairs if `tree_model = ""` or the
  asset is missing — 120 valley bushes, 100 slope/backdrop rocks), prop
  bases flush on `height_at()` (rocks sunk 0.4x height by design), roost
  clearing respected, bushes only in low valleys, crowns atop trunks, and
  prop-scattering startup time <2s (terrain/backdrop mesh builders stubbed).
  Run: `.venv/bin/python tools/verify_props.py`
- `tools/verify_visuals.py` — visuals gate: terrain carries the patched
  triplanar shader (stock `.rgba` bug gone, fog + cast_shadows present,
  the hard `N.y > 0` cut replaced by the smoothstep blend, `[visuals]`
  scales/blend band applied, side texture is the natural rock not
  brick/white_cube), the sun is a shadow caster with the raised
  bias, and the sprint-dust pools emit only above cruise*factor, plateau
  below the pool cap (fade + recycle proven), and land exactly 0.5m behind
  the animal. Run: `.venv/bin/python tools/verify_visuals.py`
- `tools/gen_rock_texture.py` — regenerates `assets/textures/rock_procedural.png`
  (seamless numpy-fBm rock/cliff texture for the triplanar side; prints a
  seam-continuity self-check). Run: `.venv/bin/python tools/gen_rock_texture.py`
