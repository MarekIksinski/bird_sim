# Day/Night Cycle + Turbo Training — implementation plan

Agreed design (2026-10-31):
- Cycle = 5 min wall sim-time: 3 min day / 2 min night, dawn/dusk transition width is its own knob.
- Light becomes a NEW NN observation on ALL four networks (hawk, starling tribes, fox pairs, wolf pairs).
  => All shape guards reject old checkpoints => evolution resets to random. ACCEPTED.
  => First commit on `testing` snapshots the current `evolution/` folder aside before anything lands.
- Hard (non-learned) ecology on top: starlings roost at dusk, hawks physically handicapped at night,
  foxes/wolves small night bonus. Networks LEARN the rest themselves.
- Generations = 600 s (every hawk lives through >= 2 full day/night cycles).
- Idle turbo-train: when the player is inactive, run K physics substeps per rendered frame
  (fixed dt, never bigger dt). GPU is free during sim (Strata off), so K can be aggressive.

---

## Task 0 — Preserve the evolved lineage [DONE 2026-10-31]
- testing merged FF into main (both pushed); user's hidden_layers retune committed first
  (all four networks already shape-reset at b596b12, so no live lineage rides into Task 3).
- evolution/ copied to evolution_backup/ (gitignored, commit 93a1c5b). Rollback path exists.

## Task 1 — daynight.py (new layer, bottom of the stack)
1. Module state: `day_phase` (float32 accumulator), `day_number` (int).
   `day_phase += dt * (2*PI / cycle)` — NEVER feed Ursina `time.time()` into oscillators (epoch quantization bug).
   dt capped like everywhere else (0.05).
2. Config `[daynight]`:
   - `day_seconds = 180`, `night_seconds = 120` (asymmetric ramp),
   - `transition_seconds` (dawn/dusk smoothstep width, ~15 s default).
3. Exported scalars every layer reads:
   - `light_level` 0..1 (smoothstep ramps at dawn/dusk, flat 1 midday, flat ~0 midnight),
   - `sun_elevation` (for the later visual sun; derive from phase),
   - `is_night` helper = light_level below a fixed constant.
4. Light encoding for NN obs (same style as the stamina sensor): `obs = -1.0 + 2.0 * light_level`
   (noon +1, midnight -1, tanh-friendly). One shared helper `light_obs()` in daynight.py.
5. Day phase is NOT checkpointed — every session starts at dawn, day_number resets.
6. Wire `update_daynight(dt)` into the sim chain (see Task 2's sim_step; before predators/flock update).

## Task 2 — sim_step() refactor (the real code cost)
1. Extract the per-frame physics chain — daynight + update_boids_gpu + update_predators +
   update_player + update_foxes + update_wolves + update_rabbits — into one callable
   `sim_step(dt)` (new module or main.py), fully independent of Ursina visuals.
2. Visual updates (FlockVisuals/RabbitVisuals/FoxVisuals/WolfVisuals mesh writes, HUD) stay in the
   render pass and read state — they must not be the thing that drives physics.
   NOTE: predators<->flock lazy-import pattern and module-global style (`predators.best_idx`, never
   `from module import var`) must survive the move.
3. Ursina `update()` becomes: `sim_step(time.dt)` once + visuals. Offscreen harnesses can later call
   sim_step directly (keep taskMgr.step() harnesses working for now; migrating them is optional cleanup).
4. Gate: existing verify_* suite passes bit-comparably before touching day/night behavior.

## Task 3 — Light input on all four networks (+1 obs each)
1. config.py: bump NET_SHAPE inputs: hawk (15->16), STARLING_NET_SHAPE (21->22),
   FOX_NET_SHAPE (15->16), WOLF_NET_SHAPE (16->17). GENOME_LEN/WGL/FGL recompute automatically.
2. Observation sites append `light_obs()` as the LAST obs (append, don't renumber existing obs):
   - predators.py `_hawk_observations` obs[15]
   - flock.py obs_gpu obs[21]
   - foxes.py pair obs obs[15]
   - wolves.py pair obs obs[16]
3. Checkpoint guards (population / starling_tribes / fox_pairs / wolf_pairs) will reject old shapes —
   verify each guard's mismatch path logs and starts fresh instead of crashing (the starling guard's
   if/else after the hawk guard must still run).
4. Weight hygiene unchanged: fresh random init + the existing decay/clip applies.

## Task 4 — Hard ecology (physical, not learned)
All factors read `[daynight]` config; modulation applies to PHYSICS only, never to NN weights/obs
(the networks get light only as the one learned input).
1. Starlings (flock.py): roost pull strength x`roost_night_mult` and cruise clamp x`cruise_night_mult`
   scale smoothly with (1 - light_level). Panic speed NOT reduced (still flee if a hawk finds them).
   Keep `vertical_amplitude <= base_altitude - 22` rule intact.
2. Hawks + player (predators.py / player.py): night handicap on max speed / boost target
   (`hawk_night_speed_mult`, e.g. 0.85 at midnight) via smoothstep on light_level — night hunting stays
   possible but physically costed, so learned night-hunting isn't free.
   Gravity/drift/stamina code untouched; multipliers go on the speed clamps only.
3. Foxes + wolves (foxes.py / wolves.py): night bonus factor on their effective sense/stealth radii
   (`pred_night_sense_mult`, e.g. +20% at midnight). For foxes piggyback the existing speed-based
   stealth anchors (eff_flee_r/eff_freeze_r rescale) rather than adding a parallel path.
4. Rabbits: no direct light handling — they feel night only through fox/wolf radius changes.
5. Beware the rabbit-threshold slope rule: any rescale of d_fox_raw must keep slope 1:1 beyond
   eff_freeze_r or distant foxes fake Freeze.

## Task 5 — Generation length + retune watch
1. `[pack] generation_seconds = 600` (2 full cycles per hawk lifetime).
2. Known constants tuned against ~45 s generations — expect retunes after first long runs:
   - `[flock_nn] casualty_decay` (~23 min half-life) + `replace_deaths`: tribes will accumulate
     casualties very differently over 600 s gens; watch `[fock_nn]` replacement rate in the log.
   - HoF EMA `hof_decay_keep` 0.7: updates 13x less often -> crowning rarer/stickier. Watch
     `CROWNED|HELD` lines; may need lower keep.
   - Fitness magnitudes scale ~13x; history.csv y-axis will jump — that's expected, not a bug.
3. These are retune-later knobs, not code changes — flag values in the log review after ~10 gens.

## Task 6 — Idle turbo-train
1. Definition of idle: no key event for `turbo_idle_seconds` (default ~10) AND curr_mode is a fixed
   camera (0). Any keypress or a chase/FP camera -> instant 1x. Explicit toggle key (default `t`)
   forces turbo on/off regardless (spectator fast-forward).
2. Turbo = substepping, NEVER larger dt: render frame runs `sim_step(fixed_dt)` K times
   (`turbo_factor` K, fixed dt = 1/60), so physics/fitness are identical per sim-second at any K.
   Day phase, generation timer, checkpoints advance per sim-time only — turbo changes wall-clock speed,
   nothing else. Genomes must transfer between turbo and 1x sessions cleanly.
3. Render decoupling: in turbo mode update visuals/HUD only every K-th substep (K=8 -> 8 sim frames
   per drawn frame). GPU is free (Strata off during sim) but measure achievable K; config-capped.
4. HUD: `[TURBO xK]` badge + the Day/clock readout (Task 7) so a jumped clock is never a mystery.
5. Ursina's `time.dt` spike after a heavy turbo frame: camera lerps keep the existing
   `cdt = min(max(time.dt,0),0.1)` clamp; sim_step uses its own fixed dt and is immune.

## Task 7 — HUD clock
1. hud_cam gen_text: `Day N  HH:MM` derived from day_phase (map phase to 24 h clock) +
   day/night glyph or light bar. Read daynight module state each frame.
2. evolution.log `[evolution]` line gains `day N` so kill records correlate with time-of-day.

## Task 8 — Harnesses
1. New `tools/verify_daynight.py`:
   - phase monotonic + wraps, day/night asymmetry matches toml (180 s light vs 120 s dark),
   - transition smoothstep within `transition_seconds`,
   - dt-invariance: 60 steps of dt=1/60 == 1 step... NO — equal SIM TIME produces equal phase
     (fixed-dt substeps), assert turbo K-substeps == K frames at 1x bit-comparable,
   - light_obs() range [-1, +1],
   - night physics asserts: starling cruise clamp lower at midnight, hawk speed mult applied,
     fox sense radius factor up at midnight (all derived from config, never hardcoded).
2. Update shape asserts (they derive from config, mostly automatic, but checkpoint-rejection fixtures
   need new expected shapes): verify_hidden_layers.py, verify_gravity_cfg.py, verify_foxes.py,
   verify_wolves.py, verify_tribe_evolution.py, verify_hof.py (NET_SHAPE), verify_stamina.py
   (obs[14] stays, new obs appended — sensor index asserts).
3. Turbo harness section can live inside verify_daynight.py (offscreen, no Ursina needed once
   sim_step is Ursina-free — the Task 2 refactor makes this the first Ursina-free harness).

## Task 9 — Finish the repo (per project convention)
1. README.md: day/night feature bullet, `[daynight]` config section, turbo keys in Controls table,
   architecture layer diagram (daynight.py placement), tools/ harness list + run commands.
2. Move any scratch harness into tools/, re-run from new path.
3. Full verify suite as one background command with notify=true, one log per harness (~10 min).
4. Commit + push to `testing` (report whether prior unpushed commits rode along).
