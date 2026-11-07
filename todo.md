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

## Task 1 — daynight.py (new layer, bottom of the stack) [DONE 2026-10-31]
- daynight.py: sim_elapsed (the ONLY physics time source now — flock roost
  orbit/vertical wave switched off Ursina time.time()), day_phase, day_number,
  light_level (smoothstep ramps), sun_height, light_obs() = -1 + 2*light.
- [daynight] toml section live: day 180 / night 120 / transition 15.
- tools/verify_daynight.py passes (first Ursina-free harness).
- NOTE: day_number currently only counts; HUD clock is Task 7.

## Task 2 — sim_step() refactor [DONE 2026-10-31]
- simstep.py:sim_step(dt) = daynight -> player -> boids -> rabbits -> foxes -> wolves;
  sets the shared Ursina Clock dt so existing per-module time.dt reads + harness
  Clock stubs keep working.
- Visuals classes: update() -> render() (mesh rebuild ONLY); main.py update()
  drives sim_step + render pass; update_pack_visuals moved out of
  update_boids_gpu into the render pass.
- Hop phases (rabbit/fox/wolf) integrated inside the physics steps (turbo-safe).
- verify_fox_cams.py re-driven via sim_step (no taskMgr).
- Gate: verify_refactor segfaults WITHOUT a live X display — pre-existing on
  baseline (proven via git stash); same for verify_visuals/verify_player (139).
  With a desktop session open, re-run those three.

## Task 3 — Light input on all four networks (+1 obs each) [DONE 2026-11-01, commit 167d0f6]
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

## Task 4 — Hard ecology (physical, not learned) [DONE 2026-11-01, commit 781739b; verify_night_ecology.py 12/12]
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

## Task 5 — Generation length + retune watch [DONE 2026-11-01: generation_seconds 90 -> 600; retune watch still OPEN after ~10 long gens]
1. `[pack] generation_seconds = 600` (2 full cycles per hawk lifetime).
2. Known constants tuned against ~45 s generations — expect retunes after first long runs:
   - `[flock_nn] casualty_decay` (~23 min half-life) + `replace_deaths`: tribes will accumulate
     casualties very differently over 600 s gens; watch `[fock_nn]` replacement rate in the log.
   - HoF EMA `hof_decay_keep` 0.7: updates 13x less often -> crowning rarer/stickier. Watch
     `CROWNED|HELD` lines; may need lower keep.
   - Fitness magnitudes scale ~13x; history.csv y-axis will jump — that's expected, not a bug.
3. These are retune-later knobs, not code changes — flag values in the log review after ~10 gens.

## Task 6 — Idle turbo-train [DONE 2026-11-01: simstep.begin_frame fixed-dt accumulator, 't' toggle, verify_turbo.py 13/13. NOTE: deviates from plan item 3 — visuals still render once per FRAME (not every K-th substep); physics chain already decoupled so this is the natural frame cadence]
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

## Task 7 — HUD clock [DONE 2026-11-01: gen_text 'Day N [day/dusk/night HH:MM]' + [TURBO xK] badge; evolution.log gains '(day N)']
1. hud_cam gen_text: `Day N  HH:MM` derived from day_phase (map phase to 24 h clock) +
   day/night glyph or light bar. Read daynight module state each frame.
2. evolution.log `[evolution]` line gains `day N` so kill records correlate with time-of-day.

## Task 8 — Harnesses [DONE 2026-11-01: verify_daynight (updated for straddling dawn ramp), verify_night_ecology.py NEW, verify_turbo.py NEW; shape asserts config-derived; physics harnesses pin_full_day()
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

## Task 9 — Finish the repo (per project convention) [PARTIAL: README day/night/turbo items still pending]
1. README.md: day/night feature bullet, `[daynight]` config section, turbo keys in Controls table,
   architecture layer diagram (daynight.py placement), tools/ harness list + run commands.
2. Move any scratch harness into tools/, re-run from new path.
3. Full verify suite as one background command with notify=true, one log per harness (~10 min).
4. Commit + push to `testing` (report whether prior unpushed commits rode along).

---

# Turbo button — design (agreed 2026-11-06)

A clickable TURBO button in the top-left corner, same widget family as ursina's exit button
(`Button(parent=window.editor_ui, eternal=True, z=-999)` — window.py:162 pattern). Tri-state cycle
matching keyboard semantics: auto -> forced ON -> forced OFF -> auto. The auto state (idle + fixed
camera) currently has NO manual entry point; the button becomes it.

## Task 10 — simstep.py: tri-state turbo_mode [DONE 2026-11-06]
1. Replace `_turbo_forced` bool with `turbo_mode` string: 'auto' | 'on' | 'off' (module global).
   - 'auto': begin_frame gates on idle AND curr_mode==0 (unchanged rule).
   - 'on': turbo regardless of idle/camera. 'off': never turbo.
2. `cycle_turbo_mode()`: auto->on->off->auto. `toggle_turbo()` ('t' key) keeps today's feel:
   toggles between forced on and auto (t from 'off' -> on). note_activity() unchanged except it
   only clears turbo_active when turbo_mode == 'auto'.
3. Keep every existing read of turbo_active working (HUD badge, begin_frame). No config changes.

## Task 11 — button widget [DONE 2026-11-06; landed in hud_cam.py init_hud_and_camera(), not main.py — HUD layer owns widgets]
1. Right after `window.exit_button.enabled = True`: `turbo_button = Button(text=..., parent=window.editor_ui,
   eternal=True, ignore_paused=True, z=-999, scale=(.09,.03))` positioned top-left
   (x = -(0.5*aspect_ratio) + margin, y near exit's .47). Mirror exit_button's styling knobs.
2. `on_click=simstep.cycle_turbo_mode` — mouse click is NOT a keypress, so it deliberately does NOT
   call note_activity(); the idle timer is untouched by clicking.
3. Label/color readout driven per-frame from hud_cam.update_camera_and_hud() (single source of truth,
   can never desync): 'TURBO: auto' muted gray, 'TURBO: ON' bright (tint by turbo_active: glowing
   while actually stepping fast), 'TURBO: OFF' dark. main.py does not import `window` yet? — it does
   (line 57); add `Button` + `color` to the ursina from-import as needed. Colors from config [colors]
   if trivial, else reuse hud colors.

## Task 12 — Harness + verification [DONE 2026-11-06: verify_turbo 4b section 27/27; live ui_check screenshot shows 'TURBO auto' top-left, no overlap. NOTE: fixed pre-existing verify_fox_cams.py broken import (imports were commented onto the pin_full_day line)]
1. Extend tools/verify_turbo.py (Ursina-free): tri-state cycle transitions; auto rule unchanged;
   'off' survives note_activity; 't' from each state. Expectations derived from config as usual.
2. Live-window screenshot (scratch/ui_check.py pattern: exec main.py into __main__.__dict__, SIGALRM
   ~8s base.screenshot + userExit) then vision_analyze the top-left corner — button visible, no
   overlap with terrain/HUD.

## Task 13 — Finish the repo [DONE 2026-11-06]
1. README: Controls table gains the turbo button row; feature bullet.
2. Commit + push to `testing`.
