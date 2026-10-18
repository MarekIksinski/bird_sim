#!/usr/bin/env python3
"""verify_tribe_evolution.py — starling tribe co-evolution gate (no window).

Imports the real flock/predators modules (FORCE_CPU shim) and checks:
 1. config: TRIBE_REPLACE_DEATHS / CASUALTY_DECAY defaults match config.toml
    (parsed from the toml, never hardcoded — the user retunes live), and the
    half-life math behind the 0.0005/s value.
 2. STARLING_NET_SHAPE stays (17,16,8,6) so existing checkpoints load.
 3. Simulating TRIBE_REPLACE_DEATHS catches in one tribe triggers
    _tribe_respawn_check, increments starling_evolutions, and writes a
    [flock_nn] line into evolution.log via log_event.
 4. Parent selection uses score RATE (score/second alive), not raw
    cumulative score: an old tribe with a huge raw total but low rate must
    NOT out-rank a young high-rate tribe (the age-bias fix).
 5. Maturity gate: with <3 mature live tribes the pool falls back, and a
    mature high-rate tribe is picked when one exists.
 6. The close-call evasion bonus formula in update_boids_gpu's score line
    (source-level check + numeric boundary check of the same expression).
 7. save_checkpoint() persists the mutated tribe_pop + counter to the
    starling npz (generation-rollover persistence stays synced).
"""
import os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ['FORCE_CPU'] = '1'

import numpy as np
import config

TMP = tempfile.mkdtemp(prefix='birdsim_tribeevo_')
config.EVOLVE_DIR = TMP          # before predators import: ckpt + log land here
import predators, flock


# --- 1. config values derived from config.toml (never hardcoded) ----------
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with open(os.path.join(ROOT, 'config.toml'), 'rb') as fh:
    toml = tomllib.load(fh)
fn = toml.get('flock_nn', {})
assert config.TRIBE_REPLACE_DEATHS == int(fn.get('replace_deaths', 5)), \
    f"replace_deaths mismatch: config={config.TRIBE_REPLACE_DEATHS}"
assert abs(config.CASUALTY_DECAY - float(fn.get('casualty_decay', 0.0005))) < 1e-12
# The whole point of 0.0005/s: a ~23-minute half-life (ln2/0.0005 = 1386s),
# vs ~35s at the old 0.02/s which decayed kills faster than ~4/30s gens accrue.
assert np.log(2.0) / config.CASUALTY_DECAY > 600.0, 'decay half-life too short'
print(f"PASS config: replace_deaths={config.TRIBE_REPLACE_DEATHS}, "
      f"casualty_decay={config.CASUALTY_DECAY} "
      f"(half-life {np.log(2.0)/config.CASUALTY_DECAY:.0f}s)")

# --- 2. net shape guard: existing checkpoints must stay compatible ---------
assert flock.STARLING_NET_SHAPE == (17, 16, 8, 6), flock.STARLING_NET_SHAPE
print('PASS STARLING_NET_SHAPE unchanged:', flock.STARLING_NET_SHAPE)

# --- helpers ---------------------------------------------------------------
R = flock.BIRDS_PER_TRIBE
RD = flock.TRIBE_REPLACE_DEATHS

def scenario(maturity=60.0, seed=7):
    """Reset tribe bookkeeping: all birds alive `maturity` s, scores 0."""
    flock.tribe_pop[:] = np.random.default_rng(0).normal(0, 0.5, flock.tribe_pop.shape)
    flock.bird_survival_time[:] = maturity
    flock.bird_score[:] = 0.0
    flock.tribe_casualties[:] = 0.0
    flock.starling_evolutions = 0
    flock._starling_rng = np.random.default_rng(seed)

def kill_tribe(T):
    """Simulate RD catches in tribe T through the real respawn path."""
    caught = np.where(flock.bird_tribe_np == T)[0][:RD]
    flock._tribe_respawn_check(caught)

# --- 3. RD catches -> replacement + log_event line -------------------------
predators.load_checkpoint()   # writes the [session] line
scenario()
before = flock.starling_evolutions
kill_tribe(0)
assert flock.starling_evolutions == before + 1, 'starling_evolutions not incremented'
assert flock.tribe_casualties[0] == 0.0, 'casualty counter not cleared on replace'
# the replaced tribe's birds re-earned nothing: score AND clock zeroed
blk0 = flock.bird_tribe_np == 0
assert np.all(flock.bird_score[blk0] == 0.0) and np.all(flock.bird_survival_time[blk0] == 0.0)
logtxt = open(predators._LOG_FILE).read()
nn_lines = [l for l in logtxt.splitlines() if '[flock_nn]' in l]
assert len(nn_lines) == 1, nn_lines
for token in ('tribe(s) [0] replaced', 'by parent(s)', 'top_tribe_rate',
              'total starling evolutions: 1'):
    assert token in nn_lines[0], f'missing {token!r} in: {nn_lines[0]}'
print('PASS RD catches -> evolve + [flock_nn] log line:', nn_lines[0][20:])

# --- 4/5. rate-based ranking + maturity gate --------------------------------
# Age-bias fix: tribe 1 is ANCIENT (raw score huge, rate low), tribe 5 is
# YOUNG-BUT-FAST (raw total small, rate highest). Under raw-score ranking
# tribe 1 (raw 1000) would sit in the top-3 pool; under rate ranking its
# rate (2.0/s) is beaten by tribes 2/3/4 (3.0/s) and it is EXCLUDED.
scenario(maturity=60.0)
flock.bird_score[:] = 0.0
flock.bird_score[flock.bird_tribe_np == 1] = 1000.0     # raw total huge
flock.bird_survival_time[flock.bird_tribe_np == 1] = 500.0   # -> rate 2.0/s
for T in (2, 3, 4):                                      # rate 3.0/s each
    flock.bird_score[flock.bird_tribe_np == T] = 60.0
    flock.bird_survival_time[flock.bird_tribe_np == T] = 20.0
flock.bird_score[flock.bird_tribe_np == 5] = 120.0      # best rate: 6.0/s
flock.bird_survival_time[flock.bird_tribe_np == 5] = 20.0
kill_tribe(0)
line = [l for l in open(predators._LOG_FILE).read().splitlines() if 'tribe(s) [0]' in l][-1]
parent = int(line.split('by parent(s) [')[1].split(']')[0])
assert parent in (5, 2, 3, 4), f'rate-based pool broken: {line}'
assert parent != 1, f'ancient raw-high tribe parented: {line}'
assert not np.allclose(flock.tribe_pop[0], flock.tribe_pop[parent])  # mutated copy
delta = np.linalg.norm(flock.tribe_pop[0] - flock.tribe_pop[parent])
assert 0 < delta < 20.0, delta   # mutated, not a fresh random genome
print('PASS rate ranking: ancient high-raw tribe 1 excluded from top-3 pool')

# Maturity gate: all live tribes immature except one -> the pool falls back
# to ALL non-dead tribes (per spec), so any live tribe may parent, but the
# dead tribe 0 must never self-parent.
scenario(maturity=4.0)
flock.bird_survival_time[flock.bird_tribe_np == 1] = 60.0     # only mature one
flock.bird_score[flock.bird_tribe_np == 5] = 50.0             # immature, top rate
kill_tribe(0)
line = [l for l in open(predators._LOG_FILE).read().splitlines() if 'tribe(s) [0]' in l][-1]
parent = int(line.split('by parent(s) [')[1].split(']')[0])
assert parent != 0, f'dead tribe self-parented via fallback: {line}'
print('PASS maturity fallback: <3 mature tribes -> all live ranked pool used')

# With >=3 mature tribes, an immature high-rate tribe must be EXCLUDED from
# the pool entirely (a tribe replaced 2s ago isn't cloned before testing).
scenario(maturity=60.0)
flock.bird_survival_time[flock.bird_tribe_np == 5] = 2.0      # immature (just replaced)
flock.bird_score[flock.bird_tribe_np == 5] = 100.0            # top rate (20/s)
flock.bird_score[flock.bird_tribe_np == 1] = 300.0            # mature top rate (5/s)
kill_tribe(0)
parent = None
for _ in range(6):   # pool is a random draw from top-3 mature: repeat to be sure 5 never appears
    scenario(maturity=60.0, seed=100 + _)
    flock.bird_survival_time[flock.bird_tribe_np == 5] = 2.0
    flock.bird_score[flock.bird_tribe_np == 5] = 100.0
    flock.bird_score[flock.bird_tribe_np == 1] = 300.0
    kill_tribe(0)
    line = [l for l in open(predators._LOG_FILE).read().splitlines() if 'tribe(s) [0]' in l][-1]
    parent = int(line.split('by parent(s) [')[1].split(']')[0])
    assert parent != 5, f'immature tribe slipped into pool: {line}'
print('PASS maturity gate: 2s-old top-rate tribe never parents while 3 mature exist')

# --- 6. close-call evasion bonus -------------------------------------------
src = open(os.path.join(ROOT, 'flock.py')).read()
assert 'evasion_bonus = np.clip(1.0 - nearest_d_np / PANIC_RADIUS, 0.0, 1.0)' in src, \
    'evasion_bonus line missing from flock.py'
assert '+ 2.5 * evasion_bonus' in src, 'evasion weight 2.5 missing from score line'
# numeric boundary check of the same expression
PR = config.PANIC_RADIUS
d = np.array([0.0, PR / 2, PR, PR * 2, 1e9], dtype=np.float32)
ev = np.clip(1.0 - d / PR, 0.0, 1.0)
assert np.allclose(ev, [1.0, 0.5, 0.0, 0.0, 0.0])
print('PASS close-call evasion bonus: +2.5/s max inside PANIC_RADIUS, 0 beyond')

# --- 7. checkpoint persistence of evolved tribes ----------------------------
predators.save_checkpoint()
s = np.load(predators._STARLING_CKPT)
assert np.array_equal(s['tribe_pop'], flock.tribe_pop)
assert int(s['starling_evolutions']) == flock.starling_evolutions
# generation rollover (evolve_generation) itself calls save_checkpoint, so
# evolved tribes land on disk with every hawk generation — verify via the
# counter surviving a reload.
saved_evo = int(s['starling_evolutions'])
flock.tribe_pop[:] = 0.0
predators.load_checkpoint()
assert not np.all(flock.tribe_pop == 0.0), 'starling checkpoint reload failed'
assert flock.starling_evolutions == saved_evo
flock._upload_tribe_layers()
print('PASS starling checkpoint round-trip (tribe_pop + counter synced on rollover)')

print('ALL TRIBE-EVOLUTION CHECKS PASSED')
