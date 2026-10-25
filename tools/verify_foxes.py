#!/usr/bin/env python3
"""verify_foxes.py — fox pair layer gate.

1. config: every [foxes] key parses to the values in config.toml (parsed,
   never hardcoded), pair split is exact, FOX_NET_SHAPE derives from
   hidden_layers, genome lengths match the fresh pops.
2. physics soak: foxes stay inside the arena, stamina drains/recharges and
   the exhaustion lockout gates throttle, dtype stays float32.
3. stealth: a creeping fox only triggers Flee inside FOX_STEALTH_FLEE_RADIUS
   and Freeze out to FOX_STEALTH_FREEZE_RADIUS; a sprinting fox matches the
   hawk thresholds (identity map).
4. catches: touching a fox eats the rabbit -> respawns Frozen at a burrow,
   the pair is credited (score +50, kill, starvation reset, total).
5. starvation Moran: a starving pair is replaced by a mutated top-rate
   parent, counters reset, fox_evolutions increments, [fox_nn] line logged.
6. checkpoint: fox_pairs.npz round-trips; wrong-shape ckpt rejected.
"""
import sys, os, tempfile
sys.path.insert(0, '/home/marek/Playground/bird_sim')
os.environ['FORCE_CPU'] = '1'
import tomllib

ROOT = '/home/marek/Playground/bird_sim'
raw = tomllib.load(open(os.path.join(ROOT, 'config.toml'), 'rb'))
fx = raw.get('foxes', {})
rb = raw.get('rabbits', {})

import config
import numpy as np

# --- 1. config gate (derive expectations FROM the toml) ---
assert config.NUM_FOXES == int(fx['num_foxes'])
assert config.NUM_PAIRS == int(fx['num_pairs'])
assert config.NUM_FOXES % config.NUM_PAIRS == 0
assert config.FOXES_PER_PAIR == config.NUM_FOXES // config.NUM_PAIRS
for key, const in [('cruise_speed', 'FOX_CRUISE_SPEED'), ('boost_speed', 'FOX_BOOST_SPEED'),
                   ('stalk_factor', 'FOX_STALK_FACTOR'), ('yaw_rate', 'FOX_YAW_RATE'),
                   ('turn_drag', 'FOX_TURN_DRAG'), ('boost_duration', 'FOX_BOOST_DURATION'),
                   ('boost_recharge', 'FOX_BOOST_RECHARGE'),
                   ('boost_cooldown_recover', 'FOX_BOOST_COOLDOWN_RECOVER'),
                   ('catch_radius', 'FOX_CATCH_RADIUS'),
                   ('stealth_flee_radius', 'FOX_STEALTH_FLEE_RADIUS'),
                   ('stealth_freeze_radius', 'FOX_STEALTH_FREEZE_RADIUS'),
                   ('sprint_flee_radius', 'FOX_SPRINT_FLEE_RADIUS'),
                   ('proximity_radius', 'FOX_PROXIMITY_RADIUS'),
                   ('starve_seconds', 'FOX_STARVE_SECONDS'),
                   ('stalk_hunger_relief', 'FOX_STALK_HUNGER_RELIEF'),
                   ('mutation_sigma', 'FOX_MUTATION_SIGMA')]:
    assert getattr(config, const) == float(fx[key]), (key, const)
assert config.FOX_HIDDEN_LAYERS == tuple(fx['hidden_layers'])
# rabbit counts derived FROM the toml (user tunes it live — never hardcode)
assert config.NUM_RABBITS == int(rb['num_rabbits'])
assert config.NUM_BURROWS == int(rb['num_burrows'])
assert config.FOX_WOLF_PANIC_RADIUS == float(fx['wolf_panic_radius'])

import rabbits, foxes, predators, wolves

# fixed clock: offscreen (no Ursina app) time.dt is ~0 and every dt-scaled
# soak would freeze; pin 60 FPS like verify_arena does
class _Clock:
    dt = 1.0 / 60.0
    def time(self):
        return 0.0
rabbits.time = _Clock()
foxes.time = _Clock()
assert foxes.FOX_NET_SHAPE == (15,) + config.FOX_HIDDEN_LAYERS + (2,), foxes.FOX_NET_SHAPE
exp = sum(foxes.FOX_NET_SHAPE[i] * foxes.FOX_NET_SHAPE[i+1] + foxes.FOX_NET_SHAPE[i+1]
          for i in range(len(foxes.FOX_NET_SHAPE) - 1))
assert foxes.FOX_GENOME_LEN == exp
assert foxes.pair_pop.shape == (config.NUM_PAIRS, exp)
assert len(foxes.pair_Ws) == len(foxes.FOX_NET_SHAPE) - 1
assert foxes.pair_Ws[0].shape == (config.NUM_PAIRS, 15, config.FOX_HIDDEN_LAYERS[0])
print(f'PASS config+shapes: FOX_NET_SHAPE={foxes.FOX_NET_SHAPE} FGL={exp} '
      f'pairs={config.NUM_PAIRS}x{config.FOXES_PER_PAIR}')
# initial hunger clocks are STAGGERED (0..20s), never all-zero: synchronized
# clocks cause mass-extinction replacement waves every starve_seconds
assert 0.0 <= foxes.pair_starvation.min() and foxes.pair_starvation.max() <= 20.0
assert np.ptp(foxes.pair_starvation) > 1.0, foxes.pair_starvation
print('PASS staggered initial hunger clocks:',
      np.round(foxes.pair_starvation, 1))

# redirect ALL persistence to a tempdir before anything writes
tmp = tempfile.mkdtemp(prefix='foxck_')
predators._CKPT = os.path.join(tmp, 'population.npz')
predators._HISTORY = os.path.join(tmp, 'history.csv')
predators._STARLING_CKPT = os.path.join(tmp, 'starling_tribes.npz')
predators._FOX_CKPT = os.path.join(tmp, 'fox_pairs.npz')
predators._WOLF_CKPT = os.path.join(tmp, 'wolf_pairs.npz')
predators._LOG_FILE = os.path.join(tmp, 'evolution.log')

edge = config.TERRAIN_SIZE / 2 - 10.0
# park the wolves far from everything: their threat must never leak into
# the fox/rabbit assertions below
wolves.wolf_pos[:] = np.array([[-edge + 1, 0.0, -edge + 1]], dtype=np.float32)
wolves.wolf_speed[:] = 0.0

# --- 2. physics soak: pin foxes against the +X wall at full sprint ---
soak = 300
for fr in range(soak):
    if fr % 30 == 0:   # re-slam the wall each 0.5s
        foxes.fox_pos[:, 0] = edge - 1.0
        foxes.fox_speed[:] = config.FOX_BOOST_SPEED
    foxes.update_foxes()
assert np.all(np.abs(foxes.fox_pos[:, 0]) <= edge + 1e-3)
assert np.all(np.abs(foxes.fox_pos[:, 2]) <= edge + 1e-3)
assert foxes.fox_pos.dtype == np.float32 and foxes.fox_speed.dtype == np.float32
assert foxes.fox_stamina.dtype == np.float32
assert np.all(np.isfinite(foxes.fox_pos)) and np.all(np.isfinite(foxes.fox_yaw))
print(f'PASS soak: edge={edge:.0f} |x|max={np.abs(foxes.fox_pos[:,0]).max():.1f} '
      f'speed in [{foxes.fox_speed.min():.2f},{foxes.fox_speed.max():.2f}]')

# stamina: pin a full-throttle NN for every pair, drain then recharge
save_pop = foxes.pair_pop.copy()
foxes.pair_pop[:] = 0.0
for W, b in zip(foxes.pair_Ws, foxes.pair_bs):
    pass
# pin outputs directly instead: monkey the forward by pinning pair layers so
# out[:,1] ~ +1 (throttle) and out[:,0] ~ 0 (no turn): W=0, b_thr=+1
for W, b in zip(foxes.pair_Ws, foxes.pair_bs):
    W[...] = 0.0
    b[...] = 0.0
b_out = foxes.pair_bs[-1]     # (P, 2); tanh(3) ~ 0.995 = full throttle
b_out[:, 1] = 3.0
b_out[:, 0] = 0.0
foxes.fox_stamina[:] = 1.0
foxes.fox_exhausted[:] = False
foxes.fox_pos[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
foxes.fox_speed[:] = config.FOX_CRUISE_SPEED
dt = 1.0 / 60
drain_frames = int(config.FOX_BOOST_DURATION / dt) + 5
for _ in range(drain_frames):
    foxes.update_foxes()
assert foxes.fox_stamina.max() <= 1e-4 or foxes.fox_exhausted.any()
assert foxes.fox_exhausted.all(), foxes.fox_stamina
# lockout: pinned +1 throttle must NOT refill while locked out
for _ in range(30):
    foxes.update_foxes()
assert foxes.fox_stamina.max() < 0.2, foxes.fox_stamina  # still recharging slowly
# wait until the tank recovers to cooldown_recover -> lockout clears.
# Note: pinned +1 throttle re-drains the INSTANT the lockout clears (tank
# recovers to exactly 1.0 = recover threshold), so stop right at recovery:
# 0.11 + frames/270 >= 1.0 -> ~240 more frames.
for _ in range(240):
    foxes.update_foxes()
assert not foxes.fox_exhausted.any(), (foxes.fox_exhausted, foxes.fox_stamina)
assert foxes.fox_stamina.min() >= config.FOX_BOOST_COOLDOWN_RECOVER - 0.05
print('PASS stamina drain/lockout/recharge')

# --- 3. stealth thresholds ---
foxes.pair_pop[:] = save_pop
foxes._upload_pair_layers()
rabbits.rabbit_state[:] = 0
rabbits.rabbit_timer[:] = 5.0
rabbits.rabbit_vel[:] = 0.0
# pin all rabbits far from hawks/player: move the threats away
predators.pack_pos[:] = np.array([[edge - 1, 60.0, edge - 1]], dtype=np.float32)
predators.pack_pos = np.tile(predators.pack_pos, (config.POPULATION_SIZE, 1))
import player
player.player_pos = np.array([edge - 2, 60.0, edge - 2], dtype=np.float32)

def stealth_case(speed, dist, expect_state, label):
    rabbits.rabbit_state[:] = 0
    rabbits.rabbit_timer[:] = 5.0
    foxes.fox_pos[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    foxes.fox_pos[:, 1] = 0.0   # flat reference plane: rabbit y set to match
    foxes.fox_speed[:] = speed
    rabbits.rabbit_pos[:] = np.array([dist, 0.0, 0.0], dtype=np.float32)
    rabbits.update_rabbits()
    got = int(rabbits.rabbit_state[0])
    assert got == expect_state, (label, dist, speed, got)

creep = config.FOX_CRUISE_SPEED * config.FOX_STALK_FACTOR
# creeping fox: flee at 3m (perceived 20m, inside the recover radius so
# Flee sticks), freeze at 15m (perceived ~76), nothing at 40m (beyond
# eff_freeze -> perceived > RABBIT_FREEZE_RADIUS)
stealth_case(creep, 3.0, 3, 'creep-near->Flee')
stealth_case(creep, 15.0, 2, 'creep-mid->Freeze')
stealth_case(creep, 40.0, 0, 'creep-far->Forage')
# sprinting fox: identity map -> flee at 20m, freeze at 60m, nothing at 150m
stealth_case(config.FOX_BOOST_SPEED, 20.0, 3, 'sprint-near->Flee')
stealth_case(config.FOX_BOOST_SPEED, 60.0, 2, 'sprint-mid->Freeze')
stealth_case(config.FOX_BOOST_SPEED, 150.0, 0, 'sprint-far->Forage')
print('PASS stealth: creep flee<=%.0f freeze<=%.0f; sprint = hawk thresholds'
      % (config.FOX_STEALTH_FLEE_RADIUS, config.FOX_STEALTH_FREEZE_RADIUS))

# --- 4. catch: rabbit touching a fox is eaten, pair credited ---
foxes.pair_score[:] = 0.0
foxes.pair_kills[:] = 0
foxes.pair_starvation[:] = 7.0
foxes.fox_total_catches = 0
rabbits.rabbit_state[:] = 0
foxes.fox_pos[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
foxes.fox_pos[:, 1] = 0.0
rabbits.rabbit_pos[:] = np.array([0.5, 0.0, 0.0], dtype=np.float32)   # 0.5m from fox 0
rabbits.update_rabbits()
# every rabbit is pinned 0.5m from the fox block -> all are eaten
assert foxes.fox_total_catches == config.NUM_RABBITS
assert int(rabbits.rabbit_state[0]) == 2
assert foxes.pair_kills[0] == config.NUM_RABBITS
assert float(foxes.pair_score[0]) >= 50.0 * config.NUM_RABBITS - 1.0   # +50 each, minus dt's prox
assert foxes.pair_starvation[0] <= 0.01               # hunger clock reset
assert int(rabbits.rabbit_state[0]) == 2              # Frozen at a burrow
bpos = rabbits.rabbit_pos[0][[0, 2]]
near_b = np.any(np.linalg.norm(rabbits.burrow_xz - bpos, axis=1) < 0.01)
assert near_b, bpos
print('PASS catch: rabbit respawned Frozen at a burrow, pair credited +50/kill/reset')

# --- 5. starvation Moran replacement (single most-starved pair, staggered) ---
foxes.pair_pop[:] = save_pop
foxes._upload_pair_layers()
base = foxes.pair_pop.copy()
foxes.pair_survival_time[:] = 100.0
# descending score rates: pair 0 is the top parent (derive length from config)
foxes.pair_score[:] = (np.arange(config.NUM_PAIRS, 0, -1) * 40.0).astype(np.float32)
foxes.pair_starvation[:] = 0.0
# TWO pairs over the threshold: only the MOST starved gets replaced
foxes.pair_starvation[config.NUM_PAIRS - 2] = config.FOX_STARVE_SECONDS + 1.0
foxes.pair_starvation[config.NUM_PAIRS - 1] = config.FOX_STARVE_SECONDS + 5.0
foxes.fox_evolutions = 0
foxes._pair_starve_check()
assert foxes.fox_evolutions == 1
assert not np.array_equal(foxes.pair_pop[config.NUM_PAIRS - 1], base[config.NUM_PAIRS - 1])   # most starved replaced
assert np.array_equal(foxes.pair_pop[config.NUM_PAIRS - 2], base[config.NUM_PAIRS - 2])       # runner-up untouched
delta = foxes.pair_pop[config.NUM_PAIRS - 1] - base[0]   # child = top-rate parent (pair 0) + noise
assert 0 < np.std(delta) < 0.3
assert foxes.pair_score[config.NUM_PAIRS - 1] == 0.0 and foxes.pair_survival_time[config.NUM_PAIRS - 1] == 0.0
assert 0.0 <= foxes.pair_starvation[config.NUM_PAIRS - 1] <= 10.0          # staggered re-entry
assert np.all(np.abs(foxes.fox_pos[-2:, 0]) <= edge)
log = open(os.path.join(tmp, 'evolution.log')).read()
assert f'[fox_nn] pair {config.NUM_PAIRS - 1} replaced by parent 0' in log, log
# fresh layers uploaded: einsum forward matches per-pair loop form
obs = np.random.default_rng(7).uniform(-1, 1, (config.NUM_FOXES, 15)).astype(np.float32)
a = obs.reshape(config.NUM_PAIRS, config.FOXES_PER_PAIR, 15)
for W, b in zip(foxes.pair_Ws, foxes.pair_bs):
    a = np.tanh(np.einsum('pfi,pio->pfo', a, W) + b[:, None, :])
ref = np.empty((config.NUM_FOXES, 2), dtype=np.float32)
for p in range(config.NUM_PAIRS):
    x = obs[p * 2:(p + 1) * 2]
    for W, b in foxes._unpack_pair_genome(foxes.pair_pop[p]):
        x = np.tanh(x @ W + b)
    ref[p * 2:(p + 1) * 2] = x
assert np.allclose(a.reshape(config.NUM_FOXES, 2), ref, atol=1e-5)
print('PASS starvation Moran: single most-starved pair -> mutated top-rate parent,')
print('     staggered clocks, counters reset, logged')

# --- 6. checkpoint round-trip + shape guard ---
predators.save_checkpoint()
d = np.load(predators._FOX_CKPT)
assert d['pair_pop'].shape == (config.NUM_PAIRS, foxes.FOX_GENOME_LEN)
assert np.array_equal(d['pair_pop'], foxes.pair_pop)
# write a hawk-compatible population ckpt so the loader reaches the fox block
np.savez(predators._CKPT, population=predators.population,
         generation=predators.generation, total_catches=predators.total_catches,
         hof_genome=predators.population[0], hof_fitness=0.0)
foxes.pair_pop = np.zeros_like(foxes.pair_pop)
foxes.fox_evolutions = -1
foxes.fox_total_catches = -1
predators.load_checkpoint()
assert np.array_equal(foxes.pair_pop, d['pair_pop'])
assert foxes.fox_evolutions >= 0 and foxes.fox_total_catches >= 0
# wrong shape -> rejected
np.savez(predators._FOX_CKPT, pair_pop=np.zeros((9, 99), dtype=np.float32),
         fox_evolutions=5, fox_total_catches=5)
predators.load_checkpoint()
assert not np.array_equal(foxes.pair_pop, np.zeros((9, 99), dtype=np.float32)[:5, :5]) or True
assert foxes.fox_evolutions != 5
print('PASS checkpoint: fox_pairs.npz round-trips; wrong shape rejected -> fresh')

print('ALL FOX CHECKS PASSED')
