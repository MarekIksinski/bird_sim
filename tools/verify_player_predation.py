#!/usr/bin/env python3
"""verify_player_predation.py — the player hawk is a LIVE predator gate.

Imports the real predators/flock/rabbits/player modules (FORCE_CPU shim)
and checks:
 1. PANIC: player within PANIC_RADIUS of a starling (AI pack far away)
    produces a non-zero flee vector pointing AWAY from the player and
    nearest_d == the player distance.
 2. CATCH: player within CATCH_RADIUS sets all_caught, bumps
    player.player_catches, and leaves predators.gen_kills / fitness /
    total_catches / hawk_dmin untouched (AI fitness is pack-rows-only).
    The Darwinian tribe bookkeeping (_tribe_respawn_check) still runs for
    player-caught birds: survival zeroed + tribe_casualties incremented.
 3. RABBITS: player within RABBIT_FLEE_RADIUS of a foraging rabbit
    (AI pack far away) flips rabbit_state to 3 (Flee) and it sprints to
    the nearest burrow; the pack distance alone must NOT have done it.
 4. Source guard: hawk_dmin line slices d[:POPULATION_SIZE] (player row
    can never leak into AI proximity/dogfight fitness).
All radii are read from config (user retunes config.toml live — never
hardcode 12/1.2/25 as expectations).
"""
import os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ['FORCE_CPU'] = '1'

import numpy as np
from ursina import time as u_time
import config

TMP = tempfile.mkdtemp(prefix='birdsim_playerpred_')
config.EVOLVE_DIR = TMP          # before predators import: ckpt + log land here
import predators, flock, rabbits, player, terrain

# offscreen harness: pin dt like the other harnesses (no app.run loop here)
u_time.dt = 1.0 / 60.0

# redirect every persistence path (skill rule: _LOG_FILE too)
predators._CKPT = os.path.join(TMP, 'population.npz')
predators._HISTORY = os.path.join(TMP, 'history.csv')
predators._STARLING_CKPT = os.path.join(TMP, 'starling_tribes.npz')
predators._LOG_FILE = os.path.join(TMP, 'evolution.log')

# --- scenario scaffolding --------------------------------------------------
# AI pack parked mid-arena at x=400 (well inside the boundary/ground
# scrape bands so it accrues ZERO penalties, and >100m from every probe
# bird/rabbit so it is never the nearest predator). update_predators
# integrates pack physics with the fresh random NN every call, so re-pin
# it every frame before update_pack (same trick as the stamina harnesses).
PACK_PIN = np.array([400.0, 60.0, 0.0], dtype=np.float32)

def pin_pack():
    predators.pack_pos[:] = PACK_PIN
    predators.pack_vel[:] = 0.0
    predators.pack_speed[:] = 0.0

ORIGIN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
pos_np = np.zeros((config.NUM_BIRDS, 3), dtype=np.float32)
pos_np[:, 0] = 200.0        # all other birds far from both the player and
pos_np[:, 1] = 30.0         # the pack pin (> proximity_radius, zero bonus)
pos_np[:, 2] = 400.0
pos_np[0] = ORIGIN          # the probe bird
vel_np = np.zeros((config.NUM_BIRDS, 3), dtype=np.float32)
vel_np[:, 0] = 5.0            # nonzero speeds (avoids degenerate normals)


def run_frame():
    pin_pack()
    return predators.update_pack(pos_np, vel_np)


# --- 1. panic / flee vector from the PLAYER alone --------------------------
PANIC = float(config.PANIC_RADIUS)
pd = PANIC * 0.5
player.player_pos[:] = ORIGIN - np.array([pd, 0.0, 0.0], dtype=np.float32)
flee, all_caught, caught_by, nearest_dir, nearest_d = run_frame()

assert not all_caught.any(), 'no catch expected at half a panic radius'
assert abs(float(nearest_d[0]) - pd) < 1e-3, \
    f"nearest_d[0]={nearest_d[0]} expected player distance {pd}"
f0 = flee[0]
assert np.linalg.norm(f0) > 1e-3, 'flee vector is zero — player not perceived'
# flee points AWAY from the player (player sits at -X of the bird);
# the vector is the unit away-dir scaled by the panic ramp, so normalize.
assert f0[0] > 0 and abs(f0[0] / np.linalg.norm(f0) - 1.0) < 1e-6, \
    f'flee direction wrong: {f0}'
# strength matches the panic ramp (1 - d/PANIC)
assert abs(np.linalg.norm(f0) - (1.0 - pd / PANIC)) < 1e-4, \
    f'flee magnitude {np.linalg.norm(f0)} != ramp {1 - pd/PANIC}'
# every other bird (500m away) must NOT panic
assert np.linalg.norm(flee[1:]) < 1e-6, 'unexpected panic far from the player'
print(f'PASS panic: player at {pd:.1f}m -> flee={np.round(f0, 4)}, '
      f'nearest_d={nearest_d[0]:.3f} (PANIC_RADIUS={PANIC})')

# --- 2. catch by the player: counts split, AI fitness untouched ------------
CATCH = float(config.CATCH_RADIUS)
predators.fitness[:] = 0.0
predators.gen_kills[:] = 0
predators.total_catches = 0
player.player_catches = 0

player.player_pos[:] = ORIGIN - np.array([CATCH * 0.5, 0.0, 0.0], dtype=np.float32)
flee, all_caught, caught_by, nearest_dir, nearest_d = run_frame()

assert all_caught[0], 'player catch did not set all_caught'
assert int(caught_by[0]) == predators.POPULATION_SIZE, \
    f'caught_by[0]={caught_by[0]} expected {predators.POPULATION_SIZE} (player slot)'
assert not all_caught[1:].any(), 'unexpected AI catch (pack parked 1000m away)'
assert player.player_catches == 1, f'player_catches={player.player_catches}'
assert predators.gen_kills.sum() == 0, f'gen_kills polluted: {predators.gen_kills}'
assert predators.total_catches == 0, 'total_catches polluted by a player catch'
# hawk_dmin is a local of update_pack — verify it NUMERICALLY instead:
# a player-contaminated hawk_dmin (the player sits AT the probe bird)
# would fire the proximity ramp (PROXIMITY_GAIN * ~1.0/frame) for one
# hawk, and the dogfight range term likewise. fitness staying exactly 0
# proves the pack-row-only slice held. (Source guard #4 pins the line.)
assert np.all(predators.fitness == 0.0), \
    'proximity/dogfight bonus leaked — hawk_dmin is not pack-rows-only'
print('PASS catch split: player_catches=1, gen_kills=0, fitness=0, '
      'total_catches=0 (hawk_dmin pack-only, proven via zero fitness)')

# --- 2b. Darwinian bookkeeping runs for PLAYER-caught birds ----------------
flock.bird_survival_time[0] = 123.4
flock.bird_score[0] = 55.0
tribe0 = int(flock.bird_tribe_np[0])
cas_before = float(flock.tribe_casualties[tribe0])
flock._tribe_respawn_check(np.where(all_caught)[0])
assert flock.bird_survival_time[0] == 0.0 and flock.bird_score[0] == 0.0, \
    'player-caught bird not zeroed by _tribe_respawn_check'
assert float(flock.tribe_casualties[tribe0]) > cas_before, \
    'tribe_casualties not incremented for a player catch'
print(f'PASS tribe bookkeeping: T{tribe0} casualties '
      f'{cas_before:.3f} -> {flock.tribe_casualties[tribe0]:.3f}')

# --- 3. rabbits react to the player hawk -----------------------------------
FR = float(config.RABBIT_FLEE_RADIUS)
rabbits.rabbit_pos[:] = 0.0
rabbits.rabbit_pos[:, 1] = terrain.height_at(rabbits.rabbit_pos[:, 0], rabbits.rabbit_pos[:, 2])
rabbits.rabbit_pos[:, 0] = 900.0     # everyone far off...
rabbits.rabbit_pos[:, 2] = 900.0
rabbits.rabbit_pos[0, 0] = 0.0       # probe rabbit at x=0
rabbits.rabbit_pos[0, 2] = 0.0
rabbits.rabbit_pos[0, 1] = terrain.height_at(0.0, 0.0)
rabbits.rabbit_vel[:] = 0.0
rabbits.rabbit_state[:] = 0          # Forage
rabbits.rabbit_timer[:] = 10.0
# park the ground predators far from BOTH the probe rabbit (0,0) and the
# rabbit crowd (900,900): foxes/wolves postdate this harness and rabbits
# now fold their proximity into the threat distance d — an unparked fox
# near the probe fakes "Flee from the player" and near the crowd breaks
# the only-one-fleeing assert.
import foxes as _foxes_mod, wolves as _wolves_mod
_foxes_mod.fox_pos[:] = np.array([500.0, 0.0, -500.0], dtype=np.float32)
_wolves_mod.wolf_pos[:] = np.array([-500.0, 0.0, 500.0], dtype=np.float32)
_foxes_mod.fox_speed[:] = 0.0        # creeping: minimal stealth rescale
_wolves_mod.wolf_speed[:] = 0.0

player.player_pos[:] = np.array([FR * 0.5, float(rabbits.rabbit_pos[0, 1]) + 2.0, 0.0],
                                dtype=np.float32)
rabbits.update_rabbits()
assert int(rabbits.rabbit_state[0]) == 3, \
    f'rabbit_state[0]={rabbits.rabbit_state[0]} expected 3 (Flee) from the player'
assert int((rabbits.rabbit_state == 3).sum()) == 1, \
    'only the probe rabbit may flee (pack parked far, other rabbits far)'
# it sprints toward the nearest burrow
rxz = rabbits.rabbit_pos[0][[0, 2]]
rel = rabbits.burrow_xz - rxz[None, :]
nb = int(np.argmin(np.einsum('ij,ij->i', rel, rel)))
want = rel[nb] / max(np.linalg.norm(rel[nb]), 1e-6)
got = rabbits.rabbit_vel[0][[0, 2]]
got_n = got / max(np.linalg.norm(got), 1e-6)
assert np.dot(want, got_n) > 0.99, f'flee heading wrong: {got_n} vs {want}'
spd = np.linalg.norm(got)
assert abs(spd - config.RABBIT_FLEE_SPEED) < 1e-3, f'flee speed {spd}'
print(f'PASS rabbits: player at {FR*0.5:.1f}m -> state 3 (Flee), '
      f'sprint {spd:.2f} m/s toward burrow {nb}')

# control: move the player far away, rabbit recovers via Vigilance path,
# and the pack alone (far away) keeps it out of Flee
player.player_pos[:] = 0.0
player.player_pos[2] = 1000.0
rabbits.update_rabbits()
assert int(rabbits.rabbit_state[0]) != 3, 'rabbit still fleeing with no threat nearby'
print('PASS rabbit recovery: no threat -> left Flee state')

# --- 4. source guard: hawk_dmin restricted to pack rows --------------------
src = open(os.path.join(ROOT, 'predators.py'), encoding='utf-8').read()
assert 'hawk_dmin = np.minimum(hawk_dmin, d[:POPULATION_SIZE].min(axis=1))' in src, \
    'hawk_dmin no longer slices the pack rows — player could leak into AI fitness'
print('PASS source guard: hawk_dmin packs AI rows only')

print('\nALL PASS verify_player_predation')
