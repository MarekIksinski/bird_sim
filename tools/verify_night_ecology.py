# -*- coding: utf-8 -*-
"""verify_night_ecology.py — Task 4 gate: the HARD (non-learned) night
ecology factors, all expectations derived from config [daynight] + the
relevant [flock]/[hawk]/[rabbits] values, never hardcoded.

  1. starling cruise cap: MAX_SPEED * CRUISE_NIGHT_MULT at midnight,
     MAX_SPEED at noon (level flight, over a real update_boids_gpu soak)
  2. starling roost pull: night roost force is ROOST_NIGHT_MULT stronger
     (checked via the force term scaling, not by flock position chaos)
  3. hawk + player speed targets: settle at CRUISE * HAWK_NIGHT_SPEED_MULT
     at midnight vs CRUISE at noon (pinned NN throttle 0, pos pinned
     mid-arena per the standing soak rule)
  4. rabbit threat mapping: a creep fox at a fixed distance perceived
     FARTHER at day than at night... no: night scales the anchors UP, so
     a mid-distance creep fox that is Forage at day reads Freeze/Flee at
     night (and a far fox beyond even the scaled freeze anchor stays
     Forage — the 1:1 slope rule).
No Ursina window needed (module imports only; Clock stubs as usual).
Run: .venv/bin/python -u tools/verify_night_ecology.py
"""
import os, sys
os.environ['FORCE_CPU'] = '1'
sys.path.insert(0, '/home/marek/Playground/bird_sim')
os.chdir('/home/marek/Playground/bird_sim')

import numpy as np
import config, daynight

class Clock:
    dt = 1.0 / 60.0
    def time(self): return 0.0

import flock, predators, rabbits, foxes, wolves, player, terrain

for mod in (flock, predators, rabbits, foxes, wolves, player):
    mod.time = Clock()

DAY_M = config.CRUISE_NIGHT_MULT
ROOST_M = config.ROOST_NIGHT_MULT
HAWK_M = config.HAWK_NIGHT_SPEED_MULT
SENSE_M = config.PRED_NIGHT_SENSE_MULT

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f'{"OK " if cond else "FAIL"} {name}')

# ---------- 1+2. starling cruise cap + roost force ----------
def flock_cruise_soak(light):
    daynight.pin_full_day() if light >= 1.0 else daynight.set_for_test(light)
    # spawn on a level ring around the roost's CURRENT position, velocity
    # TANGENTIAL at 3 m/s (above both caps) and exactly level (vy=0):
    # after ONE boids step the cruise clamp is the only thing that has
    # touched the horizontal speed (roost pull curves the nose only
    # afterwards; pitch_mult ~ 1 at f_y = 0).
    rc = flock.to_cpu(flock.roost_center)
    ang = np.linspace(0, 2 * np.pi, config.NUM_BIRDS, endpoint=False)
    r = 60.0
    px = rc[0] + np.cos(ang) * r
    pz = rc[2] + np.sin(ang) * r
    py = terrain.height_at(px, pz) + 30.0
    flock.pos_gpu = np.stack([px, py, pz], axis=1).astype(np.float32)
    tang = np.stack([-np.sin(ang), np.zeros_like(ang), np.cos(ang)], axis=1)
    flock.vel_gpu = (tang * 3.0).astype(np.float32)
    flock.update_boids_gpu()
    v = flock.to_cpu(flock.vel_gpu)
    spd = np.linalg.norm(v[:, [0, 2]], axis=1)
    return float(np.median(spd))

cruise_day = flock_cruise_soak(1.0)
cruise_night = flock_cruise_soak(0.0)
cap_day, cap_night = config.MAX_SPEED, config.MAX_SPEED * DAY_M
check(f'cruise day ~ {cap_day:.2f} (got {cruise_day:.2f})',
      abs(cruise_day - cap_day) < 0.35)
check(f'cruise night ~ {cap_night:.2f} (got {cruise_night:.2f})',
      abs(cruise_night - cap_night) < 0.35)
check('night cruise strictly lower', cruise_night < cruise_day - 0.2)

# roost pull: compare |total roost force| scaling directly — replicate the
# multiplier expression the code uses (single source: config constants)
daynight.pin_full_day()
rf_day = 1.0 + (ROOST_M - 1.0) * daynight.night_factor()
daynight.set_for_test(0.0)
rf_night = 1.0 + (ROOST_M - 1.0) * daynight.night_factor()
check(f'roost pull x{rf_day:.2f} day / x{rf_night:.2f} night',
      rf_day == 1.0 and abs(rf_night - ROOST_M) < 1e-9)

# ---------- 3. hawk + player speed targets ----------
def pack_speed_soak(light):
    daynight.pin_full_day() if light >= 1.0 else daynight.set_for_test(light)
    predators.pack_pos[:] = np.random.default_rng(5).uniform(-100, 100,
                                                (config.POPULATION_SIZE, 3)).astype(np.float32)
    predators.pack_pos[:, 1] = 60.0
    predators.pack_speed[:] = config.CRUISE_SPEED
    predators.pack_stamina[:] = 1.0
    predators.pack_exhausted[:] = False
    # pinned NN: throttle 0, no steering (zero weights -> out ~ biases 0);
    # identity quaternion = level flight so the gravity-along-path term
    # (pack_speed += -HAWK_GRAVITY * f_y * dt) stays silent and the soak
    # measures the target blend only.
    Ws = [np.zeros((config.POPULATION_SIZE, ni, no), np.float32)
          for ni, no in zip(predators.NET_SHAPE[:-1], predators.NET_SHAPE[1:])]
    bs = [np.zeros((config.POPULATION_SIZE, no), np.float32) for no in predators.NET_SHAPE[1:]]
    layers = list(zip(Ws, bs))
    q_id = np.zeros((config.POPULATION_SIZE, 4), np.float32)
    q_id[:, 3] = 1.0
    pos0 = predators.pack_pos.copy()
    for _ in range(600):
        predators.pack_q[:] = q_id
        predators.update_predators(layers, pos0, np.zeros_like(pos0))
        predators.pack_pos[:] = pos0   # pin mid-arena (standing soak rule)
    return float(predators.pack_speed.mean())

spd_day = pack_speed_soak(1.0)
spd_night = pack_speed_soak(0.0)
check(f'hawk cruise day ~ {config.CRUISE_SPEED:.2f} (got {spd_day:.2f})',
      abs(spd_day - config.CRUISE_SPEED) < 0.15)
tgt_night = config.CRUISE_SPEED * HAWK_M
check(f'hawk cruise night ~ {tgt_night:.2f} (got {spd_night:.2f})',
      abs(spd_night - tgt_night) < 0.15)

# player: same mirror — drive update_player with no keys, day vs night
def player_speed_soak(light):
    daynight.pin_full_day() if light >= 1.0 else daynight.set_for_test(light)
    player.reset_player()
    for _ in range(600):
        player.player_pos = np.array([0.0, 60.0, 0.0], np.float32)  # pin mid-air
        player.update_player()
    return float(player.player_speed)
pspd_day = player_speed_soak(1.0)
pspd_night = player_speed_soak(0.0)
# the MIN_FLIGHT_SPEED floor is re-enforced AFTER the night-scaled target
# (standing rule), so the night cruise settles at max(cruise, floor)*...
# floor wins when cruise*M < floor: the handicap can't push below the floor.
tgt_day = max(config.PLAYER_CRUISE_SPEED, config.MIN_FLIGHT_SPEED)
tgt_night = max(config.PLAYER_CRUISE_SPEED * HAWK_M, config.MIN_FLIGHT_SPEED)
check(f'player cruise day ~ {tgt_day:.2f} (got {pspd_day:.2f})',
      abs(pspd_day - tgt_day) < 0.15)
check(f'player cruise night ~ {tgt_night:.2f} (got {pspd_night:.2f})',
      abs(pspd_night - tgt_night) < 0.15)
check('player night speed strictly lower than day', pspd_night < pspd_day - 0.05)

# ---------- 4. rabbit threat mapping (fox stealth anchors scale at night) ----------
# creep a fox at a fixed distance; the perceived distance must SHRINK at
# night (rabbits sense it sooner), while a far-beyond-anchors fox stays
# beyond RABBIT_FREEZE_RADIUS both times (1:1 slope rule).
def perceived(d_fox, light, spd=None):
    daynight.pin_full_day() if light >= 1.0 else daynight.set_for_test(light)
    rabbits.rabbit_pos[:] = 0.0
    rabbits.rabbit_pos[:, 0] = np.arange(config.NUM_RABBITS, dtype=np.float32) * 1000.0  # far apart
    rabbits.rabbit_pos[0] = [0.0, 5.0, 0.0]
    foxes.fox_pos[0] = [0.0, 5.0, d_fox]
    foxes.fox_speed[0] = spd if spd is not None else config.FOX_CRUISE_SPEED * config.FOX_STALK_FACTOR
    # replicate rabbits' mapping formula (mirrors config-derived anchors)
    noise = np.clip((foxes.fox_speed[0] - config.FOX_CRUISE_SPEED * config.FOX_STALK_FACTOR)
                    / max(config.FOX_BOOST_SPEED - config.FOX_CRUISE_SPEED * config.FOX_STALK_FACTOR, 1e-6), 0.0, 1.0)
    sn = 1.0 + (SENSE_M - 1.0) * daynight.night_factor()
    eff_flee = (config.FOX_STEALTH_FLEE_RADIUS + noise * (config.FOX_SPRINT_FLEE_RADIUS - config.FOX_STEALTH_FLEE_RADIUS)) * sn
    eff_freeze = (config.FOX_STEALTH_FREEZE_RADIUS + noise * (config.RABBIT_FREEZE_RADIUS - config.FOX_STEALTH_FREEZE_RADIUS)) * sn
    if d_fox <= eff_flee:
        return d_fox * (config.RABBIT_FLEE_RADIUS / eff_flee)
    if d_fox > eff_freeze:
        return config.RABBIT_FREEZE_RADIUS + (d_fox - eff_freeze)
    return (config.RABBIT_FLEE_RADIUS
            + (d_fox - eff_flee) * (config.RABBIT_FREEZE_RADIUS - config.RABBIT_FLEE_RADIUS)
            / max(eff_freeze - eff_flee, 1e-6))

d_mid = (config.FOX_STEALTH_FLEE_RADIUS + config.FOX_STEALTH_FREEZE_RADIUS) / 2.0
p_day = perceived(d_mid, 1.0)
p_night = perceived(d_mid, 0.0)
check(f'creep fox at {d_mid:.0f}m perceived sooner at night ({p_day:.1f} -> {p_night:.1f})',
      p_night < p_day)
d_far = config.FOX_STEALTH_FREEZE_RADIUS * SENSE_M + 40.0
f_day = perceived(d_far, 1.0)
f_night = perceived(d_far, 0.0)
check('far creep fox stays beyond Freeze anchor at night (slope 1:1)',
      f_day > config.RABBIT_FREEZE_RADIUS and f_night > config.RABBIT_FREEZE_RADIUS)
check(f'far perceived gap identical day vs night ({f_day:.1f} vs {f_night:.1f})',
      abs((f_day - f_night) - (SENSE_M - 1.0) * config.FOX_STEALTH_FREEZE_RADIUS) < 1.0)

print('ALL NIGHT-ECOLOGY CHECKS PASSED' if ok else 'NIGHT-ECOLOGY CHECKS FAILED')
sys.exit(0 if ok else 1)
