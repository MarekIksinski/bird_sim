# Gate 1: compile + config constants + fresh-population shape guards.
import sys
sys.path.insert(0, '/home/marek/Playground/bird_sim')
sys.argv = ['main.py', '--cpu']
import config, predators, flock, tomllib
with open('/home/marek/Playground/bird_sim/config.toml', 'rb') as f:
    _raw = tomllib.load(f)
# expected shapes derived FROM config.toml (user retunes hidden_layers live)
_hawk_h = tuple(_raw.get('hawk', {}).get('hidden_layers', [32, 16]))
_str_h = tuple(_raw.get('flock_nn', {}).get('hidden_layers', [32, 16]))
print('NET_SHAPE', predators.NET_SHAPE, 'GENOME_LEN', predators.GENOME_LEN)
print('STARLING_NET_SHAPE', flock.STARLING_NET_SHAPE, 'STARLING_GENOME_LEN', flock.STARLING_GENOME_LEN)
assert predators.NET_SHAPE == (15,) + _hawk_h + (4,), predators.NET_SHAPE
assert flock.STARLING_NET_SHAPE == (17,) + _str_h + (6,), flock.STARLING_NET_SHAPE
print('hawk gravity consts:', config.HAWK_GRAVITY, config.DIVE_MAX_SPEED,
      config.MIN_FLIGHT_SPEED, config.STALL_SPEED)
print('hawk stamina consts:', config.BOOST_DURATION, config.BOOST_RECHARGE,
      config.BOOST_COOLDOWN_RECOVER)
print('flock gravity consts:', config.FLOCK_GRAVITY, config.MIN_BIRD_SPEED,
      config.DIVE_SPEED_MULT, config.CLIMB_SPEED_MULT)
# checkpoint mismatch: old 12-input population must be rejected -> fresh start
import numpy as np, os
predators.load_checkpoint()
old_len = 12*16+16 + 16*8+8 + 8*4+4
if predators.population.shape[1] != predators.GENOME_LEN:
    print('FAIL: population did not match new GENOME_LEN')
    sys.exit(1)
print('population shape', predators.population.shape, '(fresh 15-input)' if predators.generation == 1 else f'(gen {predators.generation})')
print('starling tribe_pop', flock.tribe_pop.shape)
assert flock.tribe_pop.shape == (flock.NUM_TRIBES, flock.STARLING_GENOME_LEN)
print('CFG_GATE_OK')
