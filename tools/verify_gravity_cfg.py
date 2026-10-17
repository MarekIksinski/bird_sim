# Gate 1: compile + config constants + fresh-population shape guards.
import sys
sys.path.insert(0, '/home/marek/Playground/bird_sim')
sys.argv = ['main.py', '--cpu']
import config, predators, flock
print('NET_SHAPE', predators.NET_SHAPE, 'GENOME_LEN', predators.GENOME_LEN)
print('STARLING_NET_SHAPE', flock.STARLING_NET_SHAPE, 'STARLING_GENOME_LEN', flock.STARLING_GENOME_LEN)
assert predators.NET_SHAPE == (14, 16, 8, 4)
assert flock.STARLING_NET_SHAPE == (17, 16, 8, 6)
print('hawk gravity consts:', config.HAWK_GRAVITY, config.DIVE_MAX_SPEED,
      config.MIN_FLIGHT_SPEED, config.STALL_SPEED)
print('flock gravity consts:', config.FLOCK_GRAVITY, config.MIN_BIRD_SPEED,
      config.DIVE_SPEED_MULT, config.CLIMB_SPEED_MULT)
# checkpoint mismatch: old 12-input population must be rejected -> fresh start
import numpy as np, os
predators.load_checkpoint()
old_len = 12*16+16 + 16*8+8 + 8*4+4
if predators.population.shape[1] != predators.GENOME_LEN:
    print('FAIL: population did not match new GENOME_LEN')
    sys.exit(1)
print('population shape', predators.population.shape, '(fresh 14-input)' if predators.generation == 1 else f'(gen {predators.generation})')
print('starling tribe_pop', flock.tribe_pop.shape)
assert flock.tribe_pop.shape == (flock.NUM_TRIBES, flock.STARLING_GENOME_LEN)
print('CFG_GATE_OK')
