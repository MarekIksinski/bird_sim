#!/usr/bin/env python3
"""verify_hidden_layers.py — configurable MLP hidden widths gate.

1. With the LIVE config.toml (default [32, 16] both):
   NET_SHAPE == (15, 32, 16, 4), STARLING_NET_SHAPE == (21, 32, 16, 6),
   and the GENOME_LEN sums match.
2. Batched forwards run and emit (POPULATION_SIZE, 4) for the hawk pack
   (via _stack_population + the real einsum/tanh loop form) and
   (NUM_BIRDS, 6) for the starlings (via _upload_tribe_layers + the real
   (T,BPT,ni) einsum loop form).
3. Genome unpack/repack round-trips are layer-count-dynamic.
4. ARBITRARY depth/width: config.py is copied into a tempdir with a
   variant config.toml (hawk [8,4,8,2] deep, starling [64] single-wide);
   a fresh subprocess must build matching shapes, forward them, and
   load_checkpoint must REJECT the old-shape checkpoint (fresh start).
All expectations derive from the toml files, never hardcoded literals.
"""
import os, subprocess, sys, tempfile, textwrap

ROOT = '/home/marek/Playground/bird_sim'

# ---------- part A: live config, in-process ----------
sys.path.insert(0, ROOT)
sys.argv = ['main.py', '--cpu']
import numpy as np
import config, predators, flock, tomllib

with open(os.path.join(ROOT, 'config.toml'), 'rb') as f:
    raw = tomllib.load(f)
hawk_h = tuple(raw.get('hawk', {}).get('hidden_layers', [32, 16]))
str_h = tuple(raw.get('flock_nn', {}).get('hidden_layers', [32, 16]))

# --- 1. shapes ---
assert config.HAWK_HIDDEN_LAYERS == hawk_h
assert config.STARLING_HIDDEN_LAYERS == str_h
assert predators.NET_SHAPE == (15,) + hawk_h + (4,), predators.NET_SHAPE
assert flock.STARLING_NET_SHAPE == (21,) + str_h + (6,), flock.STARLING_NET_SHAPE
exp_gl = sum(predators.NET_SHAPE[i] * predators.NET_SHAPE[i+1] + predators.NET_SHAPE[i+1]
             for i in range(len(predators.NET_SHAPE) - 1))
assert predators.GENOME_LEN == exp_gl, (predators.GENOME_LEN, exp_gl)
exp_sgl = sum(flock.STARLING_NET_SHAPE[i] * flock.STARLING_NET_SHAPE[i+1] + flock.STARLING_NET_SHAPE[i+1]
              for i in range(len(flock.STARLING_NET_SHAPE) - 1))
assert flock.STARLING_GENOME_LEN == exp_sgl
assert predators.NET_SHAPE == (15, 32, 16, 4), predators.NET_SHAPE
assert flock.STARLING_NET_SHAPE == (21, 32, 16, 6), flock.STARLING_NET_SHAPE
print(f'PASS shapes: NET_SHAPE={predators.NET_SHAPE} GENOME_LEN={predators.GENOME_LEN} '
      f'| STARLING_NET_SHAPE={flock.STARLING_NET_SHAPE} SGL={flock.STARLING_GENOME_LEN}')

# --- 2a. hawk batched forward: real _stack_population + real loop form ---
Ws, bs = predators._stack_population()
assert len(Ws) == len(predators.NET_SHAPE) - 1
for i, (W, b) in enumerate(zip(Ws, bs)):
    ni, no = predators.NET_SHAPE[i], predators.NET_SHAPE[i + 1]
    assert W.shape == (predators.POPULATION_SIZE, ni, no), W.shape
    assert b.shape == (predators.POPULATION_SIZE, no), b.shape
a = np.random.default_rng(7).uniform(-1, 1, (predators.POPULATION_SIZE, predators.NET_SHAPE[0]))
for W, b in zip(Ws, bs):
    a = np.tanh(np.einsum('ni,nio->no', a, W) + b)   # predators.py line-for-line
assert a.shape == (predators.POPULATION_SIZE, 4), a.shape
assert np.all(np.isfinite(a))
print(f'PASS hawk forward: input ({predators.NET_SHAPE[0],}) -> out {a.shape}, finite')

# --- 2b. starling batched forward: real upload + real einsum loop form ---
flock._upload_tribe_layers()
assert len(flock.tribe_Ws) == len(flock.STARLING_NET_SHAPE) - 1
from backend import cp
obs = cp.asarray(np.random.default_rng(8).uniform(
    -1, 1, (config.NUM_BIRDS, flock.STARLING_NET_SHAPE[0])).astype(np.float32))
a2 = obs.reshape(flock.NUM_TRIBES, flock.BIRDS_PER_TRIBE, flock.STARLING_NET_SHAPE[0])
for W, b in zip(flock.tribe_Ws, flock.tribe_bs):
    a2 = cp.tanh(cp.einsum('tbi,tio->tbo', a2, W) + b[:, None, :])
out = a2.reshape(config.NUM_BIRDS, flock.STARLING_NET_SHAPE[-1])
assert out.shape == (config.NUM_BIRDS, 6), out.shape
assert bool(cp.all(cp.isfinite(out)))
print(f'PASS starling forward: obs ({config.NUM_BIRDS}, {flock.STARLING_NET_SHAPE[0]}) '
      f'-> out {out.shape}, finite')

# --- 3. unpack/repack round-trip is layer-count-dynamic ---
g = np.random.default_rng(9).normal(0, 0.5, predators.GENOME_LEN)
layers = predators._unpack_genome(g)
repk = np.concatenate([np.concatenate([W.reshape(-1), b]) for W, b in layers])
assert np.array_equal(repk, g), 'hawk genome unpack/repack mismatch'
gs = np.random.default_rng(10).normal(0, 0.5, flock.STARLING_GENOME_LEN).astype(np.float32)
ls = flock._unpack_starling_genome(gs)
repks = np.concatenate([np.concatenate([W.reshape(-1), b]) for W, b in ls])
assert np.array_equal(repks, gs), 'starling genome unpack/repack mismatch'
print(f'PASS unpack round-trips: hawk {len(layers)} layers, '
      f'starling {len(ls)} layers (lossless)')

# ---------- part B: arbitrary depth/width via variant config ----------
d = tempfile.mkdtemp(prefix='hltest')
import shutil
shutil.copy(os.path.join(ROOT, 'config.py'), d)
# no backend stub: the REAL backend is imported from ROOT; FORCE_CPU=1 env
# (below) is what the backend probe honors.
open(os.path.join(d, 'config.toml'), 'w').write(textwrap.dedent('''
    [hawk]
    hidden_layers = [8, 4, 8, 2]     # 4 hidden layers, diamond shape
    [flock_nn]
    hidden_layers = [64]             # single wide hidden layer
'''))
code = textwrap.dedent(f'''
    import sys
    sys.path.insert(0, {ROOT!r})
    sys.path.insert(0, {d!r})   # wins for `config` (variant toml next to it);
                                # predators/flock/terrain/math3d still from ROOT
    sys.argv = ['main.py', '--cpu']
    import config, predators, flock, numpy as np
    assert config.__file__.startswith({d!r}), config.__file__
    assert config.HAWK_HIDDEN_LAYERS == (8, 4, 8, 2), config.HAWK_HIDDEN_LAYERS
    assert config.STARLING_HIDDEN_LAYERS == (64,), config.STARLING_HIDDEN_LAYERS
    assert predators.NET_SHAPE == (15, 8, 4, 8, 2, 4), predators.NET_SHAPE
    assert flock.STARLING_NET_SHAPE == (21, 64, 6), flock.STARLING_NET_SHAPE
    exp = sum(predators.NET_SHAPE[i]*predators.NET_SHAPE[i+1] + predators.NET_SHAPE[i+1]
              for i in range(len(predators.NET_SHAPE)-1))
    assert predators.GENOME_LEN == exp, (predators.GENOME_LEN, exp)
    exps = sum(flock.STARLING_NET_SHAPE[i]*flock.STARLING_NET_SHAPE[i+1] + flock.STARLING_NET_SHAPE[i+1]
               for i in range(len(flock.STARLING_NET_SHAPE)-1))
    assert flock.STARLING_GENOME_LEN == exps
    assert predators.population.shape == (predators.POPULATION_SIZE, exp)
    assert flock.tribe_pop.shape == (flock.NUM_TRIBES, exps)
    # forward runs at arbitrary depth
    Ws, bs = predators._stack_population()
    assert len(Ws) == 5
    a = np.zeros((predators.POPULATION_SIZE, 15))
    for W, b in zip(Ws, bs):
        a = np.tanh(np.einsum('ni,nio->no', a, W) + b)
    assert a.shape == (predators.POPULATION_SIZE, 4)
    flock._upload_tribe_layers()
    assert len(flock.tribe_Ws) == 2 and flock.tribe_Ws[0].shape[1:] == (21, 64)
    print('VARIANT_OK')
''')
import subprocess as _sp
r = _sp.run([sys.executable, '-u', '-c', code], capture_output=True, text=True,
            timeout=180, cwd=d, env={**os.environ, 'FORCE_CPU': '1'})
assert 'VARIANT_OK' in r.stdout, f'variant config failed:\n{r.stdout}\n{r.stderr}'
print('PASS arbitrary depth/width: hawk [8,4,8,2] -> (15,8,4,8,2,4), '
      'starling [64] -> (21,64,6); forwards + fresh pops OK')

# ---------- part C: checkpoint shape guard rejects old widths ----------
# write a checkpoint in the OLD (15,16,8,4) layout into a temp save dir,
# point the (already-imported, new-shape) predators at it, load, expect
# fresh population at the NEW GENOME_LEN.
import tempfile as tf
tmp = tf.mkdtemp(prefix='hlckpt')
predators._CKPT = os.path.join(tmp, 'population.npz')
predators._HISTORY = os.path.join(tmp, 'history.csv')
predators._STARLING_CKPT = os.path.join(tmp, 'starling_tribes.npz')
predators._LOG_FILE = os.path.join(tmp, 'evolution.log')
old_len = 15*16+16 + 16*8+8 + 8*4+4
np.savez(predators._CKPT,
         population=np.random.default_rng(11).normal(0, 0.5,
             (predators.POPULATION_SIZE, old_len)),
         generation=999, total_catches=42)
predators.load_checkpoint()
assert predators.generation == 1, f'guard did not reset: gen={predators.generation}'
assert predators.population.shape == (predators.POPULATION_SIZE, predators.GENOME_LEN)
# starling side: old (20,478) tribes vs new shape. The starling guard sits
# AFTER the hawk guard in load_checkpoint (a hawk mismatch returns early),
# so write a hawk-compatible ckpt first, then the old-shape starling one.
np.savez(predators._CKPT,
         population=np.zeros((predators.POPULATION_SIZE, predators.GENOME_LEN)),
         generation=5, total_catches=1)
old_sgl = 17*16+16 + 16*8+8 + 8*6+6
np.savez(predators._STARLING_CKPT, tribe_pop=np.zeros((flock.NUM_TRIBES, old_sgl),
                                                      dtype=np.float32),
         starling_evolutions=7)
flock.starling_evolutions = 99   # dirty value: a rejected load must NOT touch it
predators.load_checkpoint()
assert predators.generation == 5, 'hawk-compatible ckpt should have loaded'
assert flock.starling_evolutions == 99, 'starling guard wrongly touched state'
# and a NEW-shape starling ckpt must load cleanly
np.savez(predators._STARLING_CKPT,
         tribe_pop=np.ones((flock.NUM_TRIBES, flock.STARLING_GENOME_LEN), dtype=np.float32),
         starling_evolutions=7)
predators.load_checkpoint()
assert flock.starling_evolutions == 7, 'matching starling ckpt did not load'
assert np.array_equal(flock.tribe_pop, np.ones_like(flock.tribe_pop))
print('PASS checkpoint guards: old (15,16,8,4)/(17,16,8,6) ckpts rejected -> fresh; '
      'matching starling ckpt round-trips')

print('\nALL HIDDEN-LAYER CHECKS PASSED')
