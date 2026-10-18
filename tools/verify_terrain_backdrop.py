#!/usr/bin/env python3
"""verify_terrain_backdrop.py — height_at = max(playfield, backdrop) gate.

Checks (pure NumPy, no Ursina window needed beyond terrain's import):
 1. heightmap / backdrop_hm meshes are BIT-IDENTICAL to the pre-refactor
    hashes (visual shell must not shift) — recorded 2026-10.
 2. At the arena center height_at == _play_height_at (backdrop is 3m under
    the play surface there).
 3. In the four arena corners (|x|,|z| ~ 0.48*TERRAIN_SIZE, r > SIZE/2)
    height_at == backdrop_height_at > _play_height_at (the encroaching
    ridges are now solid ground).
 4. Scalar (float,float) AND 1D-array calls agree element-wise, and the
    backdrop feathering invariant still holds inside r < TERRAIN_SIZE/2
    (backdrop == play - 3 -> max() is a no-op over the whole interior).
"""
import os, sys, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.argv = ['main.py', '--cpu']
import numpy as np
import terrain
from config import TERRAIN_SIZE, TERRAIN_AMP, BACKDROP_AMP

# --- 1. meshes bit-identical: the visual shell must not have shifted ------
h_hash = hashlib.sha256(terrain.heightmap.tobytes()).hexdigest()
b_hash = hashlib.sha256(terrain.backdrop_hm.tobytes()).hexdigest()
assert h_hash.startswith('e5a3c7f9ec2f6ca5'), f'heightmap changed: {h_hash}'
# backdrop pin updated 2026-10-18: user retuned BACKDROP_RES 64 -> 32 in
# config.py (deliberate coarser/cheaper ring). Playfield hash unchanged.
assert b_hash.startswith('b68adda4b21cacbc'), f'backdrop_hm changed: {b_hash}'
print('PASS meshes bit-identical (heightmap e5a3c7f9..., backdrop b68adda4...)')

# --- 2. center: play surface wins (backdrop = play - 3 there) -------------
hc = terrain.height_at(0.0, 0.0)
pc = terrain._play_height_at(0.0, 0.0)
bc = terrain.backdrop_height_at(0.0, 0.0)
assert hc == pc and np.isclose(pc - bc, 3.0, atol=0.05), (hc, pc, bc)
print(f'PASS center: height_at==play ({hc:.2f}), backdrop 3m below ({bc:.2f})')

# --- 3. arena corners: backdrop encroaches and IS the ground --------------
# blend starts at r = TERRAIN_SIZE/2; corners at 0.48*SIZE per axis sit at
# r ~ 0.679*SIZE, well into the rising ring. The ring is noise (not a
# perfect circle), so scan a grid in the four corner regions instead of
# asserting every exact corner point encroaches.
corner = 0.48 * TERRAIN_SIZE
encroach = 0
for sx in (-1, 1):
    for sz in (-1, 1):
        hit = None
        for dx in (-24.0, 0.0, 24.0):
            for dz in (-24.0, 0.0, 24.0):
                x, z = sx * (corner + dx), sz * (corner + dz)
                h = float(terrain.height_at(x, z))
                p = float(terrain._play_height_at(x, z))
                b = float(terrain.backdrop_height_at(x, z))
                assert h == max(p, b), (x, z, h, p, b)
                if b > p:
                    assert h == b, (x, z, h, b, p)
                    if hit is None:
                        hit = (x, z, b, p)
        assert hit is not None, f'corner quadrant ({sx},{sz}) has no encroaching ridge'
        encroach += 1
        print(f'  corner ({hit[0]:+.0f},{hit[1]:+.0f}): backdrop {hit[2]:.1f} > '
              f'play {hit[3]:.1f} -> height_at {hit[2]:.1f}')
assert encroach == 4, f'backdrop must encroach all four corner regions (got {encroach}/4)'
print('PASS corners: backdrop encroaches all 4 corner regions, height_at == backdrop there')

# --- 4. scalar vs 1D array parity + interior no-op invariant --------------
rng = np.random.default_rng(42)
xs = rng.uniform(-TERRAIN_SIZE / 2, TERRAIN_SIZE / 2, 500)
zs = rng.uniform(-TERRAIN_SIZE / 2, TERRAIN_SIZE / 2, 500)
arr = terrain.height_at(xs, zs)
assert arr.shape == (500,) and np.all(np.isfinite(arr))
for k in range(0, 500, 37):
    assert float(terrain.height_at(float(xs[k]), float(zs[k]))) == float(arr[k]), k
# mixed shapes: (N,)-array x scalar (predators/flock never do this, but
# the pattern must not crash)
mix = terrain.height_at(xs, 123.0)
assert mix.shape == (500,) and np.all(np.isfinite(mix))
# interior invariant: at every backdrop LATTICE vertex with r < SIZE/2 the
# feathering is exactly play-3 (construction unchanged -> the visual shell
# is untouched). Between coarse 62.5m vertices the bilinear surface can
# locally bulge a few meters above the fine play surface — that IS the
# rendered backdrop mesh, so max() matching it is correct, not a bug.
vi = (terrain._br < TERRAIN_SIZE / 2 - 10)
assert np.allclose(terrain.backdrop_hm[vi],
                   terrain._play_height_at(terrain._bx[vi], terrain._bz[vi]) - 3.0,
                   atol=1e-5)
# everywhere inside the walls height_at >= play, and the lift above play is
# bounded by the backdrop's coarse-grid interpolation error. The bound
# SCALES with the backdrop cell size (bilinear bulge grows ~linearly with
# cell size; ~5m at RES 64 = 62.5m cells): derive it from config, since
# BACKDROP_RES is a live knob the user retunes.
from config import BACKDROP_RES
r = np.sqrt(xs**2 + zs**2)
inside = r < TERRAIN_SIZE / 2 - 10
h_in = terrain.height_at(xs[inside], zs[inside])
p_in = terrain._play_height_at(xs[inside], zs[inside])
lift = h_in - p_in
lift_bound = 8.0 * 64.0 / BACKDROP_RES
assert np.all(lift >= -1e-9) and lift.max() < lift_bound, \
    (float(lift.min()), float(lift.max()), lift_bound)
print(f'PASS interior: vertex feathering exact (play-3), max() lift bounded '
      f'{lift.max():.2f}m (< {lift_bound:.1f}m for RES={BACKDROP_RES}) '
      f'= rendered backdrop surface, never below play')

print('ALL TERRAIN-BACKDROP CHECKS PASSED')
