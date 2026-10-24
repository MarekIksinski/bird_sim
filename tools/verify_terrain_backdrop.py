#!/usr/bin/env python3
"""verify_terrain_backdrop.py — height_at = max(playfield, backdrop) gate
with the SQUARE deep-floor cutout (post clipping-fix, 2026-10-24).

Checks (pure NumPy, no Ursina window needed beyond terrain's import):
 1. play heightmap BIT-IDENTICAL to its pin (the fix only touches
    backdrop_hm); backdrop_hm re-pinned for the cutout model.
 2. At the arena center height_at == _play_height_at and the backdrop sits
    at the deep floor (-TERRAIN_AMP - 25), not the old feathered play-3.
 3. Past the walls the ring still IS the ground: where the ramped backdrop
    exceeds the (clamped-edge) playfield, height_at == backdrop.
 4. Scalar (float,float) AND 1D-array calls agree element-wise, and the
    deep-floor lattice invariant holds inside the square cutout.
"""
import os, sys, hashlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.argv = ['main.py', '--cpu']
import numpy as np
import terrain
import config
from config import TERRAIN_SIZE, TERRAIN_AMP, BACKDROP_AMP

# --- 1. play heightmap bit-identical; backdrop pin = cutout model --------
h_hash = hashlib.sha256(terrain.heightmap.tobytes()).hexdigest()
b_hash = hashlib.sha256(terrain.backdrop_hm.tobytes()).hexdigest()
# pins RE-RECORDED 2026-10-24: user retuned cell 0.6->0.8 since the 10-18
# pin (play mesh shifted then, NOT by this fix), and the backdrop pin moves
# to the square deep-floor cutout — the old feathered-center mesh (center ==
# play-3) is intentionally retired: it was the clipping bug.
assert h_hash.startswith('35a03c2a52438b5e'), f'heightmap changed: {h_hash}'
assert b_hash.startswith('73f10d8d51cb6f5b'), f'backdrop_hm changed: {b_hash}'
print('PASS meshes pinned (heightmap 35a03c2a..., backdrop cutout 73f10d8d...)')

# --- 2. center: play surface wins, backdrop at the deep floor ------------
hc = terrain.height_at(0.0, 0.0)
pc = terrain._play_height_at(0.0, 0.0)
bc = terrain.backdrop_height_at(0.0, 0.0)
assert hc == pc and np.isclose(bc, -config.TERRAIN_AMP - 25.0, atol=1e-3), (hc, pc, bc)
print(f'PASS center: height_at==play ({hc:.2f}), backdrop at deep floor ({bc:.2f})')

# --- 3. past the walls: the ring still IS the ground ---------------------
# Under the cutout model the backdrop no longer encroaches INSIDE the arena
# (that was the bug). Physics clamps entities to |x|,|z| <= TERRAIN_SIZE/2,
# but the ring beyond Chebyshev >= _bd_cut + _bd_ramp is raw noise and must
# still dominate the clamped-edge playfield where it rises.
ring = 0
for sx in (-1, 1):
    for sz in (-1, 1):
        hit = None
        for d in (550.0, 700.0, 850.0, 1000.0, 1150.0):
            for x, z in ((sx * (TERRAIN_SIZE / 2 + d), sz * (TERRAIN_SIZE / 2 + d * 0.3)),
                         (sx * (TERRAIN_SIZE / 2 + d * 0.3), sz * (TERRAIN_SIZE / 2 + d))):
                h = float(terrain.height_at(x, z))
                p = float(terrain._play_height_at(x, z))
                b = float(terrain.backdrop_height_at(x, z))
                assert h == max(p, b), (x, z, h, p, b)
                if b > p + 20.0:
                    assert h == b, (x, z, h, b, p)
                    hit = (x, z, b, p)
        assert hit is not None, f'ring quadrant ({sx},{sz}) has no mountain'
        ring += 1
        print(f'  ring ({hit[0]:+.0f},{hit[1]:+.0f}): backdrop {hit[2]:.1f} > '
              f'play {hit[3]:.1f} -> height_at {hit[2]:.1f}')
assert ring == 4, f'ring mountains must rise in all 4 quadrants (got {ring}/4)'
print('PASS ring: mountains rise past the walls, height_at == backdrop there')

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
# interior invariant: at every backdrop LATTICE vertex inside the square
# cutout (Chebyshev < _bd_cut) the backdrop sits at the deep floor
# (-TERRAIN_AMP - 25): the coarse 125m mesh can never poke through the play
# surface, so height_at() == playfield everywhere inside the arena.
_bc = np.maximum(np.abs(terrain._bx), np.abs(terrain._bz))
vi = _bc < terrain._bd_cut
assert np.allclose(terrain.backdrop_hm[vi],
                   -config.TERRAIN_AMP - 25.0, atol=1e-5)
# everywhere inside the walls height_at must EXACTLY equal play (backdrop
# loses the max() with ~40m of margin, so no bilinear bulge can reach it).
inside = np.maximum(np.abs(xs), np.abs(zs)) < TERRAIN_SIZE / 2 - 10
h_in = terrain.height_at(xs[inside], zs[inside])
p_in = terrain._play_height_at(xs[inside], zs[inside])
lift = h_in - p_in
assert np.allclose(lift, 0.0, atol=1e-6), (float(lift.min()), float(lift.max()))
print(f'PASS interior: lattice verts at deep floor (-{config.TERRAIN_AMP + 25}m), '
      f'height_at == playfield exactly inside the walls (max lift {lift.max():.2e}m)')

print('ALL TERRAIN-BACKDROP CHECKS PASSED')
