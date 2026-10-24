# -*- coding: utf-8 -*-
"""verify_backdrop.py — the coarse 125m backdrop mesh must never poke above
the playable terrain. Pure heightmap checks (no window needed):
  1. every backdrop grid vertex touching the playable square is at the deep
     floor (<= -TERRAIN_AMP - 24)
  2. dense bilinear samples inside the square: height_at == playfield
     (backdrop loses the max() everywhere -> no grey polygons through green)
  3. the mountain ring still exists outside the ramp (physics ground in the
     corners intact)
  4. the cutout derives from config TERRAIN_SIZE (change it -> cutout moves)
Run: .venv/bin/python -u tools/verify_backdrop.py
"""
import os, sys
sys.path.insert(0, '/home/marek/Playground/bird_sim')
os.chdir('/home/marek/Playground/bird_sim')
import numpy as np
import config, terrain

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f'{"OK " if cond else "FAIL"} {name}')

half = config.TERRAIN_SIZE / 2
cell = config.BACKDROP_SIZE / config.BACKDROP_RES
print(f'arena half={half}m  backdrop cell={cell}m  AMP={config.TERRAIN_AMP} '
      f'BACKDROP_AMP={config.BACKDROP_AMP}')

# 1. backdrop vertices that could touch a triangle overlapping the arena
lin = np.linspace(-config.BACKDROP_SIZE / 2, config.BACKDROP_SIZE / 2,
                  config.BACKDROP_RES + 1)
bx, bz = np.meshgrid(lin, lin)
touch = (np.abs(bx) <= half + cell) & (np.abs(bz) <= half + cell)
floor_max = terrain.backdrop_hm[touch].max()
check(f'all arena-touching backdrop verts at deep floor (max {floor_max:.1f} <= -{config.TERRAIN_AMP + 24})',
      floor_max <= -config.TERRAIN_AMP - 24)

# 2. dense sampling inside the playable square: backdrop must lose the max()
g = np.linspace(-half, half, 700)
X, Z = np.meshgrid(g, g)
bd = terrain.backdrop_height_at(X, Z)
play = terrain._play_height_at(X, Z)
eff = terrain.height_at(X, Z)
worst = float(np.max(bd - play))
check(f'backdrop below playfield everywhere inside arena (max diff {worst:+.1f}m)',
      worst < 0.0)
check('height_at == playfield inside arena',
      np.allclose(eff, play, atol=1e-6))

# 3. ring intact: outside the ramp (Chebyshev > half+500) backdrop == raw
#    noise (ramp fully applied), and the raw ring still has real mountains
raw = terrain._make_heightmap(config.TERRAIN_SEED + 1, config.BACKDROP_RES,
                              config.BACKDROP_SIZE, config.BACKDROP_AMP)
out = (np.maximum(np.abs(bx), np.abs(bz)) > half + 500)
ring_max = raw[out].max()
check(f'raw ring has mountains (max {ring_max:.0f}m > 100)', ring_max > 100.0)
check('backdrop == raw noise beyond the ramp (blend saturated at edge+500)',
      np.allclose(terrain.backdrop_hm[out], raw[out], atol=1e-3))

# 4. ramp profile along +X: monotone rise from floor to raw by edge+500
xs = np.linspace(half - 100, half + 700, 400)
prof = terrain.backdrop_height_at(xs, np.zeros_like(xs))
d = np.diff(prof)
check('ramp never dips sharply (min step > -1m over ~2m sample)',
      d.min() > -1.0)
in_buf = xs <= terrain._bd_cut          # deep-floor zone ends at edge+150
check('inside buffer the profile sits at the deep floor',
      prof[in_buf].max() <= -config.TERRAIN_AMP - 24 + 1e-3)

# 5. the cutout derives from config TERRAIN_SIZE, not a hardcoded radius
check('cutout derived from config.TERRAIN_SIZE',
      abs(terrain._bd_cut - (config.TERRAIN_SIZE / 2 + 150.0)) < 1e-9)

print('ALL BACKDROP CHECKS PASSED' if ok else 'BACKDROP CHECKS FAILED')
sys.exit(0 if ok else 1)
