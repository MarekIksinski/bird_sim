# -*- coding: utf-8 -*-
"""verify_render_fast.py — offscreen gate for the render-fast-path changes:
 1. flat float32 NumPy vertices/colors upload through ursina Mesh.generate()
    (the .tolist() removal in flock/rabbits/foxes/wolves render())
 2. FlockVisuals/RabbitVisuals/FoxVisuals/WolfVisuals .render() run clean
    and the GeomVertexData actually changes between frames (buffer updated)
 3. terrain visual mesh is capped to TERRAIN_MESH_RES (verts count) while
    the physics heightmap stays at TERRAIN_RES
 4. sun.update_bounds() with the backdrop disabled yields a shadow lens
    fitted to the arena (~1000m film), not the 4km backdrop
Boots Ursina OFFSCREEN (no X needed). Run: .venv/bin/python -u tools/verify_render_fast.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['FORCE_CPU'] = '1'
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f'{"OK " if cond else "FAIL"} {name}')

import backend
from ursina import Ursina
app = Ursina(window_type='offscreen')
from ursina.lights import DirectionalLight
import config, terrain, predators, flock, rabbits, foxes, wolves, player

# ---------- 1+2. flat numpy upload through the real render() calls ----------
# seed believable state so render() has real arrays to push
flock_viz = flock.FlockVisuals()
rabbit_viz = rabbits.RabbitVisuals()
fox_viz = foxes.FoxVisuals()
wolf_viz = wolves.WolfVisuals()

def geom_vdata(mesh):
    gn = mesh.geomNode
    if gn.getNumGeoms() == 0:
        return None
    return gn.modifyGeom(0).modifyVertexData()

def read_vdata(vd):
    """float32 view of vertex array 0 (same memoryview path ursina's
    _set_array_data writes through)."""
    arr = vd.modifyArray(0)
    n = vd.getNumRows()
    buf = np.frombuffer(memoryview(arr).cast('B'), dtype=np.float32)
    return buf.reshape(n, buf.size // n)

flock_viz.render()
vd1 = geom_vdata(flock_viz.model)
check('flock mesh built after render()', vd1 is not None and vd1.getNumRows() == config.NUM_BIRDS * 3)

# move the flock and re-render: the vertex buffer must REFLECT the move
r0a = read_vdata(vd1)
flock.pos_gpu += np.float32(17.0)
flock.last_pos_np = None; flock.last_vel_np = None   # force fresh cache
flock.update_boids_gpu()
flock_viz.render()
vd2 = geom_vdata(flock_viz.model)
r0b = read_vdata(vd2)
check('flat-numpy upload moved vertices',
      np.abs(r0b[:, 0].mean() - r0a[:, 0].mean()) > 5.0)

rabbit_viz.render(); fox_viz.render(); wolf_viz.render()
check('rabbit/fox/wolf meshes built',
      geom_vdata(rabbit_viz.model).getNumRows() == config.NUM_RABBITS * 3
      and geom_vdata(fox_viz.model).getNumRows() == config.NUM_FOXES * 3
      and geom_vdata(wolf_viz.model).getNumRows() == config.NUM_WOLVES * 3)

# wolf per-vertex colors (flat numpy colors path): the vertex FORMAT must
# carry a color column alongside vertex
check('wolf mesh carries per-vertex colors',
      geom_vdata(wolf_viz.model).getFormat().getNumColumns() >= 2)

# ---------- 3. visual mesh cap, physics heightmap untouched ----------
tv = terrain.build_terrain_mesh()
nverts = len(tv.vertices)
check(f'terrain visual verts {(config.TERRAIN_MESH_RES+1)**2} (got {nverts})',
      nverts == (config.TERRAIN_MESH_RES + 1) ** 2)
check(f'heightmap stays TERRAIN_RES={config.TERRAIN_RES}',
      terrain.heightmap.shape == (config.TERRAIN_RES + 1, config.TERRAIN_RES + 1))
import hashlib
check('heightmap pin unchanged',
      hashlib.sha256(terrain.heightmap.tobytes()).hexdigest().startswith('654dae6f40d9b26f'))

# ---------- 3b. triangle winding: front faces must point UP ----------
# Panda culls CW-as-seen-from-front faces when double_sided=False. The
# 2026-11-02 regression ([a,c,b, b,c,d]) made front faces point DOWN ->
# half the ground culled -> transparent terrain + black holes. Every
# terrain/backdrop quad must wind CCW-up: cross(b-a, c-a).y > 0.
def _check_winding(tri_list, verts_flat, name):
    v = np.asarray(verts_flat, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(tri_list, dtype=np.int64).reshape(-1, 3)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    ny = ((b[:, 0] - a[:, 0]) * (c[:, 2] - a[:, 2]) -
          (b[:, 2] - a[:, 2]) * (c[:, 0] - a[:, 0]))
    return bool(np.all(ny > 0)), f'{name} triangles all wind CCW-up'

tb = terrain.build_terrain_mesh()
bb = terrain.build_backdrop_mesh()
_c, _n = _check_winding(tb.triangles, tb.vertices, 'terrain'); check(_n, _c)
_c, _n = _check_winding(bb.triangles, bb.vertices, 'backdrop'); check(_n, _c)

# ---------- 4. shadow lens fitted to the arena, not the backdrop ----------
terrain.init_environment()   # builds terrain/backdrop/props entities
check('backdrop exists after init_environment', terrain.backdrop is not None)
sun = DirectionalLight(shadow_map_resolution=(256, 256))
sun.look_at((0, -1, 0))
terrain.backdrop.enabled = False
sun.update_bounds()
terrain.backdrop.enabled = True
lens = sun._light.getLens()
fs = lens.getFilmSize()
film = max(abs(fs[0]), abs(fs[1]))
check(f'shadow film ~ arena scale ({film:.0f}m, backdrop 4000m excluded)',
      film < 2200.0)

print('ALL RENDER-FAST CHECKS PASSED' if ok else 'RENDER-FAST CHECKS FAILED')
sys.exit(0 if ok else 1)
