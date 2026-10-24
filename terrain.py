# -*- coding: utf-8 -*-
"""
terrain.py — procedural heightmap + terrain/backdrop/sky/trees (layer 4).

The heightmap generation and height_at() are pure NumPy and run at IMPORT
time, so physics modules can import height_at without Ursina existing.
All Ursina entity creation (Sky, terrain, backdrop, trees) is deferred to
init_environment(), which main.py calls AFTER app = Ursina(vsync=False).
"""
import numpy as np
from ursina import Entity, Mesh, Sky, color, scene

from config import (TERRAIN_SIZE, TERRAIN_CELL, TERRAIN_RES, TERRAIN_AMP,
                    TERRAIN_SEED, ROOST_CLEAR_RADIUS, BACKDROP_SIZE,
                    BACKDROP_AMP, BACKDROP_RES,
                    COLOR_FOG, COLOR_TREE_TRUNK, COLOR_TREE_CROWN,
                    COLOR_BUSH, COLOR_ROCK)


def _smooth(t):
    return t * t * (3 - 2 * t)

def _make_heightmap(seed, res, size, amp):
    rng = np.random.default_rng(seed)
    uv = np.linspace(-1, 1, res + 1)
    xs, zs = np.meshgrid(uv * size / 2, uv * size / 2)
    h = np.zeros((res + 1, res + 1), dtype=np.float64)
    gain = 1.0; total = 0.0
    scale = size / 150.0   # keep hills the same physical size on bigger fields
    # Cap octaves so the noise lattice never far-exceeds the output grid:
    # finer octaves than the grid only alias, and the last lattice would be
    # gigabytes for the big backdrop (13k x 13k doubles).
    octaves = min(5 + int(np.log2(max(scale, 1))), int(np.log2(res)))
    for octave in range(octaves):
        cells = int((2 ** (octave + 1)) * scale) + 1   # lattice spanning the field
        g = rng.random((cells, cells))
        fx = (xs / size + 0.5) * (cells - 1)
        fz = (zs / size + 0.5) * (cells - 1)
        i = np.clip(np.floor(fx).astype(int), 0, cells - 2)
        j = np.clip(np.floor(fz).astype(int), 0, cells - 2)
        tx = _smooth(fx - i); tz = _smooth(fz - j)
        n = (g[i, j] * (1 - tx) * (1 - tz) + g[i + 1, j] * tx * (1 - tz) +
             g[i, j + 1] * (1 - tx) * tz + g[i + 1, j + 1] * tx * tz)
        h += n * gain
        total += gain; gain *= 0.5
    h = (h / total - 0.5) * 2 * amp
    # flatten the roost bowl so the flock keeps an open arena
    r = np.sqrt(xs**2 + zs**2)
    k = np.clip((r - ROOST_CLEAR_RADIUS) / 25.0, 0, 1)
    h *= _smooth(k)
    h = np.maximum(h, -1.0)  # valleys don't go under the waterline
    return h.astype(np.float32)

heightmap = _make_heightmap(TERRAIN_SEED, TERRAIN_RES, TERRAIN_SIZE, TERRAIN_AMP)
_cell = TERRAIN_SIZE / TERRAIN_RES

def _play_height_at(x, z):
    """Raw playfield heightmap bilinear sample (vectorized, used by physics too).
    Public callers should use height_at(), which layers the backdrop on top."""
    u = (np.asarray(x, dtype=np.float64) + TERRAIN_SIZE / 2) / _cell
    v = (np.asarray(z, dtype=np.float64) + TERRAIN_SIZE / 2) / _cell
    u = np.clip(u, 0, TERRAIN_RES - 1e-6); v = np.clip(v, 0, TERRAIN_RES - 1e-6)
    i = u.astype(int); j = v.astype(int)
    tu = u - i; tv = v - j
    hm = heightmap
    return (hm[j, i] * (1 - tu) * (1 - tv) + hm[j, i + 1] * tu * (1 - tv) +
            hm[j + 1, i] * (1 - tu) * tv + hm[j + 1, i + 1] * tu * tv)

def build_terrain_mesh():
    verts = np.zeros((TERRAIN_RES + 1, TERRAIN_RES + 1, 3), dtype=np.float32)
    lin = np.linspace(-TERRAIN_SIZE / 2, TERRAIN_SIZE / 2, TERRAIN_RES + 1)
    verts[:, :, 0] = lin[None, :]
    verts[:, :, 1] = heightmap
    verts[:, :, 2] = lin[:, None]
    tri = []
    for j in range(TERRAIN_RES):
        for i in range(TERRAIN_RES):
            a = j * (TERRAIN_RES + 1) + i
            b = a + 1
            c = a + (TERRAIN_RES + 1)
            d = c + 1
            tri += [a, c, b, b, c, d]
    # flat-shaded vertex colors from height (low-poly look)
    cols = np.zeros((TERRAIN_RES + 1, TERRAIN_RES + 1, 4), dtype=np.float32)
    h = heightmap
    grass = np.array([0.24, 0.48, 0.16]); high = np.array([0.45, 0.58, 0.28])
    t = np.clip((h + 1) / (TERRAIN_AMP + 1), 0, 1)[:, :, None]
    rgb = grass * (1 - t) + high * t
    # darken steep slopes for definition
    dh = np.abs(np.gradient(h, axis=0)) + np.abs(np.gradient(h, axis=1))
    shade = (1 - np.clip(dh / 1.2, 0, 0.45))[:, :, None]
    cols[:, :, :3] = rgb * shade
    cols[:, :, 3] = 1.0
    m = Mesh(vertices=verts.reshape(-1, 3).tolist(),
             triangles=tri,
             colors=cols.reshape(-1, 4).tolist(),
             mode='triangle', static=True)
    m.generate()
    m.generate_normals(smooth=False)
    return m

# --- 2b. BACKDROP MOUNTAINS (visual shell + physics ground in the corners) ---
# NOTE: the blend is RADIAL (r = sqrt(x^2+z^2) starts rising at TERRAIN_SIZE/2)
# while the playable arena is a SQUARE — so in the four arena corners the
# backdrop mountains rise ABOVE the playfield heightmap inside the walls.
# height_at() below therefore returns max(playfield, backdrop): the encroaching
# corner ridges are solid ground for physics, AI clearance sensors, rabbit
# ground clamping, tree placement, and camera collision, not just visuals.
backdrop_hm = _make_heightmap(TERRAIN_SEED + 1, BACKDROP_RES, BACKDROP_SIZE, BACKDROP_AMP)
# Feather the center instead of a hard basin: the backdrop continues the PLAY
# surface ~3m below it out to the map edge (r=600), then blends up into the
# mountain ring over the next 100m (just outside the invisible walls at ~599).
_blin = np.linspace(-BACKDROP_SIZE / 2, BACKDROP_SIZE / 2, BACKDROP_RES + 1)
_bx, _bz = np.meshgrid(_blin, _blin)
_br = np.sqrt(_bx**2 + _bz**2)
_play_floor = _play_height_at(_bx, _bz) - 3.0   # _play_height_at clamps coords: extends the edge
_blend = _smooth(np.clip((_br - TERRAIN_SIZE / 2) / 100.0, 0, 1))
backdrop_hm = _play_floor * (1 - _blend) + backdrop_hm * _blend

_bcell = BACKDROP_SIZE / BACKDROP_RES

def backdrop_height_at(x, z):
    """Bilinear sample of backdrop_hm (same vectorized pattern as
    _play_height_at, scalar and array inputs alike)."""
    u = (np.asarray(x, dtype=np.float64) + BACKDROP_SIZE / 2) / _bcell
    v = (np.asarray(z, dtype=np.float64) + BACKDROP_SIZE / 2) / _bcell
    u = np.clip(u, 0, BACKDROP_RES - 1e-6); v = np.clip(v, 0, BACKDROP_RES - 1e-6)
    i = u.astype(int); j = v.astype(int)
    tu = u - i; tv = v - j
    hm = backdrop_hm
    return (hm[j, i] * (1 - tu) * (1 - tv) + hm[j, i + 1] * tu * (1 - tv) +
            hm[j + 1, i] * (1 - tu) * tv + hm[j + 1, i + 1] * tu * tv)

def height_at(x, z):
    """Bilinear sample of the effective ground surface (max of playfield
    heightmap and encroaching backdrop mountains)."""
    return np.maximum(_play_height_at(x, z), backdrop_height_at(x, z))

def build_backdrop_mesh():
    verts = np.zeros((BACKDROP_RES + 1, BACKDROP_RES + 1, 3), dtype=np.float32)
    lin = _blin
    verts[:, :, 0] = lin[None, :]
    verts[:, :, 1] = backdrop_hm
    verts[:, :, 2] = lin[:, None]
    tri = []
    for j in range(BACKDROP_RES):
        for i in range(BACKDROP_RES):
            a = j * (BACKDROP_RES + 1) + i
            b = a + 1
            c = a + (BACKDROP_RES + 1)
            d = c + 1
            tri += [a, c, b, b, c, d]
    cols = np.zeros((BACKDROP_RES + 1, BACKDROP_RES + 1, 4), dtype=np.float32)
    h = backdrop_hm
    rock = np.array([0.12, 0.14, 0.18]); snow = np.array([0.45, 0.48, 0.55])
    t = np.clip((h + 50) / (BACKDROP_AMP + 50), 0, 1)[:, :, None]
    rgb = rock * (1 - t) + snow * t
    # fake atmospheric scattering: darker + bluer the higher/farther
    cols[:, :, :3] = rgb * 0.6
    cols[:, :, 3] = 1.0
    m = Mesh(vertices=verts.reshape(-1, 3).tolist(),
             triangles=tri,
             colors=cols.reshape(-1, 4).tolist(),
             mode='triangle', static=True)
    m.generate()
    m.generate_normals(smooth=False)
    return m

# module-level handles set by init_environment()
terrain = None
backdrop = None

def init_environment():
    """Instantiate the Ursina Sky, terrain, backdrop and tree entities.
    main.py calls this AFTER app = Ursina(vsync=False)."""
    global terrain, backdrop
    sky = Sky(texture='sky_sunset')  # noqa: F841  (kept alive by ursina's scene)
    scene.fog_color = color.rgba32(*COLOR_FOG)
    # NOTE: the fog_density setter only re-shaders entities that EXIST at
    # call time — main.py RE-applies it (config._FOG_RANGE) after all
    # entities are built, just before app.run().
    from config import _FOG_RANGE
    scene.fog_density = _FOG_RANGE

    terrain = Entity(model=build_terrain_mesh(), texture='white_cube')
    terrain.double_sided = True  # no backface culling; hills never vanish at grazing angles

    backdrop = Entity(model=build_backdrop_mesh(), texture='white_cube')
    backdrop.double_sided = True
    # much thinner fog than the playfield so the range reads as a hazy silhouette
    backdrop.set_shader_input('fog_start', 800)
    backdrop.set_shader_input('fog_end', 5000)

    # --- Procedural props: trees (trunk+crown), valley bushes, slope rocks ---
    # All props sit on height_at(x, z) = max(playfield, backdrop), so they stay
    # flush on the hills AND the encroaching corner/mountain skirts. Sampling is
    # vectorized rejection sampling against a fixed-seed rng -> deterministic
    # world, instant startup (380 cheap entities).
    rng = np.random.default_rng(TERRAIN_SEED + 2)
    margin = 12.0
    lo, hi = -TERRAIN_SIZE / 2 + margin, TERRAIN_SIZE / 2 - margin

    def _scatter(n, keep, attempts=30):
        """Rejection-sample n (x, z) points in the playfield box.
        keep(x, z, h) -> bool mask over vectorized candidates."""
        pts, need = [], n
        for _ in range(attempts):
            cand = rng.uniform(lo, hi, size=(int(need * 3) + 8, 2))
            h = height_at(cand[:, 0], cand[:, 1])
            sel = cand[keep(cand[:, 0], cand[:, 1], h)][:need]
            pts.append(sel)
            need -= len(sel)
            if need <= 0:
                break
        return np.vstack(pts) if pts else np.zeros((0, 2), dtype=np.float64)

    r_of = lambda x, z: np.sqrt(x * x + z * z)

    # 1. Trees — trunk + foliage crown, outside the roost clearing
    tree_pts = _scatter(80, lambda x, z, h: r_of(x, z) > ROOST_CLEAR_RADIUS + 10)
    for x, z in tree_pts:
        h = float(height_at(x, z))
        trunk_h = rng.uniform(4.0, 9.0)
        Entity(model='cube', color=color.rgba32(*COLOR_TREE_TRUNK),
               scale=(0.8, trunk_h, 0.8), position=(x, h + trunk_h / 2, z))
        cw, ch, cd = rng.uniform(3.0, 6.0, size=3)
        crown_model = 'sphere' if rng.random() < 0.6 else 'cube'
        crown = Entity(model=crown_model, color=color.rgba32(*COLOR_TREE_CROWN),
                       scale=(cw, ch, cd),
                       position=(x, h + trunk_h + ch * 0.30, z))
        crown.y_rotation = rng.uniform(0, 360)

    # 2. Bushes — low and wide in the valleys (low elevation, outside the bowl)
    bush_pts = _scatter(120, lambda x, z, h:
                        (h < TERRAIN_AMP * 0.4) & (r_of(x, z) > ROOST_CLEAR_RADIUS + 5))
    for x, z in bush_pts:
        h = float(height_at(x, z))
        bw, bd = rng.uniform(1.5, 3.0, size=2)
        Entity(model='cube', color=color.rgba32(*COLOR_BUSH),
               scale=(bw, rng.uniform(0.6, 1.1), bd),
               position=(x, h + 0.25, z), y_rotation=rng.uniform(0, 360))

    # 3. Rocks/boulders — steeper slopes (central-difference gradient) OR the
    #    backdrop mountain skirts (r > 35% of field size). Anchored by sinking
    #    the block ~40% of its height into the ground.
    def _slope(x, z):
        e = 4.0
        dhx = height_at(x + e, z) - height_at(x - e, z)
        dhz = height_at(x, z + e) - height_at(x, z - e)
        return (np.abs(dhx) + np.abs(dhz)) / (2 * e)

    rock_pts = _scatter(100, lambda x, z, h:
                        (_slope(x, z) > 0.35) | (r_of(x, z) > TERRAIN_SIZE * 0.35))
    for x, z in rock_pts:
        h = float(height_at(x, z))
        s = rng.uniform(1.5, 4.0)
        rock_model = 'cube' if rng.random() < 0.7 else 'sphere'
        rock = Entity(model=rock_model, color=color.rgba32(*COLOR_ROCK),
                      scale=(s * rng.uniform(0.8, 1.2), s, s * rng.uniform(0.8, 1.2)),
                      position=(x, h + s / 2 - s * 0.4, z))
        rock.y_rotation = rng.uniform(0, 360)
        rock.x_rotation = rng.uniform(-8, 8)  # slight tilt = less grid-aligned
    return sky, terrain, backdrop
