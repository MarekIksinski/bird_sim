# -*- coding: utf-8 -*-
"""
terrain.py — procedural heightmap + terrain/backdrop/sky/trees (layer 4).

The heightmap generation and height_at() are pure NumPy and run at IMPORT
time, so physics modules can import height_at without Ursina existing.
All Ursina entity creation (Sky, terrain, backdrop, trees) is deferred to
init_environment(), which main.py calls AFTER app = Ursina(vsync=False).
"""
import numpy as np
from ursina import Entity, Mesh, Sky, color, load_texture, scene, Shader, Vec2, Vec3

from ursina.shaders import triplanar_shader

from config import (TERRAIN_SIZE, TERRAIN_CELL, TERRAIN_RES, TERRAIN_AMP,
                    TERRAIN_SEED, ROOST_CLEAR_RADIUS, BACKDROP_SIZE,
                    BACKDROP_AMP, BACKDROP_RES,
                    COLOR_FOG, COLOR_TREE_TRUNK, COLOR_TREE_CROWN,
                    COLOR_BUSH, COLOR_ROCK,
                    TRIPLANAR_TOP_TEXTURE, TRIPLANAR_SIDE_TEXTURE,
                    TRIPLANAR_TEXTURE_SCALE, TRIPLANAR_SIDE_TEXTURE_SCALE)


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
# The backdrop grid is deliberately coarse (BACKDROP_RES=32 over 4000m =
# 125m triangles), so ANY backdrop vertex above the play heightmap inside or
# adjacent to the playable square pokes dark grey polygons through the green
# terrain. Fix: a SQUARE cutout (Chebyshev distance, matching the square
# arena — a radial mask leaves the four corners exposed) that pins the
# backdrop at a deep floor inside TERRAIN_SIZE/2 + 150m, then smoothstep-
# ramps up to the full BACKDROP_AMP ring over the next 350m (150..500m
# beyond the playable edge). height_at() below still returns
# max(playfield, backdrop): inside the arena backdrop << playfield min (-1m
# waterline), so the play surface wins everywhere; outside, the ramped-up
# corner ridges stay solid ground for physics, AI clearance sensors, rabbit
# ground clamping, tree placement, and camera collision.
backdrop_hm = _make_heightmap(TERRAIN_SEED + 1, BACKDROP_RES, BACKDROP_SIZE, BACKDROP_AMP)
_bd_cut = TERRAIN_SIZE / 2 + 150.0    # deep floor out to 150m past the walls
_bd_ramp = 350.0                      # full mountain height by +500m past
_bd_floor = -TERRAIN_AMP - 25.0       # below the lowest play elevation
_blin = np.linspace(-BACKDROP_SIZE / 2, BACKDROP_SIZE / 2, BACKDROP_RES + 1)
_bx, _bz = np.meshgrid(_blin, _blin)
_bc = np.maximum(np.abs(_bx), np.abs(_bz))          # square (Chebyshev) distance
_blend = _smooth(np.clip((_bc - _bd_cut) / _bd_ramp, 0, 1))
backdrop_hm = _bd_floor * (1 - _blend) + backdrop_hm * _blend

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

# --- TRIPLANAR TERRAIN SHADER ------------------------------------------------
# Derived from ursina's built-in triplanar_shader (imported above) with three
# fixes, verified against a live GL context:
#  1. its last fragment line is `fragColor = vec4(triPlanar.rgba)` —
#     triPlanar is a vec3 and .rgba does not exist, so panda's Cg compiler
#     rejects the stock shader outright (created-shader error C1031);
#  2. it has no fog uniforms, so a triplanar terrain would not fog with the
#     rest of the scene (main.py's fog re-application only touches entities
#     whose shader declares fog_start/fog_end);
#  3. it is unlit, so a triplanar terrain would not receive the sun's
#     dynamic shadows. The Lambert + shadow-sampling block from ursina's
#     lit_with_shadows_shader is spliced in, so the terrain gets triplanar
#     blending AND lit shadows in one pass.
# The blend itself is the stock one: top texture on upward faces, side
# texture on slopes, weights from the world normal — no UVs, so hills and
# cliffs never get texture distortion.
_LIGHT_STRUCT = '''uniform struct {
    vec4 position;
    vec3 color;
    vec3 attenuation;
    vec3 spotDirection;
    float spotCosCutoff;
    float spotExponent;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[1];
'''

_triplanar_vert = triplanar_shader.vertex.replace(
    'uniform mat4 p3d_ModelMatrix;',
    'uniform mat4 p3d_ModelMatrix;\n'
    'uniform mat4 p3d_ModelViewMatrix;\n'
    'uniform mat3 p3d_NormalMatrix;\n'
    + _LIGHT_STRUCT +
    'out vec3 vertex_position;\n'
    'out vec4 shadow_coord[1];')
_triplanar_vert = _triplanar_vert.replace(
    '  vertex_color = p3d_Color;',
    '  vertex_color = p3d_Color;\n'
    '  vertex_position = vec3(p3d_ModelViewMatrix * p3d_Vertex);\n'
    '  shadow_coord[0] = p3d_LightSource[0].shadowViewMatrix * vec4(vertex_position, 1);')

_triplanar_frag = triplanar_shader.fragment.replace(
    'uniform vec2 side_texture_scale;',
    'uniform vec2 side_texture_scale;\n'
    + _LIGHT_STRUCT +
    'in vec4 shadow_coord[1];\n'
    'in vec3 vertex_position;\n'
    'uniform float shadow_bias;\n'
    'uniform vec4 shadow_color;\n'
    'uniform float shadow_blur;\n'
    'uniform int shadow_samples;\n'
    'uniform vec4 fog_color;\n'
    'uniform float fog_start;\n'
    'uniform float fog_end;\n'
    'uniform vec3 camera_world_position;\n'
    '''
const float M_PI = 3.141592653589793;
vec4 cast_shadows(vec4 color, vec3 N) {
    float shadow_samples_float = float(shadow_samples);
    float half_shadow_blur = shadow_blur / 2.0;
    for (int i = 0; i < p3d_LightSource.length(); ++i) {
        vec3 diff = p3d_LightSource[i].position.xyz - vertex_position * p3d_LightSource[i].position.w;
        vec3 L = normalize(diff);
        float NdotL = clamp(dot(N, L), 0.001, 1.0);
        vec3 light_contribution = NdotL * p3d_LightSource[i].color / M_PI;
        vec4 sc = shadow_coord[i];
        sc.z += shadow_bias;
        vec3 converted_shadow_color = (vec3(1.0) - shadow_color.rgb) * shadow_color.a;
        color.rgb *= p3d_LightSource[i].color.rgb;
        float shadow = 0.0;
        for (int x = 0; x < shadow_samples; ++x) {
            for (int y = 0; y < shadow_samples; ++y) {
                vec4 coord = sc;
                coord.x += float(x) * shadow_blur / shadow_samples_float - half_shadow_blur;
                coord.y += float(y) * shadow_blur / shadow_samples_float - half_shadow_blur;
                shadow += textureProj(p3d_LightSource[i].shadowMap, coord);
            }
        }
        shadow /= (shadow_samples_float * shadow_samples_float);
        color.rgb += shadow * converted_shadow_color;
        color.rgb += light_contribution - converted_shadow_color;
    }
    return color;
}
''')
_triplanar_frag = _triplanar_frag.replace(
    'fragColor = vec4(triPlanar.rgba) * vertex_color;',
    'vec4 _c = vec4(triPlanar, 1.0) * vertex_color;\n'
    '    _c = cast_shadows(_c, normalize(world_normal));\n'
    '    float _d = length(vertex_world_position - camera_world_position);\n'
    '    float _t = clamp((_d - fog_start) / max(fog_end - fog_start, 1e-6), 0.0, 1.0);\n'
    '    fragColor = mix(_c, fog_color, _t * fog_color.a);')
assert _triplanar_vert != triplanar_shader.vertex and \
       _triplanar_frag != triplanar_shader.fragment and \
       'cast_shadows(_c' in _triplanar_frag, \
    'ursina triplanar_shader source changed: the lighting/fog/vec4 patches no longer apply'

triplanar_terrain_shader = Shader(
    name='triplanar_terrain', language=Shader.GLSL,
    vertex=_triplanar_vert, fragment=_triplanar_frag,
    default_input={
        'texture_scale': Vec2(1, 1),
        'side_texture': 'brick',
        'side_texture_scale': Vec2(1, 1),
        'shadow_color': color.rgba(0, .5, 1, .25),
        'shadow_bias': 0.001,
        'shadow_blur': .005,
        'shadow_samples': 4,
        'fog_color': color.clear,
        'fog_start': 10,
        'fog_end': 100,
        'camera_world_position': Vec3.zero,
    })

def _camera_world_position():
    from ursina import camera
    return camera.world_position
triplanar_terrain_shader.continuous_input['camera_world_position'] = _camera_world_position


def _load_tex_wrap(name, fallback='white_cube'):
    """Load a texture for triplanar sampling and set repeat wrapping (the
    shader samples WORLD position * scale, so coordinates far exceed 0..1
    and clamp-to-edge would smear the outermost texel across the field).
    Falls back safely (missing texture -> white_cube -> None)."""
    try:
        t = load_texture(name)
    except Exception:
        print(f"[terrain] texture '{name}' unavailable; using '{fallback}'")
        try:
            t = load_texture(fallback)
        except Exception:
            return None
    try:
        from panda3d.core import Texture as PandaTexture
        t._texture.setWrapMode(PandaTexture.WM_repeat)
    except Exception:
        pass
    return t

def init_environment():
    """Instantiate the Ursina Sky, triplanar terrain, backdrop and tree
    entities. main.py calls this AFTER app = Ursina(vsync=False)."""
    global terrain, backdrop
    sky = Sky(texture='sky_sunset')  # noqa: F841  (kept alive by ursina's scene)
    scene.fog_color = color.rgba32(*COLOR_FOG)
    # NOTE: the fog_density setter only re-shaders entities that EXIST at
    # call time — main.py RE-applies it (config._FOG_RANGE) after all
    # entities are built, just before app.run().
    from config import _FOG_RANGE
    scene.fog_density = _FOG_RANGE

    # --- Triplanar terrain: grass on flat ground, rock on slopes/cliffs,
    # blended by world normal (no UV distortion on hills). Scales are
    # world-space (1/m): 0.05 -> one texture tile per 20m. ---
    top_tex = _load_tex_wrap(TRIPLANAR_TOP_TEXTURE)
    side_tex = _load_tex_wrap(TRIPLANAR_SIDE_TEXTURE)
    terrain = Entity(model=build_terrain_mesh(), texture=top_tex,
                     shader=triplanar_terrain_shader)
    terrain.double_sided = True  # no backface culling; hills never vanish at grazing angles
    terrain.set_shader_input('texture_scale',
                             Vec2(TRIPLANAR_TEXTURE_SCALE, TRIPLANAR_TEXTURE_SCALE))
    terrain.set_shader_input('side_texture_scale',
                             Vec2(TRIPLANAR_SIDE_TEXTURE_SCALE, TRIPLANAR_SIDE_TEXTURE_SCALE))
    if side_tex is not None:
        terrain.set_shader_input('side_texture', side_tex)

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
