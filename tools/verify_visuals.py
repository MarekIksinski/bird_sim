# Offscreen-Ursina harness for the visuals task:
#  STEP1 terrain uses the patched triplanar shader (compiles clean, scales set)
#  STEP2 a shadow-casting DirectionalLight exists; entities carry lit shader
#  STEP3 sprint dust: emission above cruise*factor, fade + recycle, pool caps
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ursina.main
_um = ursina.main
_real = _um.Ursina.__new__.__closure__
_cls = None
for c in _real:
    try:
        v = c.cell_contents
        if getattr(v, '__module__', None) == 'ursina.main':
            _cls = v
    except Exception:
        pass
_cls.run = lambda self, info=True: None
_orig_init = _cls.__init__
def _patched(self, *a, **kw):
    kw['window_type'] = 'offscreen'; kw['development_mode'] = False
    _orig_init(self, *a, **kw)
_cls.__init__ = _patched

from ursina import Ursina, Entity, scene
from ursina.shaders import lit_with_shadows_shader
from ursina.lights import DirectionalLight
app = Ursina()

import config
Entity.default_shader = lit_with_shadows_shader
lit_with_shadows_shader.default_input['shadow_bias'] = config.SHADOW_BIAS
sun = DirectionalLight(shadow_map_resolution=(config.SHADOW_MAP_SIZE, config.SHADOW_MAP_SIZE),
                       shadows=True)
from ursina import Vec3
sun.look_at(Vec3(-1, -1.2, -1))

import terrain
# stub the huge mesh builders so the harness runs in seconds (verify_props pattern)
def _tiny_mesh():
    from ursina import Mesh
    m = Mesh(vertices=[(0,0,0),(1,0,0),(0,0,1)], triangles=[(0,1,2)], mode='triangle', static=True)
    m.generate(); return m
terrain.build_terrain_mesh = _tiny_mesh
terrain.build_backdrop_mesh = _tiny_mesh
terrain.init_environment()
app.taskMgr.step(); app.taskMgr.step()

# --- STEP1: triplanar terrain ---
t = terrain.terrain
assert t.shader is terrain.triplanar_terrain_shader, "terrain missing triplanar shader"
# the derived shader must NOT contain the broken .rgba line and MUST have fog+shadow
frag = terrain.triplanar_terrain_shader.fragment
assert 'triPlanar.rgba' not in frag, 'broken stock line still present'
assert 'cast_shadows' in frag and 'fog_start' in frag
# shader inputs from config
si = t._shader_inputs
assert abs(si['texture_scale'][0] - config.TRIPLANAR_TEXTURE_SCALE) < 1e-9 and \
       abs(si['texture_scale'][1] - config.TRIPLANAR_TEXTURE_SCALE) < 1e-9, si['texture_scale']
assert abs(si['side_texture_scale'][0] - config.TRIPLANAR_SIDE_TEXTURE_SCALE) < 1e-9 and \
       abs(si['side_texture_scale'][1] - config.TRIPLANAR_SIDE_TEXTURE_SCALE) < 1e-9
assert 'side_texture' in si
print('STEP1 triplanar OK: shader patched, scales', tuple(si['texture_scale']))

# --- STEP2: shadows + lit entities ---
assert sun._light.is_shadow_caster(), 'sun is not a shadow caster'
assert terrain.backdrop.shader is lit_with_shadows_shader, 'backdrop not lit_with_shadows'
assert terrain.backdrop._shader_inputs.get('shadow_bias') == config.SHADOW_BIAS
sun.update_bounds()
print('STEP2 shadow sun OK: caster on, bias', config.SHADOW_BIAS)

# --- STEP3: dust pools ---
# redirect checkpoints so nothing live is touched (not strictly needed here;
# no evolution runs — we drive foxes/wolves update directly)
import numpy as np
import foxes, wolves, rabbits, predators
# tempdir the evolution log/ckpt in case a starve check fires
import tempfile
_tmp = tempfile.mkdtemp()
predators._CKPT = os.path.join(_tmp, 'population.npz')
predators._HISTORY = os.path.join(_tmp, 'history.csv')
predators._LOG_FILE = os.path.join(_tmp, 'evolution.log')
predators._STARLING_CKPT = os.path.join(_tmp, 'starling_tribes.npz')
predators._FOX_CKPT = os.path.join(_tmp, 'fox_pairs.npz')
if hasattr(predators, '_WOLF_CKPT'):
    predators._WOLF_CKPT = os.path.join(_tmp, 'wolf_pairs.npz')

fox_viz = foxes.FoxVisuals()
wolf_viz = wolves.WolfVisuals()
app.taskMgr.step()
assert foxes._dust_pool is not None and wolves._dust_pool is not None
assert len(foxes._dust_pool.pips) == config.FOX_DUST_POOL
assert len(wolves._dust_pool.pips) == config.WOLF_DUST_POOL
assert all(not e.enabled for e in foxes._dust_pool.pips), 'pips should start disabled'

# pin the Clock (offscreen dt ~0 freezes the physics per skill notes)
class Clock:
    def __init__(self, dt): self.dt = dt; self.time = 0.0
for mod in (foxes, wolves, rabbits):
    mod.time = Clock(1/60)

# force all foxes to sprint: pin speed above the threshold each frame
foxes.fox_speed[:] = config.FOX_BOOST_SPEED
wolves.wolf_speed[:] = config.WOLF_BOOST_SPEED

# drive update_foxes/update_wolves directly (Visuals.update pulls the whole
# chain; call the physics fns and step the pool exactly as they do)
n_emitted = 0
def step_fox():
    global n_emitted
    before = sum(1 for e in foxes._dust_pool.pips if e.enabled)
    foxes.update_foxes()
    foxes.fox_speed[:] = config.FOX_BOOST_SPEED
    after = sum(1 for e in foxes._dust_pool.pips if e.enabled)
    return before, after

b, a = step_fox()
assert a > b, 'no dust emitted on frame 1 of sprint'
# steady state: at 60fps and dust_interval 0.12, ~5 emits/s/fox; pool 64,
# lifetime 0.4s -> ~20*0.4=8 live per 20 foxes*0.12... run 120 frames and
# assert the live count plateaus BELOW the pool size (recycling works)
live_hist = []
for f in range(120):
    b, a = step_fox()
    live_hist.append(a)
assert max(live_hist) < config.FOX_DUST_POOL, \
    f'live pips {max(live_hist)} hit the pool cap — recycling/fading broken'
print(f'STEP3a fox dust OK: live pips plateau {live_hist[-1]} (cap {config.FOX_DUST_POOL})')

# wolves
def step_wolf():
    before = sum(1 for e in wolves._dust_pool.pips if e.enabled)
    wolves.update_wolves()
    wolves.wolf_speed[:] = config.WOLF_BOOST_SPEED
    return sum(1 for e in wolves._dust_pool.pips if e.enabled)
w = step_wolf()
assert w > 0, 'no wolf dust'
for f in range(120):
    w = step_wolf()
assert w < config.WOLF_DUST_POOL
print(f'STEP3b wolf dust OK: live pips {w} (cap {config.WOLF_DUST_POOL})')

# slow animals emit nothing
foxes.fox_speed[:] = config.FOX_CRUISE_SPEED * 0.5
foxes._dust_pool.step(1.0)   # expire everything
for e in foxes._dust_pool.pips: e.enabled = False
for f in range(60):
    foxes.update_foxes()
    foxes.fox_speed[:] = config.FOX_CRUISE_SPEED * 0.5
live = sum(1 for e in foxes._dust_pool.pips if e.enabled)
assert live == 0, f'trotting foxes emitted {live} pips'
print('STEP3c no dust below threshold OK')

# emission geometry: pip sits ~0.5m BEHIND the fox (rear), on the ground
foxes.fox_speed[:] = config.FOX_BOOST_SPEED
foxes._dust_cooldown[:] = 0.0
foxes._dust_pool.cursor = 0
for e in foxes._dust_pool.pips: e.enabled = False
foxes.update_foxes()
foxes.fox_speed[:] = config.FOX_BOOST_SPEED
pip = foxes._dust_pool.pips[0]     # cursor reset: fox 0 emitted into slot 0
assert pip.enabled
d = np.array(pip.position) - foxes.fox_pos[0][[0,1,2]]
fwd = np.array([np.sin(foxes.fox_yaw[0]), np.cos(foxes.fox_yaw[0])])
behind = -np.dot(d[[0,2]], fwd)
assert 0.3 < behind < 0.7, f'pip not at rear: behind={behind}'
print(f'STEP3d rear geometry OK: pip {behind:.2f}m behind the fox')

print('VISUALS_ALL_OK')
application_exit = getattr(app, 'userExit', None)
if application_exit: app.userExit()
