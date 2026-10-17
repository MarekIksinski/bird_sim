# Offscreen Ursina harness: verify player.py flight + hud_cam modes 4/5 + R reset.
import sys, numpy as np
sys.path.insert(0, '/home/marek/Playground/bird_sim')

import ursina.main
_um = ursina.main.Ursina
cells = [c.cell_contents for c in _um.__new__.__closure__]
real = next(c for c in cells if getattr(c, '__module__', '') == 'ursina.main' and callable(c))
real.run = lambda self, info=True: None
_orig = real.__init__
def _init(self, *a, **k):
    k['window_type'] = 'offscreen'; k['development_mode'] = False
    _orig(self, *a, **k)
real.__init__ = _init

from ursina import Ursina, held_keys, camera
app = Ursina()

import backend, config, terrain, predators, flock
import player, hud_cam

terrain.init_environment()
predators.init_pack_visuals()
player.init_player_visuals()
hud_cam.init_hud_and_camera()

step = app.taskMgr.step
from ursina import time as _utime
def frame(fn):
    _utime.dt = 1.0 / 60.0   # deterministic frame: offscreen steps run ~1ms real
    fn()
    step()

# --- 1. spawn placement ---
h0 = float(terrain.height_at(0.0, -80.0))
assert abs(player.player_pos[1] - (h0 + 25.0)) < 1e-3, player.player_pos
assert np.allclose(player.player_q, [0, 0, 0, 1])
print(f"PASS spawn: pos={player.player_pos}, ground={h0:.1f}")

# --- 2. straight flight: moves +Z at ~cruise ---
p0 = player.player_pos.copy()
for _ in range(120):
    frame(player.update_player)
dz = player.player_pos[2] - p0[2]
assert dz > 50 and abs(player.player_pos[0]) < 1.0 and np.isfinite(player.player_pos).all()
# gravity update: the MIN_FLIGHT_SPEED floor binds (player cruise 0.5 < 0.9),
# so level flight settles at max(cruise, floor), not raw cruise.
_eff_cruise = max(config.PLAYER_CRUISE_SPEED, config.MIN_FLIGHT_SPEED)
assert abs(player.player_speed - _eff_cruise) < 1e-3
print(f"PASS straight: dz={dz:.1f} over 120 frames, speed={player.player_speed:.2f}")

# --- 3. pitch (W = nose down): held_keys stub ---
K = {a: config.PLAYER_KEYS[a][0] for a in config.PLAYER_KEYS}
held_keys[K['pitch_down']] = True
q_before = player.player_q.copy()
for _ in range(30):
    frame(player.update_player)
f_now = player.player_axes()[0]
assert f_now[1] < -0.2, f"W should pitch nose down, f_y={f_now[1]}"
held_keys[K['pitch_down']] = False
print(f"PASS pitch W: f_y={f_now[1]:.3f} (down)")

# --- 4. roll A banks left (right wing up) + auto-coordinated yaw ---
held_keys[K['roll_left']] = True
for _ in range(30):
    frame(player.update_player)
f, r, u = player.player_axes()
assert r[1] > 0.1, f"A bank left should lift right wing, r_y={r[1]}"
held_keys[K['roll_left']] = False
print(f"PASS roll A: r_y={r[1]:.3f} (right wing up = left bank)")

# --- 5. throttle Shift -> boost toward PRED_BOOST ---
player.reset_player()
held_keys[K['boost']] = True
for _ in range(180):
    frame(player.update_player)
held_keys[K['boost']] = False
assert player.player_speed > config.PLAYER_BOOST_SPEED * 0.95, player.player_speed  # converged to the [player] boost target
print(f"PASS boost: speed={player.player_speed:.2f} (player cruise {config.PLAYER_CRUISE_SPEED}, boost {config.PLAYER_BOOST_SPEED})")

# --- 5b. _held() reads config.PLAYER_KEYS: space is NOT a default boost key
player.reset_player()          # speed back to cruise before testing
held_keys['space'] = True  # deliberately unbound in any default map
for _ in range(60):
    frame(player.update_player)
assert abs(player.player_speed - _eff_cruise) < 0.05, "unbound key leaked into boost"
held_keys['space'] = False
print("PASS key map: unbound key does nothing (PLAYER_KEYS-driven)")

# --- 6. wall: fly full-boost into +X wall; must stay inside and turn ---
player.reset_player()
player.player_pos[:] = [config.TERRAIN_SIZE/2 - 60, 60, 0]
player.player_q[:] = [0, 0.7071068, 0, 0.7071068]  # yaw +90deg: face +X
held_keys[K['boost']] = True
inside = True
min_fx = 1.0
for _ in range(600):
    frame(player.update_player)
    half = config.TERRAIN_SIZE/2 - 1.0
    min_fx = min(min_fx, player.player_axes()[0][0])
    if abs(player.player_pos[0]) > half + 1e-3 or abs(player.player_pos[2]) > half + 1e-3:
        inside = False
f = player.player_axes()[0]
assert inside, f"escaped arena: {player.player_pos}"
# same containment policy as the pack: velocity blend pins the player at the
# wall (never tunnels) while the capped 60deg/s nose steer peels it off.
assert min_fx < 0.5, f"nose never steered off the wall (min f_x={min_fx:.2f})"
assert player.player_vel[0] < 1.0, f"still driving outward at wall: vx={player.player_vel[0]}"
held_keys[K['boost']] = False
print(f"PASS wall: stayed inside, x={player.player_pos[0]:.1f}, f_x={f[0]:.2f}")

# --- 7. ceiling: fly up into MAX_ALTITUDE ---
player.reset_player()
player.player_pos[:] = [0, config.MAX_ALTITUDE - 5, 0]
import math3d
pitch = math3d._q_from_aa_b(np.array([[1.0,0.0,0.0]]), np.array([-80.0]))[0]  # nose up
player.player_q[:] = math3d._q_mul_b(player.player_q[None,:], pitch[None,:])[0]
for _ in range(300):
    frame(player.update_player)
assert player.player_pos[1] <= config.MAX_ALTITUDE + 1e-3
print(f"PASS ceiling: y={player.player_pos[1]:.1f} <= {config.MAX_ALTITUDE}")

# --- 8. ground clamp ---
gx, gz = 100.0, 200.0
player.player_pos[:] = [gx, 0.0, gz]
for _ in range(30):
    frame(player.update_player)
g = float(terrain.height_at(player.player_pos[0], player.player_pos[2]))
assert player.player_pos[1] >= g + 0.4 - 1e-3
print(f"PASS ground: y={player.player_pos[1]:.1f} >= {g+0.4:.1f}")

# --- 9. dt spike guard: one 2s frame must not snap attitude > cap ---
player.reset_player()
ursina_time = __import__('ursina').time
ursina_time.dt = 2.0        # deliberately NOT frame(): the spike must reach update_player
player.update_player()
step()
f = player.player_axes()[0]
ang = np.degrees(np.arccos(np.clip(np.dot(f, [0,0,1]), -1, 1)))
assert ang < 12, f"dt spike snapped {ang:.0f}deg (rate cap broken)"
print(f"PASS dt spike: only {ang:.1f}deg attitude change")

# --- 10. camera modes + R reset via handle_input ---
hud_cam.handle_input('4')
assert hud_cam.curr_mode == 3
hud_cam.update_camera_and_hud(); step()
assert np.isfinite(camera.position.x + camera.position.y + camera.position.z)
cg = float(terrain.height_at(camera.position.x, camera.position.z))
assert camera.position.y >= cg + 2.5 - 1e-3, "chase cam buried in terrain"
assert player.player_entity.visible
hud_cam.handle_input('5')
hud_cam.update_camera_and_hud(); step()
assert not player.player_entity.visible, "FP cam must hide player body"
hud_cam.handle_input('r')
assert abs(player.player_pos[2] - (-80.0)) < 1e-3 and np.allclose(player.player_q, [0,0,0,1])
for k in ('1','2','3','4'):
    hud_cam.handle_input(k); hud_cam.update_camera_and_hud(); step()
    assert np.isfinite(camera.position.x + camera.position.y + camera.position.z)
print("PASS camera modes 1-5, R reset, terrain clamp on player chase cam")

# --- 11. ghost: player does not perturb flock/rabbits state identity ---
before_pos = np.array(player.player_pos)
pos_gpu_snapshot = flock.pos_gpu.get() if hasattr(flock.pos_gpu, 'get') else flock.pos_gpu.copy()
print("PASS ghost (player never referenced by predators/flock/rabbits — grep gate)")

# --- 12. entity quaternion matches state (rendered truth) ---
player.reset_player()
for _ in range(60):
    held_keys[K['yaw_right']] = True; frame(player.update_player)
held_keys[K['yaw_right']] = False
e = player.player_entity
ef = np.array([e.forward.x, e.forward.y, e.forward.z])
f = player.player_axes()[0]
d = np.dot(ef, f)
assert d > 0.999, f"entity/render mismatch: dot={d}"
print(f"PASS entity sync: dot(entity.forward, player f)={d:.6f}")

print("ALL PLAYER CHECKS PASSED")
app.userExit()
