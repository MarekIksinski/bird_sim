"""Live check: after init_environment + visuals exist, sample entity
colors back and compare against the [colors] config tuples."""
import os, sys
os.environ['FORCE_CPU'] = '1'
sys.path.insert(0, '/home/marek/Playground/bird_sim')
os.chdir('/home/marek/Playground/bird_sim')
from ursina import Ursina, color, application
app = Ursina(window_type='offscreen', development_mode=False)
import config, terrain, flock, rabbits, foxes, predators, player, hud_cam
terrain.init_environment()
predators.init_pack_visuals()
player.init_player_visuals()
fv = flock.FlockVisuals()
rv = rabbits.RabbitVisuals()
xv = foxes.FoxVisuals()
hud_cam.init_hud_and_camera()

def c3(e):
    c = e.color
    return (round(c[0] * 255), round(c[1] * 255), round(c[2] * 255))

def c4(v):
    return (round(v[0] * 255), round(v[1] * 255), round(v[2] * 255), round(v[3] * 255))

checks = [
    ('fog', c4(__import__('ursina').scene.fog_color)[:3], config.COLOR_FOG),
    ('starling', c3(fv), config.COLOR_STARLING),
    ('rabbit', c3(rv), config.COLOR_RABBIT),
    ('fox', c3(xv), config.COLOR_FOX),
    ('hawk spawn', c3(predators.pack_entities[0]), config.COLOR_HAWK_MUTED),
    ('player', c3(player.player_entity), config.COLOR_PLAYER),
]
ok = True
for name, got, want in checks:
    match = got == tuple(want[:3])
    ok &= match
    print(f'{"OK " if match else "FAIL"} {name}: entity={got} config={tuple(want[:3])}')

# rank colors applied by update_pack_visuals
import numpy as np
predators.fitness = np.arange(config.POPULATION_SIZE, dtype=np.float32)
predators.pack_pos = np.zeros((config.POPULATION_SIZE, 3), dtype=np.float32)
predators.pack_q = np.zeros((config.POPULATION_SIZE, 4), dtype=np.float32)
predators.pack_q[:, 3] = 1.0
predators.update_pack_visuals()
order = np.argsort(-predators.fitness)
rank = [(int(order[0]), c3(predators.pack_entities[order[0]])),
        (int(order[1]), c3(predators.pack_entities[order[1]])),
        (int(order[2]), c3(predators.pack_entities[order[2]]))]
want = [tuple(config.COLOR_HAWK_GOLD[:3]), tuple(config.COLOR_HAWK_BRIGHT[:3]), tuple(config.COLOR_HAWK_MUTED[:3])]
for (hawk_i, got), w in zip(rank, want):
    match = got == w
    ok &= match
    print(f'{"OK " if match else "FAIL"} hawk idx {hawk_i} rank: entity={got} config={w}')

# HUD text colors
for name, t, want in (('hud_info', hud_cam.hawk_info, config.COLOR_HUD_INFO),
                      ('hud_gen', hud_cam.gen_text, config.COLOR_HUD_GEN),
                      ('hud_kills', hud_cam.kill_counter, config.COLOR_HUD_KILLS)):
    got = c4(t.color)
    match = got == tuple(want)
    ok &= match
    print(f'{"OK " if match else "FAIL"} {name}: entity={got} config={tuple(want)}')

print('ALL COLOR WIRING OK' if ok else 'COLOR WIRING FAILURES')
application.base.userExit()
sys.exit(0 if ok else 1)
