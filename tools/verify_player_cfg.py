# Verify [player] config: speeds/rates/keys flow into player.py + hud_cam.
# Runs config.py against TWO config.toml variants (default + remapped).
import sys, os, subprocess, tempfile, textwrap

proj = '/home/marek/Playground/bird_sim'

# --- variant A: the live config.toml (player section mirrors hawk) ---
code_a = textwrap.dedent('''
    import sys
    sys.path.insert(1, '/home/marek/Playground/bird_sim')
    import config
    assert config.PLAYER_CRUISE_SPEED == 2.5, config.PLAYER_CRUISE_SPEED
    assert config.PLAYER_BOOST_SPEED == 3.5
    assert config.PLAYER_YAW_RATE == 80.0 and config.PLAYER_PITCH_RATE == 60.0
    assert config.PLAYER_ROLL_RATE == 60.0
    assert config.PLAYER_AUTO_COORD == 0.35
    assert config.PLAYER_KEYS['pitch_down'] == ('w', 'up arrow')
    assert config.PLAYER_KEYS['yaw_left'] == ('q',)
    assert config.PLAYER_KEYS['reset'] == ('r',)
    print("PASS live config: player section parsed, values + keys correct")
''')

# --- variant B: remapped keys + different speeds, in an isolated dir ---
d = tempfile.mkdtemp(prefix='cfgtest')
open(os.path.join(d, 'config.toml'), 'w').write(textwrap.dedent('''
    [hawk]
    cruise_speed = 2.5
    boost_speed = 3.5
    yaw_rate = 80.0
    pitch_rate = 60.0
    roll_speed = 60.0
    [player]
    cruise_speed = 4.0
    boost_speed = 7.0
    yaw_rate = 120.0
    auto_coord = 0.0
    pitch_down_keys = "i"
    roll_left_keys = ["j", "left arrow"]
    boost_keys = ["space", "right shift"]
    reset_keys = ["backspace", "r"]
    # slow_factor / pitch_rate / roll_speed intentionally ABSENT -> must
    # fall back to the [hawk] values.
'''))
# config.py resolves config.toml next to ITSELF, so copy config.py (+backend
# dependency-free? backend probes CuPy... stub it) into the temp dir.
import shutil
shutil.copy(os.path.join(proj, 'config.py'), d)
open(os.path.join(d, 'backend.py'), 'w').write('FORCE_CPU=True\n')  # unused by config.py; safety
code_b = textwrap.dedent('''
    import sys
    sys.path.insert(0, '/tmp/cfgtest')
    import config
    assert config.PLAYER_CRUISE_SPEED == 4.0
    assert config.PLAYER_BOOST_SPEED == 7.0
    assert config.PLAYER_YAW_RATE == 120.0
    assert config.PLAYER_AUTO_COORD == 0.0
    assert config.PLAYER_SLOW_FACTOR == 0.45, "missing -> [hawk] fallback"
    assert config.PLAYER_PITCH_RATE == 60.0, "missing -> [hawk] fallback"
    assert config.PLAYER_ROLL_RATE == 60.0
    assert config.PLAYER_KEYS['pitch_down'] == ('i',)
    assert config.PLAYER_KEYS['roll_left'] == ('j', 'left arrow')
    assert config.PLAYER_KEYS['boost'] == ('space', 'right shift')
    assert config.PLAYER_KEYS['reset'] == ('backspace', 'r')
    print("PASS remapped config: string/list keys + [hawk] fallback correct")
''')

r = subprocess.run([sys.executable, '-c', code_a], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-500:]); assert r.returncode == 0
code_b = code_b.replace('/tmp/cfgtest', d)
r = subprocess.run([sys.executable, '-c', code_b], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-500:]); assert r.returncode == 0
print("CONFIG CHECKS PASSED")
