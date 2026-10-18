# Gate: evolution.log + rich history.csv telemetry.
import sys, os, tempfile
sys.path.insert(0, '/home/marek/Playground/bird_sim')
sys.argv = ['main.py', '--cpu']
import numpy as np
import config
TMP = tempfile.mkdtemp(prefix='logtest_')
config.EVOLVE_DIR = TMP
import predators, flock

assert predators._LOG_FILE == os.path.join(TMP, 'evolution.log')

# --- session line via load_checkpoint (no checkpoint file -> fresh, still logs)
predators.load_checkpoint()
with open(predators._LOG_FILE) as fh:
    lines = fh.read().splitlines()
assert len(lines) == 1 and '[session]' in lines[0] and 'gen 1' in lines[0]
assert lines[0][:19].replace('-', '').replace(':', '').replace(' ', '').isdigit(), lines[0]
print('PASS session line: timestamped, printed at load_checkpoint')

# --- generation summary -> log + rich csv ---
predators.population = np.random.default_rng(1).normal(0, 0.5, (10, predators.GENOME_LEN))
predators.fitness[:] = np.array([5, 9, 7, 1, 2, 3, 0, 4, 6, 8])
predators.gen_kills[:] = [1, 2, 0, 0, 0, 0, 0, 0, 0, 1]
predators.gen_max_speed = 6.33
predators.evolve_generation()
with open(predators._LOG_FILE) as fh:
    lines = fh.read().splitlines()
evo = [l for l in lines if '[evolution]' in l]
assert len(evo) == 1, evo
for token in ('gen 1:', 'best fitness 9 (hawk #1)', 'mean 4', 'kills best 2, total 4',
              'HoF CROWNED at 9', 'max_spd 6.33', 'flock_evos 0'):
    assert token in evo[0], f'missing {token!r} in: {evo[0]}'
print('PASS evolution line enriched:', evo[0][20:])

hdr = open(predators._HISTORY).readline().strip()
assert hdr == predators._HISTORY_HEADER, hdr
row = open(predators._HISTORY).read().splitlines()[1].split(',')
assert row[0] == '1' and row[3] == '2' and row[4] == '4' and row[5] == 'CROWNED'
assert row[6] == '9.0' and row[7] == '5.0' and row[8] == '1' and row[9] == '6.33' and row[10] == '0'
print('PASS history.csv new schema row:', row)

# --- HELD variant + second row appends ---
predators.fitness[:] = np.array([4, 1, 1, 1, 1, 1, 1, 1, 1, 1])   # slot0 King loses
predators.gen_kills[:] = 0
predators.gen_max_speed = 4.10
predators.evolve_generation()
rows = open(predators._HISTORY).read().splitlines()
assert len(rows) == 3 and rows[2].split(',')[5] == 'HELD'
logtxt = open(predators._LOG_FILE).read()
assert 'HoF holds at' in logtxt and 'slot0 live:' in logtxt
print('PASS HELD row + slot0 live in log')

# --- old-schema history.csv gets rewritten with the new header ---
with open(predators._HISTORY, 'w') as fh:
    fh.write('gen,best_fitness,mean_fitness,total_kills\n1,5,2,1\n')
predators.fitness[:] = 1.0
predators.evolve_generation()
lines = open(predators._HISTORY).read().splitlines()
assert lines[0] == predators._HISTORY_HEADER and len(lines) == 2, lines
print('PASS old-schema csv rewritten cleanly')

# --- tribe replacement routed through log_event ---
flock.bird_score[:] = np.arange(flock.NUM_BIRDS, dtype=np.float32)  # tribe 19 best
flock.tribe_casualties[0] = flock.TRIBE_REPLACE_DEATHS
caught = np.where(flock.bird_tribe_np == 0)[0][:flock.TRIBE_REPLACE_DEATHS]
flock._tribe_respawn_check(caught)
logtxt = open(predators._LOG_FILE).read()
assert '[flock_nn] tribe(s) [0] replaced' in logtxt
assert 'T0<-T' in logtxt and 'top tribe mean score' in logtxt
print('PASS flock_nn replacement logged with lineage + top score')

# --- PERSIST=False: prints, never touches the filesystem ---
predators.PERSIST = False
os.remove(predators._LOG_FILE)
predators.log_event('[persist-off] should print only')
assert not os.path.exists(predators._LOG_FILE)
print('PASS PERSIST=False: no filesystem write')
print('ALL LOGGING CHECKS PASSED')
