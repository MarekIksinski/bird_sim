#!/usr/bin/env python3
"""verify_hof.py — King of the Hill speciation gate (no Ursina window needed).

Imports the real predators module (FORCE_CPU shim), drives evolve_generation
with synthetic fitness vectors, and checks:
 1. first crowning + Tier1 == King clone, Tier2/3 == order[1]/order[2]
 2. King retention: EMA decay of hof_fitness, challenger promotion past slot 0
 3. fluke dethroning after enough bad live generations
 4. per-tier mutation sigmas (child-parent ||delta|| ~ sigma*sqrt(GENOME_LEN))
 5. checkpoint round-trip with hof keys + backward-compat with old npz
"""
import os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ['FORCE_CPU'] = '1'

import numpy as np
import config
TMP = tempfile.mkdtemp(prefix='birdsim_hof_')
config.EVOLVE_DIR = TMP
import predators   # noqa: E402

N, GL = predators.POPULATION_SIZE, predators.GENOME_LEN
assert N == 10

def base_pop(seed=42):
    return np.random.default_rng(seed).normal(0, 0.5, size=(N, GL))

def reset(pop):
    predators.population = pop.copy()
    predators.fitness[:] = 0.0
    predators.gen_kills[:] = 0
    predators.generation = 1
    predators.hof_genome = None
    predators.hof_fitness = -1e9

# --- 1. first crowning -------------------------------------------------------
pop = base_pop()
reset(pop)
predators.fitness[:] = np.array([5, 9, 7, 1, 2, 3, 0, 4, 6, 8])   # winner = slot 1
predators.evolve_generation()
assert predators.hof_genome is not None
assert np.array_equal(predators.hof_genome, pop[1]), "King must be order[0]=slot1 genome"
assert predators.hof_fitness == 9.0
assert np.array_equal(predators.population[0], pop[1]), "slot0 must be untouched King clone"
for i in (1, 2, 3):
    d = predators.population[i] - pop[1]
    assert abs(d.std() - config.MUTATION_TINY) < 0.01, f"tier1 sigma slot{i}"
# King crowned from order[0]=slot1 -> challengers skip 1: order=[1,9,7,...]
# order by fitness desc: 1(9),9(8),7(6)... wait check exact ranking:
order = np.argsort(-np.array([5, 9, 7, 1, 2, 3, 0, 4, 6, 8]))
assert list(order[:4]) == [1, 9, 2, 8]
assert np.array_equal(predators.population[4], pop[9]), "tier2 parent must skip crowned slot 1"
# LINEAGE DIVERSITY rule: tier parents come from OUTSIDE Tier 1 (slots 4-9)
# when they scored positive fitness -> slot 8 (fitness 6), NOT slot 2 (Tier-1 clone).
assert np.array_equal(predators.population[7], pop[8]), "tier3 parent must come from slots 4-9 (diversity rule)"
print("T1 OK: crowning + non-duplicate tier parents (skipped crowned index)")

# --- 1b. lineage diversity + immigrant slot + weight hygiene ---------------
pop1b = base_pop(43)
reset(pop1b)
# Tier-1 clones (slots 1-3) DOMINATE the ranking; positive outsiders exist at 4-9.
predators.fitness[:] = np.array([1, 12, 11, 10, 5, 4, 0, 0, 3, 2])
predators.gen_kills[:] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]   # nonzero kills -> crossover path
predators.evolve_generation()
# crowned slot 1; tier parents must be slots 4 and 5 (outside Tier 1), NOT 12/11/10 clones
assert np.array_equal(predators.population[4], pop1b[4]), "tier2 must prefer outside-Tier1 positive performer"
assert np.array_equal(predators.population[7], pop1b[5]), "tier3 must prefer outside-Tier1 positive performer"
# slot 9 immigrant: crossover of King x tier2 parent + heavy noise (not a tier3 clone)
assert not np.array_equal(predators.population[9], pop1b[5]), "slot9 must not be a tier3 offspring"
d9 = predators.population[9] - (0.5 * pop1b[1] + 0.5 * pop1b[4])
assert abs(d9.std() - config.MUTATION_HEAVY) < 0.05, "slot9 = 50/50 crossover + heavy mutation"
# weight hygiene: every mutated slot inside the clip band
assert np.abs(predators.population).max() <= predators.WEIGHT_CLIP + 1e-9
# zero-kill generation -> slot 9 is a fully fresh random genome
pop1c = base_pop(44)
reset(pop1c)   # reset zeroes gen_kills
predators.fitness[:] = np.array([3, 2, 1, 0, 5, 4, 0, 0, 0, 0])
predators.evolve_generation()
d9c = predators.population[9]
assert 0.3 < d9c.std() < 0.7, f"zero-kill gen: slot9 must be fresh N(0,0.5), std={d9c.std():.3f}"
# restore the T1 King state so the retention tests below start clean
predators.hof_genome = pop[1].copy()
predators.hof_fitness = 9.0
print("T1b OK: diversity parents, immigrant slot (crossover + fresh fallback), weight clip on mutation")

# --- 2. retention + EMA + challenger promotion -------------------------------
# King (in slot 0 now) flies badly (fitness 1); best challenger scores 5.
prev_hof = predators.hof_fitness   # 9.0
prev_king = predators.hof_genome.copy()
pop6 = base_pop(6); pop6[0] = prev_king
predators.population = pop6
predators.fitness[:] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 5])
fit6 = predators.fitness.copy()
predators.evolve_generation()
expect_ema = config.HOF_DECAY_KEEP * prev_hof + (1 - config.HOF_DECAY_KEEP) * 1.0  # 6.6
assert predators.hof_fitness == expect_ema, "EMA blend wrong (retain case)"
assert np.array_equal(predators.hof_genome, prev_king), "King must RETAIN crown (5 < 6.6)"
# retain path: king_slot = 0 -> challengers promoted past slot 0.
# (order ties among the zero-fitness slots are unstable in argsort, so
# derive the expected parents from the same order array the code uses.)
order6 = np.argsort(-fit6)
assert np.array_equal(predators.population[0], prev_king), "slot0 keeps the King"
exp2 = [int(i) for i in order6 if int(i) != 0][:2]
assert np.array_equal(predators.population[4], pop6[exp2[0]]), f"tier2 parent = challenger {exp2[0]}"
assert np.array_equal(predators.population[7], pop6[exp2[1]]), f"tier3 parent = challenger {exp2[1]}"
print("T2 OK: EMA blend exact, King retains, challengers promoted past slot 0")

# now a strong challenger (8) crosses the decayed rating -> dethroned (>= rule)
predators.hof_fitness = prev_hof  # rewind to 9 for a clean second pass
pop7 = base_pop(7)
predators.population = pop7
predators.fitness[:] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 8])
predators.evolve_generation()
# ema = .7*9 + .3*1 = 6.6 ; best 8 >= 6.6 -> crowned challenger
assert predators.hof_fitness == 8.0, "crowned challenger sets hof_fitness to its live score"
assert np.array_equal(predators.hof_genome, pop7[9]), "new King must be order[0]=slot9"
# king_slot = order[0] = 9 -> challengers: order[1]=0 (fitness 1), then a zero
assert np.array_equal(predators.population[0], pop7[9]), "slot0 = untouched new King"
assert np.array_equal(predators.population[4], pop7[0]), "tier2 parent = order[1] challenger"
assert not np.array_equal(predators.population[4], pop7[9]), "tier2 must not clone the King"
print("T3 OK: EMA exact, challenger dethrones decayed King, tiers skip it")

# pure RETAIN: king flies well, nobody beats blended rating
king = predators.hof_genome.copy()
hof_before = predators.hof_fitness
pop2 = base_pop(11); pop2[0] = king
predators.population = pop2
predators.fitness[:] = np.array([200, 1, 1, 1, 1, 1, 1, 1, 1, 1])
predators.evolve_generation()
# ema = .7*hof_before + .3*200 ; 200 >= that? likely yes -> crowned again.
# To force retain we need best < ema: king itself IS slot 0 and best... so
# with the >= rule a strong king is re-crowned (hof_fitness := 200). Check:
assert predators.hof_fitness == 200.0 and np.array_equal(predators.hof_genome, king)
assert predators.population[0] is not king and np.array_equal(predators.population[0], king)
# king_slot = order[0] = 0 -> challengers = order[1], order[2] (any two of 1..9)
assert not np.array_equal(predators.population[4], king), "tier2 must not clone the King"
print("T3 OK: strong King re-crowned, tiers never duplicate the King genome")

# --- 4. fluke decay: mediocre king, steady mediocre live score ---------------
predators.hof_fitness = 1000.0   # lucky fluke crown
predators.hof_genome = base_pop(3)[0]
fluke_king = predators.hof_genome.copy()
flew = []
for g in range(24):
    pop3 = base_pop(100 + g); pop3[0] = predators.hof_genome
    predators.population = pop3
    predators.fitness[:] = np.array([10] + [9] * 9)   # king lives at 10, challengers 9
    predators.evolve_generation()
    flew.append(predators.hof_fitness)
# EMA decays monotonically from the fluke 1000 toward the live ~10; the
# challenger field at 9 never crosses it, so the King holds throughout.
assert all(flew[i] >= flew[i + 1] for i in range(len(flew) - 1)), "EMA must decay monotonically"
assert flew[0] > flew[8], "rating must visibly decay over generations"
assert abs(flew[-1] - 10.0) < 1.5, f"rating must converge to live ~10, got {flew[-1]}"
assert np.array_equal(predators.hof_genome, fluke_king), "king genome unchanged"
print("T4 OK: fluke King rating decays to live performance and holds against weaker field")

# dethrone when field exceeds converged rating:
pop4 = base_pop(200); pop4[0] = predators.hof_genome
predators.population = pop4
predators.fitness[:] = np.array([10] + [50] + [9] * 8)
predators.evolve_generation()
assert predators.hof_fitness == 50.0, "field best must dethrone converged fluke"
print("T5 OK: real challenger dethrones the decayed fluke")

# --- 5. checkpoint round-trip + backward compat ------------------------------
predators.save_checkpoint()
d = np.load(os.path.join(TMP, 'population.npz'))
assert 'hof_genome' in d.files and d['hof_genome'].shape == (GL,)
assert np.array_equal(d['hof_genome'], predators.hof_genome)
assert float(d['hof_fitness']) == predators.hof_fitness
# old-format checkpoint (no hof keys)
np.savez(os.path.join(TMP, 'population.npz'),
         population=base_pop(5), generation=42, total_catches=7)
predators.hof_genome = None; predators.hof_fitness = -1e9
predators.load_checkpoint()
assert predators.generation == 42
assert predators.hof_genome is not None and predators.hof_fitness == 0.0
assert np.array_equal(predators.hof_genome, predators.population[0]), \
    "old ckpt: slot0 must become incumbent King"
print("T6 OK: checkpoint round-trip + backward-compatible HoF init")

# saturated checkpoint recovery: a checkpoint with |W| far outside the
# clip band must load BACK inside it (resumed runs recover sensitivity)
np.savez(os.path.join(TMP, 'population.npz'),
         population=base_pop(45) * 10.0, generation=7, total_catches=0,
         hof_genome=base_pop(46)[0] * 10.0, hof_fitness=1.0)
predators.hof_genome = None; predators.hof_fitness = -1e9
predators.load_checkpoint()
assert predators.generation == 7
assert np.abs(predators.population).max() <= predators.WEIGHT_CLIP + 1e-9, "loaded pop must be clipped"
assert np.abs(predators.hof_genome).max() <= predators.WEIGHT_CLIP + 1e-9, "loaded King must be clipped"
print("T7 OK: saturated checkpoint loads back clipped into tanh's linear region")

import shutil
shutil.rmtree(TMP, ignore_errors=True)
print("ALL_OK")
