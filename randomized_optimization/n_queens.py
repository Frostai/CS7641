# https://github.com/hiive/mlrose/tree/master/mlrose_hiive/runners

import numpy as np
import mlrose_hiive as mlrose
from plotting import plot_ro_curve_df

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True, max_val=8)
SEED = 42

rhc = mlrose.RHCRunner(
    problem=problem,
    experiment_name="RandomizedHillClimbing",
    seed=SEED,
    iteration_list=[1, 10, 50, 100, 250, 500, 1000],
    max_attempts=1000,
    restart_list=[0],
)

sa = mlrose.SARunner(
    problem=problem,
    experiment_name="SimmulatedAnnealing",
    seed=SEED,
    iteration_list=[1, 10, 50, 100, 250, 500, 1000],
    max_attempts=1000,
    temperature_list=[250],
    decay_list=[mlrose.GeomDecay],
)

ga = mlrose.GARunner(
    problem=problem,
    experiment_name="GeneticAlgorithm",
    seed=SEED,
    iteration_list=[1, 10, 50, 100, 250, 500, 1000],
    max_attempts=1000,
    population_sizes=[200],
    mutation_rates=[0.3],
)

mimic = mlrose.MIMICRunner(
    problem=problem,
    experiment_name="MIMIC",
    seed=SEED,
    iteration_list=[1, 10, 50, 100, 250, 500],
    population_sizes=[200],
    max_attempts=500,
    keep_percent_list=[0.2],
    use_fast_mimic=True,
)


print('RHC')
rhc_run_stats, rhc_run_curves = rhc.run()
print('SA')
sa_run_stats, sa_run_curves = sa.run()
print('GA')
ga_run_stats, ga_run_curves = ga.run()
print('MIMIC')
mimic_run_stats, mimic_run_curves = mimic.run()
print(mimic_run_stats)
print(mimic_run_curves)
plot_ro_curve_df(rhc_run_curves, "Randomized Hill Climbing", 'nqueens')
plot_ro_curve_df(ga_run_curves, "Genetic Algorithm", 'nqueens')
plot_ro_curve_df(sa_run_curves, "Simulated Annealing", 'nqueens')
plot_ro_curve_df(mimic_run_curves, "MIMIC", 'nqueens')
