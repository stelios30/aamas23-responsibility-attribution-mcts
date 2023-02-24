import click
import multiprocessing
import time

from src.generate_data import generate_data
from src.generate_results import generate_results
from src.generate_plots import generate_plots

@click.command()
# CPU
@click.option('--n_jobs', default=multiprocessing.cpu_count(), type=int, help='Number of jobs')
# experiment mode
@click.option('--ground_truth', default='full', type=click.Choice(['full', 'partial']), help='Considered availability for ground truth actual causes')
@click.option('--uncertainty', is_flag=True, help='Uncertainty over the context')
# environment configuration
@click.option('--env_name', default='Euchre', type=click.Choice(['Team_Goofspiel', 'Euchre', 'Spades']), help='Name of the environment')
@click.option('--lst_env_size', multiple=True, default=[8, 9, 10], type=int, help='List of values for parameter N -- environment size')
@click.option('--num_agents', default=2, type=int, help='Number of agents')
# evaluation parameters
@click.option('--num_trajectories', default=50, type=int, help='Number of trajectories')
@click.option('--num_seeds', default=10, type=int, help='Number of seeds with which the several implemented methods will be executed')
@click.option('--kappa', default=4, type=int, help='Parameter kappa -- maximum actual cause size')
@click.option('--kappa_partial', default=5, type=int, help='Parameter kappa partial (kappa prime)')
@click.option('--lst_tot_env_steps', multiple=True, default=[100000, 300000, 500000], type=int, help='List of computational budgets: total number of environment steps corresponding to each of the given environment sizes')
@click.option('--frq_env_steps', default=100, type=int, help='Frequency (in terms of computational budget) of registering algorithmic results')
@click.option('--num_cf_samples', default=10, type=int, help='Number of counterfactual samples')
@click.option('--d', default=0, type=click.FloatRange(0, 1), help='Threshold D for experiments with known context.')
@click.option('--lst_d_unc', multiple=True, default=[.15, .2, .25], type=click.FloatRange(0, 1), help='List of thresholds D for experiments with uncertainty over the context.')
# responsibility attribution methods
@click.option('--methods', multiple=True, default=['brute_force', 'random_search', 'method_no_pruning_no_exploitation', 'mcts'], type=click.Choice(['brute_force', 'random_search', 'method_no_pruning_no_exploitation', 'method_no_exploitation', 'mcts']), help='List of responsibility attribution methods')
# mcts parameters
@click.option('--c', default=2, type=float, help='MCTS exploration parameter')
@click.option('--q', default=0, type=float, help='MCTS mixmax parameter')
@click.option('--w', default=.5, type=float, help='MCTS cf_change weight')
# 
def run_experiment(n_jobs, ground_truth, uncertainty, env_name, lst_env_size, num_agents, num_trajectories, num_seeds, kappa, kappa_partial, lst_tot_env_steps, frq_env_steps, num_cf_samples, d, lst_d_unc, methods, c, q, w):

    start = time.time()
    print("Begin Experiment\n")

    params = {}
    # CPU
    params['n_jobs'] = n_jobs
    # experiment mode
    params['ground_truth'] = ground_truth
    params['uncertainty'] = uncertainty
    # environment configuration
    params['env_name'] = env_name
    params['lst_env_size'] = lst_env_size
    params['num_agents'] = num_agents
    # evaluation parameters
    params['num_trajectories'] = num_trajectories
    params['num_seeds'] = num_seeds
    params['kappa'] = kappa
    if ground_truth == 'partial':
        params['kappa_partial'] = kappa_partial
    params['lst_tot_env_steps'] = lst_tot_env_steps
    params['frq_env_steps'] = frq_env_steps
    params['d'] = d
    if uncertainty:
        params['num_cf_samples'] = num_cf_samples
        params['lst_d_unc'] = lst_d_unc
    # responsibility attribution methods
    params['methods'] = methods
    # mcts parameters
    params['c'] = c
    params['q'] = q
    params['w'] = w

    assert not uncertainty or 'brute_force' not in methods, "The brute_force baseline method is computed only under no uncertainty over the context."
    assert ground_truth == 'full' or 'brute_force' not in methods, "The brute_force baseline method is computed only for experiments with full knowledge of the ground truth."
    assert len(lst_env_size) == len(lst_tot_env_steps), "Unequal number of environment sizes and number of environment steps is given."
    assert all(frq_env_steps <= tot_env_steps for tot_env_steps in lst_tot_env_steps), "Frequency of environment steps is larger than the total number of environment steps."
    assert ground_truth == 'full' or all(kappa_partial <= env_size for env_size in lst_env_size), "kappa_partial has to be at most equal to env_size."

    generate_data(params)
    generate_results(params)
    generate_plots(params)

    end = time.time()
    print(f"Total time of experiment = {end - start}\n")
    print("Finish experiment")

if __name__ == '__main__':
    
    run_experiment()