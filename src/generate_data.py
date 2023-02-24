import numpy as np
from os.path import exists
from tqdm import tqdm
from joblib import Parallel, delayed
import json

# environment and agents
from src.envs.team_goofspiel import Team_Goofspiel
from src.actors.team_goofspiel_players import Team_Goofspiel_Agent, Team_Goofspiel_Poisoned_Agent
from src.envs.euchre import Euchre
from src.actors.euchre_players import Euchre_Agent, Euchre_Poisoned_Agent
from src.envs.spades import Spades
from src.actors.spades_players import Spades_Agent, Spades_Poisoned_Agent
# responsibility attribution methods
from src.respTools.brute_force import brute_force
from src.respTools.random_search import random_search
from src.respTools.mcts.mcts import method_no_pruning_no_exploitation, method_no_exploitation, mcts
# responsibility attribution tools
from src.respTools.utils import mean_responsibility
# utilities
from src.utils import *

def generate_data(params):

    print("Begin Generating Data\n")

    # Load Experiment Parameters
    # CPU
    n_jobs = params['n_jobs']
    # experiment mode
    ground_truth = params['ground_truth']
    uncertainty = params['uncertainty']
    # environment configuration
    env_name = params['env_name']
    agent_name = env_name + '_Agent'
    if ground_truth == 'partial':
        poisoned_agent_name = env_name + '_Poisoned_Agent'
    lst_env_size = params['lst_env_size']
    num_agents = params['num_agents']
    # evaluation parameters
    num_trajectories = params['num_trajectories']
    num_seeds = params['num_seeds']
    kappa = params['kappa']
    if ground_truth == 'partial':
        kappa_partial = params['kappa_partial']
    lst_tot_env_steps = params['lst_tot_env_steps']
    frq_env_steps = params['frq_env_steps']
    if uncertainty:
        num_cf_samples = params['num_cf_samples']
    # responsibility attribution methods
    methods = params['methods']
    # mcts parameters
    c = params['c']
    q = params['q']
    w = params['w']

    # Prepare Methods' Configuration
    config = {}
    config['frq_env_steps'] = frq_env_steps
    config['kappa'] = kappa
    if ground_truth == 'partial':
        config['kappa_partial'] = kappa_partial
    config['pruning'] = True
    config['exploitation'] = True
    config['c'] = c
    config['q'] = q
    config['w'] = w

    # Instantiate Environment and Agents, and Generate Trajectories
    d_envs = {}
    d_agents = {}
    d_trajectories = {}
    for env_size in lst_env_size:
        # instantiate and save environment
        env = globals()[env_name](env_size, num_agents)
        d_envs[env_size] = env
        # instantiate non-poisoned agents
        agents = [globals()[agent_name](id) for id in range(num_agents)]
        # generate trajectories
        print(f"Begin generating trajectories with environment size = {env_size}")
        trajectories = []
        # file that stores trajectories in a readable form
        f_traj = open(f'data/trajectories/env_size={env_size}.txt', 'w')
        seed = 1
        count = 0
        # Under Full Knowldege of the Ground Truth
        if ground_truth == 'full':
            while count < num_trajectories:
                rng = np.random.default_rng(seed)
                trajectory = env.sample_trajectory(agents, rng)
                if env._is_trajectory_failed(trajectory):
                    trajectory['id'] = count
                    count += 1
                    env.store_trajectory(f_traj, trajectory)
                    trajectories.append(trajectory)
                seed += 1
            # save non-poisoned agents
            d_agents[env_size] = agents
        # Under Partial Knowledge of the Ground Truth
        else:
            # instantiate poisoned agents by first defining for which time-steps their policies will be poisoned
            rng = np.random.default_rng(0)
            if env.turn_based:
                num_rounds = env.horizon // env.num_turns
                poisoned_rounds = rng.choice(range(num_rounds), kappa_partial, replace=False)
                poisoned_time_steps = []
                for round in poisoned_rounds:
                    poisoned_time_steps += [env.num_turns * round + turn for turn in range(env.num_turns)]
            else:
                poisoned_time_steps = rng.choice(range(env.horizon), kappa_partial, replace=False)
            poisoned_agents = [globals()[poisoned_agent_name](id, poisoned_time_steps) for id in range(num_agents)]
            # save poisoned agents
            d_agents[env_size] = poisoned_agents
            # add the poisoned time-steps to the configuration for the brute force algorithm -- computing partial ground truth
            config['bf_time_steps'] = poisoned_time_steps
            # Generate Poisoned Trajectories
            while count < num_trajectories:
                rng = np.random.default_rng(seed)
                # first generate trajectory in which the non-poisoned agents win
                trajectory = env.sample_trajectory(agents, rng)
                if not env._is_trajectory_failed(trajectory):
                    # then poison the trajectory
                    poisoned_trajectory = env.sample_trajectory(poisoned_agents, given_action_gumbels=trajectory['action_gumbels'], given_env_gumbels=trajectory['env_gumbels'], initial_state=trajectory['states'][0])
                    if env._is_trajectory_failed(poisoned_trajectory):
                        poisoned_trajectory['id'] = count
                        count += 1
                        env.store_trajectory(f_traj, poisoned_trajectory)
                        trajectories.append(poisoned_trajectory)
                seed += 1
        f_traj.close()
        # save trajectories
        d_trajectories[env_size] = trajectories
        print(f"Finish generating tajectories with environment size = {env_size}\n")
    
    # Compute Ground Truth Degrees of Responsibility
    for env_size in lst_env_size:
        if exists(f'data/ground_truth/responsibility/env_size={env_size}.json'):
            # if allready computed then do not compute them again
            continue
        with tqdm_joblib(tqdm(desc=f"Computing Ground Truth Degrees of Responsibility for Environment Size = {env_size}", total=num_trajectories)) as progress_bar:
            results = Parallel(n_jobs=n_jobs)(
                delayed(brute_force)(d_envs[env_size], d_agents[env_size], trajectory, trajectory['action_gumbels'], trajectory['env_gumbels'], config)
                for trajectory in d_trajectories[env_size]
            )
        # unpack and store results
        gt_responsibility = {}  # ground truth degrees of responsibility
        responsibility = {}     # brute force with limited environment steps (baseline method)
        actual_causes = {}
        for i in range(num_trajectories):
            last_batch = results[i]['env_steps'] // frq_env_steps + (results[i]['env_steps'] % frq_env_steps != 0)
            gt_responsibility[f'trajectory {i}'] = results[i]['responsibility'][f'env_steps={last_batch * frq_env_steps}']
            responsibility[f'trajectory {i}'] = results[i]['responsibility']
            actual_causes[f'trajectory {i}'] = results[i]['actual_causes']
        with open(f'data/ground_truth/responsibility/env_size={env_size}.json', 'w') as f:
            json.dump(gt_responsibility, f, indent=4)
        with open(f'data/methods/responsibility/brute_force/env_size={env_size}.json', 'w') as f:
            json.dump(responsibility, f, indent=4)
        with open(f'data/ground_truth/actual_causes/env_size={env_size}.json', 'w') as f:
            json.dump(actual_causes, f, indent=4)
        env_steps = [result['env_steps'] for result in results]
        with open(f'data/ground_truth/env_steps.txt', 'a') as f:
            f.write(f"Max, mean and std of total number of environment steps for env_size={env_size}: {np.max(env_steps)}, {np.mean(env_steps)} and {np.std(env_steps)}\n")
        tot_time = [result['tot_time'] for result in results]
        with open(f'data/ground_truth/time.txt', 'a') as f:
            f.write(f"Max, mean and std of total execution time for env_size={env_size}: {np.max(tot_time)}, {np.mean(tot_time)} and {np.std(tot_time)}\n")
    # (if there) remove brute force baseline method from list of responsibility attribution methods, in order to not execute it again
    methods = [method for method in methods if method != 'brute_force']
    
    # Compute Methods' Degrees of Responsibility
    # Under No Uncertainty over the Context
    if not uncertainty:
        # Execute Responsibility Attribution Methods
        for env_size in lst_env_size:
            tot_env_steps = lst_tot_env_steps[lst_env_size.index(env_size)]
            config['tot_env_steps'] = tot_env_steps
            responsibility = {}
            for method in methods:
                responsibility = {}
                tot_time = []
                for seed in range(num_seeds):
                    rng = np.random.default_rng(seed)
                    responsibility[f'seed={seed}'] = {}
                    with tqdm_joblib(tqdm(desc=f"Executing {method} for Environment Size = {env_size} and Seed={seed}", total=num_trajectories)) as progress_bar:
                        results = Parallel(n_jobs=n_jobs)(
                            delayed(globals()[method])(d_envs[env_size], d_agents[env_size], trajectory, trajectory['action_gumbels'], trajectory['env_gumbels'], rng, config)
                            for trajectory in d_trajectories[env_size]
                        )
                    # unpack and store results
                    for i in range(num_trajectories):
                        responsibility[f'seed={seed}'][f'trajectory {i}'] = results[i]['responsibility']
                    tot_time += [result['tot_time'] for result in results]
                with open(f'data/methods/time/{method}/env_size={env_size}.txt', 'a') as f:
                    f.write(f"Max, mean and std of total execution time for env_size={env_size}: {np.max(tot_time)}, {np.mean(tot_time)} and {np.std(tot_time)}\n")
                with open(f'data/methods/responsibility/{method}/env_size={env_size}.json', 'w') as f:
                    json.dump(responsibility, f, indent=4)
    # Under Uncertainty over the Context
    else:
        # Sample Counterfactual Gumbels from the Posterior Distribution
        print("Begin Sampling Counterfactal Gumbels")
        for env_size in lst_env_size:
            for trajectory in d_trajectories[env_size]:
                # sample
                cf_action_gumbels, cf_env_gumbels = d_envs[env_size].sample_cf_gumbels(trajectory, d_agents[env_size], num_cf_samples)
                # sanity check
                for sample in range(num_cf_samples):
                    cf_trajectory = d_envs[env_size].sample_trajectory(d_agents[env_size], given_action_gumbels=cf_action_gumbels[sample], given_env_gumbels=cf_env_gumbels[sample], initial_state=trajectory['states'][0])
                    assert trajectory['states'] == cf_trajectory['states'], 'Sanity Check for counterfactual gumbels has failed!'
                # attach counterfactual gumbels to trajectory
                trajectory['cf_env_gumbels'] = cf_env_gumbels
                trajectory['cf_action_gumbels'] = cf_action_gumbels
        print("Finish Sampling Counterfactal Gumbels\n")
        # Execute Responsibility Attribution Methods
        for env_size in lst_env_size:
            tot_env_steps = lst_tot_env_steps[lst_env_size.index(env_size)]
            config['tot_env_steps'] = tot_env_steps
            responsibility = {}
            for method in methods:
                responsibility = {}
                tot_time = []
                for seed in range(num_seeds):
                    rng = np.random.default_rng(seed)
                    responsibility[f'seed={seed}'] = {}
                    # compute sampled degrees of responsibility
                    samples_responsibility = {}
                    for i in range(num_trajectories):
                        responsibility[f'seed={seed}'][f'trajectory {i}'] = {}
                        samples_responsibility[i] = []
                    for sample in range(num_cf_samples):
                        with tqdm_joblib(tqdm(desc=f"Executing {method} for Environment Size = {env_size}, Seed={seed} and Sample Number = {sample}", total=num_trajectories)) as progress_bar:
                            results = Parallel(n_jobs=n_jobs)(
                                delayed(globals()[method])(d_envs[env_size], d_agents[env_size], trajectory, trajectory['cf_action_gumbels'][sample], trajectory['cf_env_gumbels'][sample], rng, config)
                                for trajectory in d_trajectories[env_size]
                            )
                        # unpack results
                        for i in range(num_trajectories):
                            samples_responsibility[i].append(results[i]['responsibility'])                      
                        tot_time += [result['tot_time'] for result in results]
                    # compute mean and std over samples
                    for i in range(num_trajectories):
                        responsibility[f'seed={seed}'][f'trajectory {i}'] = {}
                        for env_steps in range(0, tot_env_steps + 1, frq_env_steps):
                            responsibility[f'seed={seed}'][f'trajectory {i}'][f'env_steps={env_steps}'] = {}
                            responsibility[f'seed={seed}'][f'trajectory {i}'][f'env_steps={env_steps}']['mean'] = [np.mean([samples_responsibility[i][sample][f'env_steps={env_steps}'][ag] for sample in range(num_cf_samples)]) for ag in range(num_agents)]
                            responsibility[f'seed={seed}'][f'trajectory {i}'][f'env_steps={env_steps}']['std'] = [np.std([samples_responsibility[i][sample][f'env_steps={env_steps}'][ag] for sample in range(num_cf_samples)]) for ag in range(num_agents)]
                # store results
                with open(f'data/methods/time/{method}/env_size={env_size}.txt', 'a') as f:
                    f.write(f"Mean and std of total execution time of method {method} for env_size={env_size}: {np.mean(tot_time)}, {np.std(tot_time)}\n")    
                with open(f'data/methods/responsibility/{method}/env_size={env_size}.json', 'w') as f:
                    json.dump(responsibility, f, indent=4)

    print("\nFinish Generating Data\n")

    return