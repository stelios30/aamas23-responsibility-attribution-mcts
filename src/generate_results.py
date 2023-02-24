import numpy as np
import json

def generate_results(params):
    
    print("Begin Generating Results\n")

    # Load Experiment Parameters
    # experiment mode
    ground_truth = params['ground_truth']
    uncertainty = params['uncertainty']
    # environment configuration
    num_agents = params['num_agents']
    lst_env_size = params['lst_env_size']
    # evaluation parameters
    num_trajectories = params['num_trajectories']
    num_seeds = params['num_seeds']
    lst_tot_env_steps = params['lst_tot_env_steps']
    frq_env_steps = params['frq_env_steps']
    d = params['d']
    if uncertainty:
        lst_d_unc = params['lst_d_unc']
    # responsibility attribution method
    methods = params['methods']

    # Load Data
    # ground truth
    gt_responsibility = {}
    for env_size in lst_env_size:
        with open(f'data/ground_truth/responsibility/env_size={env_size}.json', 'r') as f:
            gt_responsibility[env_size] = json.load(f)
    # responsibility attribution methods
    responsibility = {}
    for env_size in lst_env_size:
        tot_env_steps = lst_tot_env_steps[lst_env_size.index(env_size)]
        responsibility[env_size] = {}
        for method in methods:
            with open(f'data/methods/responsibility/{method}/env_size={env_size}.json', 'r') as f:
                responsibility[env_size][f'method={method}'] = json.load(f)
        # if brute force baseline is in methods, propagate last value if needed
        if 'brute_force' in methods:
            for i in range(num_trajectories):
                num_batches = len(responsibility[env_size]['method=brute_force'][f'trajectory {i}']) - 1
                if num_batches < tot_env_steps // frq_env_steps:
                    last_responsibility = responsibility[env_size]['method=brute_force'][f'trajectory {i}'][f'env_steps={num_batches * frq_env_steps}']
                    for batch in range(num_batches + 1, tot_env_steps//frq_env_steps + 1):
                        responsibility[env_size]['method=brute_force'][f'trajectory {i}'][f'env_steps={batch * frq_env_steps}'] = last_responsibility

    #
    if not uncertainty:
        d_list = [d]
    else:
        d_list = lst_d_unc
    
    # Compare Degrees of Responsibility w.r.t. the Ground Truth
    for threshold in d_list:
        resuts = {}
        for env_size in lst_env_size:
            tot_env_steps = lst_tot_env_steps[lst_env_size.index(env_size)]
            resuts[f'env_size={env_size}'] = {}
            for method in methods:
                resuts[f'env_size={env_size}'][f'method={method}'] = {}
                if method == 'brute_force':
                    # no seeding for brute_force baseline method
                    for env_steps in range(0, tot_env_steps + 1, frq_env_steps):
                        resuts[f'env_size={env_size}'][f'method={method}'][f'env_steps={env_steps}'] = {}
                        F_traj_lst = []
                        for i in range(num_trajectories):
                            F_traj_lst.append(np.all([
                                abs(responsibility[env_size][f'method={method}'][f'trajectory {i}'][f'env_steps={env_steps}'][ag] - gt_responsibility[env_size][f'trajectory {i}'][ag]) <= threshold
                                for ag in range(num_agents)
                            ]))
                        resuts[f'env_size={env_size}'][f'method={method}'][f'env_steps={env_steps}']['mean'] = np.mean(F_traj_lst)
                        resuts[f'env_size={env_size}'][f'method={method}'][f'env_steps={env_steps}']['std'] = 0
                else:
                    for env_steps in range(0, tot_env_steps + 1, frq_env_steps):
                        resuts[f'env_size={env_size}'][f'method={method}'][f'env_steps={env_steps}'] = {}
                        F_traj_lst = []
                        for i in range(num_trajectories):
                            lst_traj = []
                            for seed in range(num_seeds):
                                if ground_truth == 'full' and not uncertainty:
                                    lst_traj.append(np.all([
                                        abs(responsibility[env_size][f'method={method}'][f'seed={seed}'][f'trajectory {i}'][f'env_steps={env_steps}'][ag] - gt_responsibility[env_size][f'trajectory {i}'][ag]) <= threshold
                                        for ag in range(num_agents)
                                    ]))
                                elif not uncertainty:
                                    lst_traj.append(np.all([
                                        abs(min(0, responsibility[env_size][f'method={method}'][f'seed={seed}'][f'trajectory {i}'][f'env_steps={env_steps}'][ag] - gt_responsibility[env_size][f'trajectory {i}'][ag])) <= threshold
                                        for ag in range(num_agents)
                                    ]))
                                elif ground_truth == 'full':
                                    lst_traj.append(np.all([
                                        abs(responsibility[env_size][f'method={method}'][f'seed={seed}'][f'trajectory {i}'][f'env_steps={env_steps}']['mean'][ag] - gt_responsibility[env_size][f'trajectory {i}'][ag]) <= threshold
                                        for ag in range(num_agents)
                                    ]))
                                else:
                                    lst_traj.append(np.all([
                                        abs(min(0, responsibility[env_size][f'method={method}'][f'seed={seed}'][f'trajectory {i}'][f'env_steps={env_steps}']['mean'][ag] - gt_responsibility[env_size][f'trajectory {i}'][ag])) <= threshold
                                        for ag in range(num_agents)
                                    ]))
                            F_traj_lst.append(np.mean(lst_traj))
                        # report the mean and standard deviation over multiple seeds
                        resuts[f'env_size={env_size}'][f'method={method}'][f'env_steps={env_steps}']['mean'] = np.mean(F_traj_lst)
                        resuts[f'env_size={env_size}'][f'method={method}'][f'env_steps={env_steps}']['std'] = np.sqrt(np.sum([F_traj * (1 - F_traj) for F_traj in F_traj_lst])/num_seeds)/num_trajectories
        # Store Results
        with open(f'results/threshold={threshold}.json', 'w') as f:
            json.dump(resuts, f, indent=4)
    
    print("Finish Generating Results\n")

    return

def is_within_range(x, y, z):
    """
    Checks if x is within z range from y
    """
    return x >= y - z and x <= y + z