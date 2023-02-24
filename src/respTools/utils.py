import numpy as np

from src.envs.causal_env import Causal_Env

"""
Utilities for responsibility attribution methods
"""
def _get_new_intervention(trajectory, t, agent, info_state, default_action, new_action):
    """
    Returns a new intervention
    """
    new_intervention = {
        't': t,
        'agent': agent.id,
        'default_action': default_action,
        'new_action': new_action,
        'contingency': info_state != trajectory['info_states'][t][agent.id]
    }

    return new_intervention

def compute_actual_causes(candidate_actual_causes):
    """
    Filters out non-minimal candidate actual causes
    - Does not remove duplicates!
    """
    actual_causes = list(filter(
        lambda f: not any(
            set((intervention['t'], intervention['agent']) for intervention in f) > set((intervention['t'], intervention['agent']) for intervention in g) 
            for g in candidate_actual_causes
        ), candidate_actual_causes
    ))

    return actual_causes

def CH_responsibility(actual_causes, agent):
    """
    Computes the agent's degree of responsibility according to the Chockler and Halpern definition
    """
    resp = 0
    for actual_cause in actual_causes:
        m = len([i for i in actual_cause if i['agent'] == agent.id and not i['contingency']])
        k = len(actual_cause)
        if m/k > resp:
            resp = m/k
    
    return resp

"""
Utilities for responsibility attribution under uncertainty
"""
def mean_responsibility(lst_responsibility):
    """
    Computes the mean degrees of responsibility
    """
    num_agents = len(lst_responsibility[0])
    
    mean_responsibility = []
    std_responsibility = []
    for ag in range(num_agents):
        mean_responsibility.append(np.mean([responsibility[ag] for responsibility in lst_responsibility]))
        std_responsibility.append(np.std([responsibility[ag] for responsibility in lst_responsibility]))

    return mean_responsibility, std_responsibility


