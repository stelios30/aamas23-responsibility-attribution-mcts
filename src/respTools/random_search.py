import numpy as np
import time
import math

from src.envs.causal_env import Causal_Env
from src.respTools.utils import compute_actual_causes, CH_responsibility, _get_new_intervention

def random_search(env: Causal_Env, agents, trajectory, action_gumbels, env_gumbels, rng, config):
    """
    Random Search responsibility attribution method
    - First, it performs random interventions to search for actual causes under some computational budget (tot_env_steps)
    - Then based on the found actual causes, it computes the agents' degrees of responsibility
    - Monitors progress with step frequency = frq_env_steps
    """
    kappa = config['kappa']
    tot_env_steps = config['tot_env_steps']
    frq_env_steps = config['frq_env_steps']
    
    start = time.time()

    responsibility = {}
    # initial attributed responsibility is zero for both agents
    responsibility[f'env_steps={0}'] = [0 for _ in agents]
    # execute search method
    actual_causes = []
    extra_steps = 0
    for batch in range(1, tot_env_steps//frq_env_steps + 1):
        algorithm = Random_Search(env, agents, trajectory, action_gumbels, env_gumbels, kappa, rng, frq_env_steps + extra_steps)
        # search for candidate actual causes
        algorithm.search()
        extra_steps = frq_env_steps - algorithm.env_steps
        # apply minimality condition
        actual_causes = compute_actual_causes(actual_causes + algorithm.candidate_actual_causes)
        # compute agents' degrees of responsibility
        responsibility[f'env_steps={batch * frq_env_steps}'] = []
        for agent in agents:
            degree = CH_responsibility(actual_causes, agent)
            responsibility[f'env_steps={batch * frq_env_steps}'].append(degree)

    end = time.time()

    output = {'responsibility': responsibility, 'tot_time': end - start}
    
    return output

class Random_Search():
    
    def __init__(self, env: Causal_Env, agents, trajectory, action_gumbels, env_gumbels, kappa, rng, tot_env_steps):

        self.env = env
        self.agents = agents
        self.trajectory = trajectory
        self.action_gumbels = action_gumbels
        self.env_gumbels = env_gumbels
        self.kappa = kappa
        self.rng = rng
        self.tot_env_steps = tot_env_steps

        # list of candidate actual causes
        self.candidate_actual_causes = []
        # number of environment steps taken by the algorithm
        self.env_steps = 0

        return

    def search(self):
        """
        Performs random interventions to search for actual causes
        """
        env = self.env
        trajectory = self.trajectory
        kappa = self.kappa
        rng = self.rng
        horizon = env.horizon

        while self.env_steps < self.tot_env_steps - horizon:
            # sample how many and which actions to intervene to
            # all subsets of actions (with size at most kappa) have the same probability of being chosen
            tot_num_actions = env._get_num_remaining_actions(trajectory['states'][0])
            tot_num_combinations = sum([math.comb(tot_num_actions, k) for k in range(1, kappa + 1)])
            p=[math.comb(tot_num_actions, k) / tot_num_combinations  for k in range(1, kappa + 1)]
            num_interventions = rng.choice(range(1, kappa + 1), p=p)
            intervention_idxs = rng.choice(range(tot_num_actions), num_interventions, replace=False)
            # perform interventions on the actions with the selected indexes
            interventions, outcome_improved = self.rollout(intervention_idxs)
            if outcome_improved:
                # include in candidate actual causes
                self.candidate_actual_causes.append(interventions)
            
        return

    def rollout(self, intervention_idxs):
        
        env = self.env
        agents = self.agents
        trajectory = self.trajectory
        rng = self.rng

        state = trajectory['states'][0]
        last_info_states = [None] * env.num_agents
        last_actions = [None] * env.num_agents
        action_idx = 0
        interventions = []

        for t in range(env.horizon):
            # gumbels
            action_gumbels = self.action_gumbels[t]
            env_gumbels = self.env_gumbels[t]
            # determine agents' actions
            default_actions, info_states = env._get_default_actions(state, agents, env_gumbels, action_gumbels, last_info_states, last_actions)
            actions = default_actions
            acting_agents = env._get_acting_agents(state, agents)
            for agent in acting_agents:
                if action_idx in intervention_idxs:
                    # perform an intervention (if there are any alternative actions to take)
                    valid_actions = env._get_valid_actions(state, agent.id)
                    alternative_actions = [action for action in valid_actions if action != default_actions[agent.id]]
                    if alternative_actions:
                        arr = np.array(alternative_actions, dtype=object)   # for the type of action to not change (np int not json serializable)
                        actions[agent.id] = rng.choice(arr)
                        interventions.append(_get_new_intervention(self.trajectory, t, agent, info_states[agent.id], default_actions[agent.id], actions[agent.id]))
                # update the action index
                action_idx += 1

            # prepare for next time-step
            _, _, state = env.perform_time_step(agents, state, env_gumbels, action_gumbels, last_info_states, last_actions, predefined_actions=actions)
            self.env_steps += 1
            last_info_states = info_states
            last_actions = actions

        # evaluate counterfactual trajectory
        outcome_improved = env._is_final_outcome_improved(trajectory['states'][-1], state)

        return interventions, outcome_improved

