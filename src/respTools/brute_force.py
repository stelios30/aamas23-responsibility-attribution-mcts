import time
import copy
from itertools import product

from src.envs.causal_env import Causal_Env
from src.respTools.utils import compute_actual_causes, CH_responsibility

def brute_force(env: Causal_Env, agents, trajectory, action_gumbels, env_gumbels, config: dict):
    """
    Brute Force responsibility attribution method
    - First, it exhaustively searches for all actual causes
    - Then based on the found actual causes, it computes the agents' degrees of responsibility
    - Monitors progress with step frequency = frq_env_steps -- for brute_force baseline method
    - If bf_time_steps is in configuration, the the brute force algorithm is restricted only to the actions that take place in these time-steps
    """
    kappa = config['kappa']
    frq_env_steps = config['frq_env_steps']
    if 'bf_time_steps' in config.keys():
        # brute force applied only in specific time-steps
        bf_time_steps = config['bf_time_steps']
    else:
        # brute force applied in all time-steps
        bf_time_steps = range(env.horizon)

    start = time.time()

    responsibility = {}
    actual_causes = []
    # execute search method
    algorithm = Brute_Force(env, agents, trajectory, action_gumbels, env_gumbels, kappa, frq_env_steps, bf_time_steps)
    algorithm.search(algorithm.root_node)
    # total number of environment steps taken
    env_steps = algorithm.env_steps
    # compute agents' degrees of responsibility
    for batch in range(0, len(algorithm.candidate_actual_causes)):
        # apply minimality condition
        actual_causes = compute_actual_causes(actual_causes + algorithm.candidate_actual_causes[f'env_steps={batch * frq_env_steps}'])
        # compute agents' degrees of responsibility
        responsibility[f'env_steps={batch * frq_env_steps}'] = []
        for agent in agents:
            degree = CH_responsibility(actual_causes, agent)
            responsibility[f'env_steps={batch * frq_env_steps}'].append(degree)

    end = time.time()

    output = {'responsibility': responsibility, 'actual_causes': actual_causes, 'env_steps': env_steps, 'tot_time': end - start}

    return output


class BFNode():
    
    def __init__(self, parent, t, state, last_info_states, last_actions, interventions):
        """
        - parent: parent node
        - t: time-step
        - state: underlying environment state
        - last_info_states: agents' last information states
        - last_actions: agents' last actions
        - interventions: set of interventions performed so far in the tree
        """
        self.parent = parent
        self.t = t
        self.state = state
        self.last_info_states = last_info_states
        self.last_actions = last_actions
        self.interventions = interventions

        self.children = {}

        return

class Brute_Force():

    def __init__(self, env: Causal_Env, agents, trajectory, action_gumbels, env_gumbels, kappa, frq_env_steps, bf_time_steps):
    
        self.env = env
        self.agents = agents
        self.trajectory = trajectory
        self.action_gumbels = action_gumbels
        self.env_gumbels = env_gumbels
        self.kappa = kappa
        self.frq_env_steps = frq_env_steps
        self.bf_time_steps = bf_time_steps
        
        # list of candidate actual causes segmented in environment step batches
        self.candidate_actual_causes = {}
        self.candidate_actual_causes[f'env_steps={0}'] = []
        # number of environment steps taken by the algorithm
        self.env_steps = 0
        
        # root node
        initial_state = trajectory['states'][0]
        proxy_node = BFNode(None, None, None, [None] * env.num_agents, [None] * env.num_agents, None)
        self.root_node = BFNode(proxy_node, 0, initial_state, [None] * env.num_agents, [None] * env.num_agents, [])

        return

    def search(self, node: BFNode):
        """
        Performs an exhaustive tree search (brute force) in order to identify all actual causes of size at most kappa
        """ 
        env = self.env
        agents = self.agents
        trajectory = self.trajectory
        kappa = self.kappa

        t = node.t
        state = node.state
        last_info_states = node.last_info_states
        last_actions = node.last_actions
        interventions = copy.deepcopy(node.interventions)
        num_interventions = len(interventions)

        # new batch starts
        if self.env_steps % self.frq_env_steps == 0:
            self.candidate_actual_causes[f'env_steps={self.env_steps + self.frq_env_steps}'] = []

        # terminal node
        if t == env.horizon:
            if env._is_final_outcome_improved(trajectory['states'][-1], state):
                # find current batch number
                batch_num = self.env_steps // self.frq_env_steps + 1
                # include in candidate actual causes
                self.candidate_actual_causes[f'env_steps={batch_num * self.frq_env_steps}'].append(interventions)
            return

        # gumbels
        action_gumbels = self.action_gumbels[t]
        env_gumbels = self.env_gumbels[t]
        
        # take default actions (no interventions)
        info_states, default_actions, next_state = env.perform_time_step(agents, state, env_gumbels, action_gumbels, last_info_states, last_actions)
        self.env_steps += 1
        # expand node
        hashable_actions = [env._get_readable_form(action) for action in default_actions]
        node.children[tuple(hashable_actions)] = BFNode(node, t + 1, next_state, info_states, default_actions, interventions)
        self.search(node.children[tuple(hashable_actions)])

        # perform interventions
        if num_interventions < kappa and t in self.bf_time_steps:
            # consider all valid alternative joint actions
            valid_joint_actions = [list(tup) for tup in product(*[env._get_valid_actions(state, ag.id) for ag in agents])]
            alternative_joint_actions = [actions for actions in valid_joint_actions if actions != default_actions]
            for actions in alternative_joint_actions:
                # make sure that the total number of interventions does not exceed kappa
                num_new_interventions = len([a for a, b in zip(default_actions, actions) if a != b])
                if num_interventions + num_new_interventions > kappa:
                    continue
                info_states, _, next_state = env.perform_time_step(agents, state, env_gumbels, action_gumbels, last_info_states, last_actions, predefined_actions=actions)
                self.env_steps += 1
                # expand node
                updated_interventions = self._get_updated_interventions(t, info_states, default_actions, actions, interventions)
                hashable_actions = [env._get_readable_form(action) for action in actions]
                node.children[tuple(hashable_actions)] = BFNode(node, t + 1, next_state, info_states, actions, updated_interventions)
                self.search(node.children[tuple(hashable_actions)])
            
        return

    def _get_updated_interventions(self, t, info_states, default_actions, new_actions, interventions):
        """
        Updates set of interventions
        """
        env = self.env
        updated_interventions = copy.deepcopy(interventions)
        for ag in self.agents:
            if default_actions[ag.id] != new_actions[ag.id]:
                updated_interventions.append({
                    't': t,
                    'agent': ag.id,
                    'default_action': env._get_readable_form(default_actions[ag.id]),
                    'new_action': env._get_readable_form(new_actions[ag.id]),
                    'contingency': info_states[ag.id] != self.trajectory['info_states'][t][ag.id]
                })
                
        return updated_interventions

