import numpy as np

from src.gumbelTools import _sample_gumbels

class Causal_Env():

    def __init__(self):
        """
        Required variables
        - num_agents
        - horizon
        - turn_based
        - num_turns (if turn_based)
        - scaling_factor
        """
        self.num_agents = None
        self.horizon = None
        self.turn_based = None
        # number of turns per round
        self.num_turns = None
        self.scaling_factor = 1

        return

    def reset(self, rng=None):
        """
        Input: <rng(random generator)>
        Output: initial state
        - Resets the environment
        """
        raise NotImplementedError

    def step(self, state, actions, env_gumbels):
        """
        Input: state, actions, env_gumbels(dict)
        Output: next_state, info
        - Performs an environment step
        """
        raise NotImplementedError

    def sample_trajectory(self, agents, rng=None, given_action_gumbels=None, given_env_gumbels=None, initial_state=None):
        """
        INPUT: agents(objects), rng(random generator), given action_gumbles, given env_gumbels, initial state
        OUTPUT: trajectory
        - Samples a trajectory from the environment
        - A trajectory includes states, agents' information states, agents' actions, env_gumbels and action_gumbels
        - If gumbels are not given then they are randomly generated (same for initial state)
        """
        trajectory = {
            'states' : [],
            'info_states' : [],
            'actions' : [],
            'env_gumbels' : [],
            'action_gumbels' : [],
        }
        # initial state
        if initial_state is None:
            state = self.reset(rng)
        else:
            state = initial_state
        # last agents' information states and actions
        last_info_states = [None] * self.num_agents
        last_actions = [None] * self.num_agents

        for t in range(self.horizon):
            # gumbels
            if given_env_gumbels:
                env_gumbels = given_env_gumbels[t]
            elif self.turn_based and t % self.num_turns:
                # in turn-based environments compute env_gumbels only in the beginning of each round
                env_gumbels = trajectory['env_gumbels'][self.num_turns * (t//self.num_turns)]
            else:
                env_gumbels = self._get_env_gumbels(rng, state)
            if given_action_gumbels:
                action_gumbels = given_action_gumbels[t]
            elif self.turn_based and t % self.num_turns:
                # in turn-based environments compute action_gumbels only in the beginning of each round
                action_gumbels = trajectory['action_gumbels'][self.num_turns * (t//self.num_turns)]
            else:
                num_available_actions = [len(self._get_available_actions(state, ag.id)) for ag in agents]
                action_gumbels = self._get_action_gumbels(rng, num_available_actions)
            # update trajectory
            trajectory['states'].append(state)
            trajectory['env_gumbels'].append(env_gumbels)
            trajectory['action_gumbels'].append(action_gumbels)
            # perform one full time-step
            info_states, actions, state = self.perform_time_step(agents, state, env_gumbels, action_gumbels, last_info_states, last_actions)
            # update trajectory
            trajectory['info_states'].append(info_states)
            trajectory['actions'].append(actions)
            # prepare for next time-step
            last_info_states = info_states
            last_actions = actions
        # final state
        trajectory['states'].append(state)

        return trajectory

    # make consistent the order agents, state ...
    def perform_time_step(self, agents, state, env_gumbels, action_gumbels, last_info_states, last_actions, predefined_actions=None):
        """
        INPUT: agents, state, env_gumbels, action gumbels, agents' last actions and last information states, 
               predefined_actions is not None when actions are pre-defined
        OUTPUT: current state's agents' information states and taken actions , and the next state
        - Performs one full time-step in the environment:
            - Computes agents' observations and information states
            - Find agents' actions (if not pre-defined)
            - Performs an environment step using the self.step() function
        """
        # agents' observations
        obs = self._get_obs(state, env_gumbels)
        # agents' information states
        info_states = [ag._get_info_state(obs[ag.id], last_actions[ag.id], last_info_states[ag.id]) for ag in agents]
        # determine agents who act in the current state
        acting_agents = self._get_acting_agents(state, agents)
        # a None action is taken by each agent who is not acting on this state
        actions = [None] * self.num_agents
        for agent in acting_agents:
            if predefined_actions is not None:
                actions[agent.id] = predefined_actions[agent.id]
            else:
                actions[agent.id] = agent._get_action(info_states[agent.id], action_gumbels[agent.id])
        # next state
        next_state = self.step(state, actions, env_gumbels)

        return info_states, actions, next_state
            
    def sample_cf_gumbels(self, trajectory, agents, num_cf_samples):
        """
        INPUT: trajectory, agents and number of counterfactual samples
        OUTPUT: countefractual action_gumbels and env_gumbels
        - Samples counterfactual gumbels from trajectory's posterior distribution
        """
        action_gumbels = [[] for _ in range(num_cf_samples)]
        env_gumbels = [[] for _ in range(num_cf_samples)]

        # the procedure is different between simutaneous-actions and turn-based environments
        if not self.turn_based:
            with np.errstate(divide='ignore'):
                for cf_sample in range(num_cf_samples):
                    rng = np.random.default_rng(cf_sample)
                    for t in range(self.horizon):
                        # action_gumbels
                        gumbels = []
                        for ag in agents:
                            available_actions = self._get_available_actions(trajectory['states'][t], ag.id)
                            gumbels.append(_sample_gumbels(
                                ag.policy(trajectory['info_states'][t][ag.id]),             # probabilities
                                available_actions.index(trajectory['actions'][t][ag.id]),   # index of taken action
                                1,                                                          # number of samples          
                                rng                                                         # random number generator
                            ))
                        action_gumbels[cf_sample].append(gumbels)
                        # env_gumbels
                        gumbels = self._sample_cf_env_gumbels(trajectory['states'][t], trajectory['states'][t + 1], rng)
                        env_gumbels[cf_sample].append(gumbels)
        else:
            with np.errstate(divide='ignore'):
                num_rounds = self.horizon//self.num_turns
                for cf_sample in range(num_cf_samples):
                    rng = np.random.default_rng(cf_sample)
                    for round in range(num_rounds):
                        # action_gumbels
                        # find (a) the actions taken by the agents in this round (b) the information states under which these actions were taken
                        info_states = [None] * self.num_agents
                        actions = [None] * self.num_agents
                        for t in range(round * self.num_turns, (round + 1) * self.num_turns):
                            acting_agents = self._get_acting_agents(trajectory['states'][t], agents)
                            for ag in acting_agents:
                                info_states[ag.id] = trajectory['info_states'][t][ag.id]
                                actions[ag.id] = trajectory['actions'][t][ag.id]
                        # sample action_gumbels from the posterior distribution
                        gumbels = []
                        for ag in agents:
                            available_actions = self._get_available_actions(trajectory['states'][round * self.num_turns], ag.id)
                            gumbels.append(_sample_gumbels(
                                ag.policy(info_states[ag.id]),             # probabilities
                                available_actions.index(actions[ag.id]),   # taken action
                                1,                                         # number of samples          
                                rng                                        # random number generator           
                            ))
                        # action_gumbels are the same for all time-steps within the same round
                        for _ in range(round * self.num_turns, (round + 1) * self.num_turns):
                            action_gumbels[cf_sample].append(gumbels)
                        # env_gumbels
                        round_states = [trajectory['states'][t] for t in range(round * self.num_turns, (round + 1) * self.num_turns)]
                        next_round_first_state = trajectory['states'][(round + 1) * self.num_turns]
                        gumbels = self._sample_cf_env_gumbels(round_states, next_round_first_state, rng)
                        # env_gumbels are the same for all time-steps within the same round
                        for _ in range(round * self.num_turns, (round + 1) * self.num_turns):
                            env_gumbels[cf_sample].append(gumbels)

        return action_gumbels, env_gumbels

    # utils
    def _is_final_outcome_improved(self, old_outcome, new_outcome):
        """
        Input: old outcome and new outcome (outcome states)
        Output: improved(bool)
        - Checks if the final outcome is counterfactually improved or not
        """
        raise NotImplementedError

    def _get_counterfactual_change(self, old_outcome, new_outcome):
        """
        Input: old outcome and new outcome (states)
        Output: value of counterfactual change over the final outcome
        """
        raise NotImplementedError

    def _get_obs(self, state, env_gumbels):
        """
        INPUT: state, env_gumbels--may include obs_gumbels
        OUTPUT: agents' observations
        """
        raise NotImplementedError

    def _is_trajectory_failed(self, trajectory):
        """
        INPUT: trajectory
        OUTPOUT: failed(boolean)
        - Tests if trajectory's final outcome is undesirable
        """
        raise NotImplementedError

    def _get_available_actions(self, state, agent_id):
        """
        INPUT: state, agent_id
        OUTPUT: available agent's actions
        """
        raise NotImplementedError

    def _get_valid_actions(self, state, agent_id):
        """
        INPUT: state, agent_id
        OUTPUT: valid agent's actions
        """
        raise NotImplementedError

    def _get_default_actions(self, state, agents, env_gumbels, action_gumbels, last_info_states, last_actions):
        """
        INPUT: agents, state, env_gumbels(of time-step), action gumbels(of time-step), agents' last actions and last information states 
        OUTPUT: default actions and information states of agents
        """
        # agents' observations
        obs = self._get_obs(state, env_gumbels)
        # agents' information states
        info_states = [ag._get_info_state(obs[ag.id], last_actions[ag.id], last_info_states[ag.id]) for ag in agents]
        # determine agents who act in the current state
        acting_agents = self._get_acting_agents(state, agents)
        # a None action is taken by each agent who is not acting on this state
        default_actions = [None] * self.num_agents
        for agent in acting_agents:
            default_actions[agent.id] = agent._get_action(info_states[agent.id], action_gumbels[agent.id])

        return default_actions, info_states

    def _get_readable_form(self, action):
        """
        INPUT: action
        OUTPUT: action in readable form (if needed)
        """
        return action

    def _get_acting_agents(self, state, agents):
        """
        INPUT: state, agents
        OUTPUT: a list of the agents who act on this state
        """
        raise NotImplementedError

    def _get_num_remaining_actions(self, state):
        """
        INPUT: state
        OUTPUT: number of remaining actions from this state onwards
        """
        raise NotImplementedError

    def store_trajectory(self, f, trajectory):
        """
        INPUT: f(txt file), trajectory
        OUTPUT: -
        - Stores trajectory into file f in a readable way
        """
        raise NotImplementedError

    # gumbels
    def _get_action_gumbels(self, rng, num_available_actions):
        """
        INPUT: rng(random generator), number of available actions for each agent
        OUTPUT: action_gumbels
        """ 
        return [rng.gumbel(size=num_available_actions[ag]).tolist() for ag in range(self.num_agents)]

    def _get_env_gumbels(self, rng, state):
        """
        Environment specific
        """
        raise NotImplementedError
    
    def _sample_cf_env_gumbels(self, state, next_state):
        """
        Environment specific
        """
        raise NotImplementedError