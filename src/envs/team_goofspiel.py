import numpy as np

from src.envs.causal_env import Causal_Env
from src.actors.team_goofspiel_players import Team_Goofspiel_Opponent
from src.gumbelTools import _sample_gumbels

class Team_Goofspiel(Causal_Env):

    def __init__(self, N, num_agents):
        """
        - N: number of cards dealt to each player/number of cards in the deck
        - num_agents: number of agents
        - Opponents are part of the environment
        """
        super().__init__()
        self.N = N
        self.horizon = N
        self.scaling_factor = np.sum(range(N + 1)) / 2
        self.num_agents = num_agents
        self.num_opponents = num_agents
        self.opponents = [Team_Goofspiel_Opponent(id) for id in range(self.num_opponents)]
        self.turn_based = False

        return

    def reset(self, rng):

        # shuffle deck
        deck = list(range(1, self.N + 1))
        rng.shuffle(deck)
        # intial state
        state = {}
        state['deck'] = deck
        state['score'] = {'agents' : 0, 'opponents' : 0}
        state['agents_hands'] = [list(range(1, self.N + 1)) for _ in range(self.num_agents)]
        state['opponents_hands'] = [list(range(1, self.N + 1)) for _ in range(self.num_opponents)]
        state['opponents_last_info_states'] = [None] * self.num_opponents
        state['opponents_last_actions'] = [None] * self.num_opponents

        return state

    def step(self, state, actions, env_gumbels):

        for ag in range(self.num_agents):
            assert actions[ag] in self._get_available_actions(state, ag), "An agent attempted to play a card that does not exist in their hand."

        # opponents' observations
        opponents_obs = [{
            'hand' : state['opponents_hands'][opp.id], 
            'prize' : state['deck'][-1], 
            'score' : state['score']
        } for opp in self.opponents
        ]
        # opponents' information states
        opponents_info_states = [
                opp._get_info_state(opponents_obs[opp.id], state['opponents_last_info_states'][opp.id], state['opponents_last_actions'][opp.id])
                for opp in self.opponents
            ]
        # opponents' actions
        gumbels = env_gumbels['opponents_action_gumbels']
        opponents_actions = [opp._get_action(opponents_info_states[opp.id], gumbels[opp.id]) for opp in self.opponents]
        # next state
        next_state = {}
        next_state['deck'] = state['deck'][:-1]
        next_state['score'] = {
            'agents' : state['score']['agents'] + (sum(actions) > sum(opponents_actions)) * state['deck'][-1],
            'opponents' : state['score']['opponents'] + (sum(actions) < sum(opponents_actions)) * state['deck'][-1]
        }
        next_state['agents_hands'] = [
            [card for card in state['agents_hands'][ag] if card != actions[ag]] 
            for ag in range(self.num_agents)
        ]
        next_state['opponents_hands'] = [
            [card for card in state['opponents_hands'][opp.id] if card != opponents_actions[opp.id]] 
            for opp in self.opponents
        ]
        next_state['opponents_last_info_states'] = opponents_info_states
        next_state['opponents_last_actions'] = opponents_actions

        return next_state

    ####################### Utils ############################
    ##########################################################

    def _is_final_outcome_improved(self, old_outcome, new_outcome):
        
        old = old_outcome['score']['agents'] > old_outcome['score']['opponents']
        new = new_outcome['score']['agents'] > new_outcome['score']['opponents']

        return new > old

    def _get_counterfactual_change(self, old_outcome, new_outcome):

        old = old_outcome['score']['agents'] - old_outcome['score']['opponents']
        new = new_outcome['score']['agents'] - new_outcome['score']['opponents']

        return new - old

    def _get_obs(self, state, env_gumbels):
        """
        - Agents' observations include their hand, the current prize and the current score
        - Observations are deterministic, hence gumbels are not needed
        """
        if not state['deck']:
            return [None] * self.num_agents
        obs = [{
            'hand' : state['agents_hands'][ag], 
            'prize' : state['deck'][-1], 
            'score' : state['score']
        } for ag in range(self.num_agents)
        ]

        return obs
    
    def _is_trajectory_failed(self, trajectory):
        
        final_state = trajectory['states'][-1]

        return final_state['score']['agents'] <= final_state['score']['opponents']

    def _get_available_actions(self, state, agent_id):
       
        return state['agents_hands'][agent_id]
    
    def _get_valid_actions(self, state, agent_id):

        return self._get_available_actions(state, agent_id)

    def _get_acting_agents(self, state, agents):

        return agents

    def _get_num_remaining_actions(self, state):

        return sum([len(hand) for hand in state['agents_hands']])

    def store_trajectory(self, f, trajectory):
       
        f.write(f"Trajectory: {trajectory['id']}\n\n")
        for t in range(self.horizon):
            f.write(f"Time-Step {t}\n")
            f.write(f"Score: Agents {trajectory['states'][t]['score']['agents']}, Opponents {trajectory['states'][t]['score']['opponents']}\n")
            f.write(f"Prize: {trajectory['states'][t]['deck'][-1]}\n")
            f.write(f"Agents' hands: {trajectory['states'][t]['agents_hands']}\n")
            f.write(f"Opponents' hands: {trajectory['states'][t]['opponents_hands']}\n")
            f.write(f"Agents' actions: {trajectory['actions'][t]}\n")
            f.write(f"Opponents' actions: {trajectory['states'][t+1]['opponents_last_actions']}\n")
        f.write(f"Time-Step {self.horizon}\n")
        f.write(f"Final Score: Agents {trajectory['states'][-1]['score']['agents']}, Opponents {trajectory['states'][-1]['score']['opponents']}\n\n\n")

        return

    ##########################################################

    # gumbels    
    def _get_env_gumbels(self, rng, state):
        """
        - env_gumbels include the gumbels needed for determining the opponents' actions
        """
        env_gumbels = {}
        env_gumbels['opponents_action_gumbels'] = rng.gumbel(size=np.shape(state['opponents_hands'])).tolist()
        
        return env_gumbels

    def _sample_cf_env_gumbels(self, state, next_state, rng):
        
        env_gumbels = {}
        env_gumbels['opponents_action_gumbels'] = []
        # sample env_gumbels based from the posterior distribution
        for opp in self.opponents:
            hand = state['opponents_hands'][opp.id]
            action = next_state['opponents_last_actions'][opp.id]
            env_gumbels['opponents_action_gumbels'].append(_sample_gumbels(
                opp.policy(next_state['opponents_last_info_states'][opp.id]),     # probabilities
                hand.index(action),                                               # taken action
                1,                                                                # number of samples           
                rng,                                                              # random number generator
            ))
        
        return env_gumbels