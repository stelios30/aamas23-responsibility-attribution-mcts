import numpy as np

from src.actors.causal_actor import Causal_Actor

class Team_Goofspiel_Player(Causal_Actor):

    def __init__(self, id, player_type):

        super().__init__(id)
        self.type = player_type

        return

    def __eq__(self, other): 

        if not isinstance(other, Team_Goofspiel_Player):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.id == other.id and self.type == other.type

    def _get_action(self, info_state, action_gumbels):
        
        if info_state is None:
            raise Warning("A None info_state was given.")

        hand = info_state['hand']
        probs = self.policy(info_state)
        with np.errstate(divide='ignore'):
            action = hand[np.argmax(np.log(probs) + action_gumbels)]
        
        return action

class Team_Goofspiel_Agent(Team_Goofspiel_Player):

    def __init__(self, id):

        super().__init__(id, 'agent')

        return

    def _get_info_state(self, obs, last_action, last_info_state):
        """
        Agent's information state includes their hand, the current prize and if they are winning or not
        """
        if not obs:
            return None

        info_state = {}
        info_state['hand'] = obs['hand']
        info_state['prize'] = obs['prize']
        score = obs['score']
        info_state['winning'] = bool(score['agents'] > score['opponents'])

        return info_state

    def policy(self, info_state):
        """
        Each agent has a different deterministic policy
        """
        hand = info_state['hand']
        prize = info_state['prize']
        winning = info_state['winning']

        probs = [0] * len(hand)
        # agent 0
        if self.id == 0:
            if prize in hand: action = prize
            else:
                gre = [x for x in hand if x > prize]
                le = [x for x in hand if x < prize]
                if (winning and le) or not gre: action = max(le)
                else: action = min(gre)
        # agent 1
        else:
            if prize > sum(hand)/len(hand) - winning: action = max(hand)
            else: action = min(hand)

        probs[hand.index(action)] = 1

        return probs
        
class Team_Goofspiel_Opponent(Team_Goofspiel_Player):
    
    def __init__(self, id):
        
        super().__init__(id, 'opponent')

        return
        
    def _get_info_state(self, obs, last_action, last_info_state):
        """
        Opponent's information state includes their hand, the current prize and if they are winning or not
        """
        if not obs:
            return None
        info_state = {}
        info_state['hand'] = obs['hand']
        info_state['prize'] = obs['prize']
        score = obs['score']
        info_state['winning'] = bool(score['agents'] < score['opponents'])

        return info_state

    def policy(self, info_state):
        """
        All opponents have the same stochastic policy
        - Cards in hand are assumed to be sorted
        """
        hand = info_state['hand']
        prize = info_state['prize']
        winning = info_state['winning']

        probs = [0] * len(hand)
        greq = [x for x in hand if x >= prize]
        leq = [x for x in hand if x <= prize]
        if (winning and leq) or not greq:
            probs = [1/len(leq)] * len(leq) + [0] * len([x for x in greq if x != prize])
        else:
            probs = [0] * len([x for x in leq if x != prize]) + [1/len(greq)] * len(greq)
        
        return probs

class Team_Goofspiel_Poisoned_Agent(Team_Goofspiel_Agent):
    """
    Poisoned_Agent is used in the partial ground truth experiments
    """
    def __init__(self, id, poisoned_time_steps):

        super().__init__(id)
        self.poisoned_time_steps = poisoned_time_steps
        
        return

    def _get_info_state(self, obs, last_action, last_info_state):
        """
        Additionally to an agent's information state, a poisoned agent's information state also includes the current time-step
        """
        if not obs:
            return None

        info_state = super()._get_info_state(obs, last_action, last_info_state)
        if last_info_state is None:
            info_state['t'] = 0
        else:
            info_state['t'] = last_info_state['t'] + 1
        
        return info_state

    def policy(self, info_state):
        """
        Poisoned policy
        """
        # action probabilities given by the non-poisoned policy
        probs = super().policy(info_state)
        if info_state['t'] not in self.poisoned_time_steps:
            return probs

        hand = info_state['hand']
        if len(hand) == 1:
            return probs

        poisoned_probs = [1/(len(hand) - 1) * (1 - probs[c]) for c in range(len(hand))]

        return poisoned_probs