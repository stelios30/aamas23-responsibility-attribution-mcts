import numpy as np
import copy

from src.envs.causal_env import Causal_Env
from src.actors.euchre_players import Euchre_Opponent
from src.gumbelTools import _sample_gumbels

# global
card_values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
suits = ['spades', 'clubs', 'hearts', 'diamonds']
card_value_mapping = {
            '2' : 2,
            '3' : 3,
            '4' : 4,
            '5' : 5,
            '6' : 6,
            '7' : 7,
            '8' : 8,
            '9' : 9,
            '10' : 10,
            'Jack' : 11,
            'Queen' : 12,
            'King' : 13,
            'Ace' : 14,
        }

class Euchre(Causal_Env):

    def __init__(self, N, num_agents):
        """
        - N: number of cards dealt to each player/number of rounds
        - num_agents: number of agents
        - Opponents are part of the environment
        """
        super().__init__()
        self.N = N
        self.turn_based = True
        self.num_turns = 4
        self.horizon = self.num_turns * N
        self.scaling_factor = N / 2
        self.num_agents = num_agents
        self.num_opponents = num_agents
        self.card_players = []
        for id in range(2):
            # add opponents and agents in alternating order
            self.card_players.append(Card_Player('opponent', id))
            self.card_players.append(Card_Player('agent', id))
        self.opponents = [Euchre_Opponent(id) for id in range(self.num_opponents)]

        return

    def reset(self, rng):

        # shuffle deck
        deck = []
        for i in range(1, self.N + 2):
            for suit in suits:
               deck.append(Card(card_values[-i], suit))
        rng.shuffle(deck)
        # initial state
        state = {}
        state['trump_suit'] = rng.choice(suits)
        state['leading_suit'] = None
        state['score'] = {'agents' : 0, 'opponents' : 0}
        state['trick'] = []
        state['agents_hands'] = [[] for _ in range(self.num_agents)]
        state['opponents_hands'] = [[] for _ in range(self.num_opponents)]
        # deal cards
        for i in range(self.N):
            # w.l.g. deal always in the same pred-defined order
            for opp in range(self.num_opponents):
                state['opponents_hands'][opp].append(deck.pop(-1))
            for ag in range(self.num_agents):
                state['agents_hands'][ag].append(deck.pop(-1))
        # determine players' order in the first round
        initial_player = rng.choice(self.card_players)
        state['round_order'] = self._get_round_order(initial_player)
        # opponents' information states and actions
        state['opponents_last_info_states'] = [None] * self.num_opponents
        state['opponents_last_actions'] = [None] * self.num_opponents

        return state

    def step(self, state, actions, env_gumbels):

        # player whose turn is to play
        acting_card_player = state['round_order'][len(state['trick'])]
        # assert actions
        for ag in range(self.num_agents):
            assert actions[ag] in self._get_valid_actions(state, ag), f"Agent {ag} attempted a non-valid action -- {[self._get_readable_form(card) for card in self._get_valid_actions(state, ag)]}, {self._get_readable_form(actions[ag])}."

        # next state
        next_state = {}
        next_state['trump_suit'] = state['trump_suit']
        trick = copy.deepcopy(state['trick'])
        next_state['score'] = copy.deepcopy(state['score'])
        next_state['opponents_last_info_states'] = copy.deepcopy(state['opponents_last_info_states'])
        next_state['opponents_last_actions'] = copy.deepcopy(state['opponents_last_actions'])

        if acting_card_player.type == 'agent':
            action = actions[acting_card_player.id]
            next_state['agents_hands'] = [
                [card for card in state['agents_hands'][ag] if card != actions[ag]] 
                for ag in range(self.num_agents)
            ]
            next_state['opponents_hands'] = copy.deepcopy(state['opponents_hands'])
            trick.append(action)
            if len(trick) == 1:
                # first card played in the current round
                next_state['leading_suit'] = action.suit
        else:
            opponent = [opp for opp in self.opponents if opp.id == acting_card_player.id][0]
            # opponent's observation
            opponent_obs = {
                'trump_suit' : state['trump_suit'],
                'leading_suit' : state['leading_suit'],
                'score' : state['score'],
                'trick' : state['trick'],
                'round_order' : state['round_order'],
                'hand' : state['opponents_hands'][opponent.id]
            }
            # opponent's information state
            opponent_info_state = opponent._get_info_state(opponent_obs, state['opponents_last_info_states'][opponent.id], state['opponents_last_actions'][opponent.id])
            # opponent's action
            gumbels = env_gumbels['opponents_action_gumbels']
            opponent_action = opponent._get_action(opponent_info_state, gumbels[opponent.id])
            opponents_actions = [None] * self.num_opponents
            opponents_actions[opponent.id] = opponent_action
            next_state['agents_hands'] = copy.deepcopy(state['agents_hands'])
            next_state['opponents_hands'] = [
                [card for card in state['opponents_hands'][opp.id] if card != opponents_actions[opp.id]] 
                for opp in self.opponents
            ]
            trick.append(opponent_action)
            if len(trick) == 1:
                # first card played in the current round
                next_state['leading_suit'] = opponent_action.suit
            next_state['opponents_last_info_states'][opponent.id] = opponent_info_state
            next_state['opponents_last_actions'][opponent.id] = opponent_action
            
        if len(trick) < self.num_turns:
            # current round is not yet over, i.e., not every player has played a card
            next_state['trick'] = trick
            if len(trick) > 1:
                # not first card played in the current round
                next_state['leading_suit'] = state['leading_suit']
            next_state['round_order'] = copy.deepcopy(state['round_order'])
        else:
            # current round is over
            next_state['trick'] = []
            next_state['leading_suit'] = None
            winning_card = self._get_winning_card(state['trump_suit'], state['leading_suit'], trick)
            round_winner = state['round_order'][trick.index(winning_card)]
            if round_winner.type == 'agent':
                next_state['score']['agents'] += 1
            else:
                next_state['score']['opponents'] += 1
            next_state['round_order'] = self._get_round_order(round_winner)
        
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
        - Agents' observations include the trump and leading suits, the current score, the current trick, 
        the current players' order and the agent's hand 
        - Observations are deterministic, hence gumbels are not needed
        """
        if all(not hand for hand in state['agents_hands'] + state['opponents_hands']):
            # final state
            return [None] * self.num_agents
        obs = [{
            'trump_suit' : state['trump_suit'],
            'leading_suit' : state['leading_suit'],
            'score' : state['score'],
            'trick' : state['trick'],
            'round_order' : state['round_order'],
            'hand' : state['agents_hands'][ag]
        } for ag in range(self.num_agents)
        ]

        return obs

    def _is_trajectory_failed(self, trajectory):
        
        final_state = trajectory['states'][-1]

        return final_state['score']['agents'] <= final_state['score']['opponents']

    def _get_available_actions(self, state, agent_id):
       
        return state['agents_hands'][agent_id]
    
    def _get_valid_actions(self, state, agent_id):
       
        # if not agent's turn then it has no valid actions
        acting_card_player = state['round_order'][len(state['trick'])]
        if acting_card_player.type != 'agent' or acting_card_player.id != agent_id:
            valid_actions = [None]
        else:
            hand = state['agents_hands'][agent_id]
            # if there are cards of the leading suit available then remove all other cards
            if any(card.suit == state['leading_suit'] for card in hand):
                valid_actions = [card for card in hand if card.suit == state['leading_suit']]
            else:
                valid_actions = hand
        
        return valid_actions
    
    def _get_readable_form(self, action):

        if action is None:
            return None
        else:
            return (action.suit, action.value)
    
    def _get_acting_agents(self, state, agents):

        acting_card_player = state['round_order'][len(state['trick'])]
        if acting_card_player.type == 'opponent':
            return []

        return [ag for ag in agents if ag.id == acting_card_player.id]

    def _get_num_remaining_actions(self, state):

        return sum([len(hand) for hand in state['agents_hands']])

    def store_trajectory(self, f, trajectory):
       
        f.write(f"Trajectory: {trajectory['id']}\n\n")
        f.write(f"Trump suit: {trajectory['states'][0]['trump_suit']}\n")
        for t in range(self.horizon):
            if not t % self.num_turns:
                f.write("\n")
            f.write(f"Time-Step {t}\n")
            if not t % self.num_turns:
                # beginning of round
                f.write(f"Score: Agents {trajectory['states'][t]['score']['agents']}, Opponents {trajectory['states'][t]['score']['opponents']}\n")
                f.write(f"Agents' hands: {[[self._get_readable_form(card) for card in trajectory['states'][t]['agents_hands'][ag]] for ag in range(self.num_agents)]}\n")
                f.write(f"Opponents' hands: {[[self._get_readable_form(card) for card in trajectory['states'][t]['opponents_hands'][opp]] for opp in range(self.num_opponents)]}\n")
            acting_card_player = trajectory['states'][t]['round_order'][len(trajectory['states'][t]['trick'])]
            if acting_card_player.type == 'agent':
                f.write(f"Agent {acting_card_player.id}'s action: {self._get_readable_form(trajectory['actions'][t][acting_card_player.id])}\n")
            else:
                f.write(f"Opponent {acting_card_player.id}'s action: {self._get_readable_form(trajectory['states'][t+1]['opponents_last_actions'][acting_card_player.id])}\n")
        f.write(f"\nTime-Step {self.horizon}\n")
        f.write(f"Final Score: Agents {trajectory['states'][-1]['score']['agents']}, Opponents {trajectory['states'][-1]['score']['opponents']}\n\n\n")

        return

    ##########################################################

    def _get_round_order(self, initial_player):
        """
        Return the order in which the players will play in the current round
        """
        round_order = copy.deepcopy(self.card_players)
        while round_order[0] != initial_player:
            round_order.append(round_order.pop(0))

        return round_order

    def _get_winning_card(self, trump_suit, leading_suit, trick):
        """
        Return card that wins the given trick
        """
        if not trick:
            return None
        
        # check for the Jack of the trump suit
        trump_jack = Card('Jack', trump_suit)
        if trump_jack in trick:
            return trump_jack
        # check for the Jack with the suit color same as the color of the trump suit
        if trump_suit == 'hearts':
            color_suit = 'diamonds'
        elif trump_suit == 'diamonds':
            color_suit = 'hearts'
        elif trump_suit == 'spades':
            color_suit = 'clubs'
        else:
            color_suit = 'spades'
        color_jack = Card('Jack', color_suit)
        if color_jack in trick:
            return color_jack
        # remove cards with suits that are not the leading suit or the trump suit
        trick = [card for card in trick if card.suit == leading_suit or card.suit == trump_suit]
        # if there is any card with the trump suit then remove cards with leading suit
        if any(card.suit == trump_suit for card in trick):
            trick = [card for card in trick if card.suit == trump_suit]
        # return the card with the highest value
        return max(trick, key=lambda card: card_value_mapping[card.value])

    # gumbels    
    def _get_env_gumbels(self, rng, state):
        """
        - env_gumbels include the gumbels needed for determining the opponents' actions
        """
        env_gumbels = {}
        env_gumbels['opponents_action_gumbels'] = rng.gumbel(size=np.shape(state['opponents_hands'])).tolist()
        
        return env_gumbels

    def _sample_cf_env_gumbels(self, round_states, next_round_first_state, rng):
        """
        - round_states: all states visited during the current round
        - next_round_first_state: first state visited during the next round
        - rng: random number generator
        """        
        # find (a) the actions taken by the opponents in this round (b) the information states under which these actions were taken
        info_states = [None] * self.num_opponents
        actions = [None] * self.num_opponents
        for turn in range(len(round_states)):
            state = round_states[turn]
            if turn < len(round_states) - 1:
                next_state = round_states[turn + 1]
            else:
                next_state = next_round_first_state
            acting_card_player = state['round_order'][len(state['trick'])]
            if acting_card_player.type == 'opponent':
                info_states[acting_card_player.id] = next_state['opponents_last_info_states'][acting_card_player.id]
                actions[acting_card_player.id] = next_state['opponents_last_actions'][acting_card_player.id]
        # sample env_gumbels from the posterior distribution
        gumbels = []
        for opp in self.opponents:
            hand = round_states[0]['opponents_hands'][opp.id]
            gumbels.append(_sample_gumbels(
                opp.policy(info_states[opp.id]),    # probabilities
                hand.index(actions[opp.id]),        # taken action
                1,                                  # number of samples           
                rng,                                # random number generator
            ))
        
        env_gumbels = {}
        env_gumbels['opponents_action_gumbels'] = gumbels

        return env_gumbels

####################### Supporting Classes ############################
#######################################################################

class Card():

    def __init__(self, value, suit):

        assert suit in suits
        self.value= value
        self.suit = suit
        if suit == 'hearts' or suit == 'diamonds':
            self.color = 'red'
        else:
            self.color = 'back'

        return

    def __eq__(self, other): 

        if not isinstance(other, Card):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.value == other.value and self.suit == other.suit

class Card_Player():
    """
    It is used only to help keep track of the order in which players play in each round
    """
    def __init__(self, type, id):
        
        self.type = type
        self.id = id

        return

    def __eq__(self, other): 

        if not isinstance(other, Card_Player):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.id == other.id and self.type == other.type
