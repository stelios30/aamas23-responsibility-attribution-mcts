import numpy as np
import copy

from src.actors.causal_actor import Causal_Actor

class Spades_Player(Causal_Actor):

    def __init__(self, id, player_type):

        super().__init__(id)
        self.type = player_type

        return
    
    def __eq__(self, other): 

        if not isinstance(other, Spades_Player):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.id == other.id and self.type == other.type
    
    def _get_info_state(self, obs, last_action, last_info_state):
        """
        Player's information state includes their hand, the player who is winning the current trick so far, the current turn,
        the list of cards in their hand that can win the current trick, the current leading suit and if the spades are broken so far or not
        """
        if not obs:
            return None

        info_state = {}
        info_state['leading_suit'] = obs['leading_suit']
        info_state['spades_broken'] = obs['spades_broken']
        info_state['hand'] = obs['hand']
        trick = obs['trick']
        info_state['turn'] = len(trick)
        winning_card = self._get_winning_card(info_state['leading_suit'], trick)
        # winning player
        if winning_card is None:
            # first turn
            info_state['winning_player'] = None
        else:    
            info_state['winning_player'] = obs['round_order'][trick.index(winning_card)]
        # winning cards in hand
        if winning_card is None:
            # first turn
            info_state['winning_cards'] = copy.deepcopy(info_state['hand'])
        else:
            valid_actions = self._get_valid_actions(info_state['turn'], info_state['hand'], info_state['leading_suit'], info_state['spades_broken'])
            info_state['winning_cards'] = []
            for card in valid_actions:
                if self._get_winning_card(info_state['leading_suit'], [winning_card, card]) == card:
                    info_state['winning_cards'].append(card)
        
        return info_state

    def _get_action(self, info_state, action_gumbels):
        
        if info_state is None:
            raise Warning("A None info_state was given.")

        hand = info_state['hand']
        probs = self.policy(info_state)
        with np.errstate(divide='ignore'):
            action = hand[np.argmax(np.log(probs) + action_gumbels)]
        
        return action
    
    def _get_valid_actions(self, turn, hand, leading_suit, spades_borken):
        """
        Return Player's valid actions
        """
        if turn == 0 and not spades_borken:
            # spades are not allowed to lead in case spades are not broken and cards of other suits are available 
            if any(card.suit != 'spades' for card in hand):
                valid_actions =  [card for card in hand if card.suit != 'spades']
            else:
                valid_actions = copy.deepcopy(hand)
        else:
            # if there are cards of the leading suit available then remove all other cards
            if any(card.suit == leading_suit for card in hand):
                valid_actions = [card for card in hand if card.suit == leading_suit]
            else:
                valid_actions = copy.deepcopy(hand)

        return valid_actions
    
    def _get_winning_card(self, leading_suit, trick):
        """
        Return card that wins the given trick
        """
        if not trick:
            return None

        # remove cards with suits that are not the leading suit or spades
        trick = [card for card in trick if card.suit == leading_suit or card.suit == 'spades']
        # if there is any card with spades suit then remove cards with leading suit
        if any(card.suit == 'spades' for card in trick):
            trick = [card for card in trick if card.suit == 'spades']
        # return the card with the highest value
        return max(trick, key=lambda card: card_value_mapping[card.value])
    
    def _get_strongest_card(self, cards):
        """
        Returns the strongest/highest ranked card from a set of cards
        """
        if cards is None:
            return None

        # if there is any card with spades suit remove all cards with a different suit
        if any(card.suit == 'spades' for card in cards):
            cards = [card for card in cards if card.suit == 'spades']
        # return the card with the highest value
        return max(cards, key=lambda card: card_value_mapping[card.value])

    def _get_weakest_card(self, cards):
        """
        Returns the weakest/lowest ranked card from a set of cards
        """
        if cards is None:
            return None

        # if there is any card with a suit different than the trump suit remove all cards with the trump suit
        if any(card.suit != 'spades' for card in cards):
            cards = [card for card in cards if card.suit != 'spades']
        # return the card with the highest value
        return min(cards, key=lambda card: card_value_mapping[card.value]) 

class Spades_Agent(Spades_Player):

    def __init__(self, id):

        super().__init__(id, 'agent')

        return

    def policy(self, info_state):
        """ 
        Each agent has a different deterministic policy
        """
        leading_suit = info_state['leading_suit']
        spades_broken = info_state['spades_broken']
        hand = info_state['hand']
        turn = info_state['turn']
        winning_player = info_state['winning_player']
        winning_cards = info_state['winning_cards']

        # first turn
        if turn == 0:
            # agent 0
            if self.id == 0:
                action = self._get_weakest_card(hand)
            # agent 1
            else:
                if any(card.suit != 'spades' for card in hand):
                    action = self._get_strongest_card([card for card in hand if card.suit != 'spades'])
                else:
                    action = self._get_strongest_card(hand)
        # second or third turn
        elif turn == 1 or turn == 2:
            if not winning_cards or winning_player.type == 'agent':
                action = self._get_weakest_card(self._get_valid_actions(turn, hand, leading_suit, spades_broken))
            else:
                # agent 0
                if self.id == 0:
                    action = self._get_weakest_card(winning_cards)
                # agent 1
                else:
                    action = self._get_strongest_card(winning_cards)
        # last turn
        else:
            if not winning_cards or winning_player.type == 'agent':
                action = self._get_weakest_card(self._get_valid_actions(turn, hand, leading_suit, spades_broken))
            else:
                action = self._get_weakest_card(winning_cards)
        
        probs = [0] * len(hand)
        probs[hand.index(action)] = 1

        return probs


class Spades_Opponent(Spades_Player):
    
    def __init__(self, id):
        
        super().__init__(id, 'opponent')
        
        return

    def policy(self, info_state):
        """
        All opponents have the same stochastic policy
        """
        leading_suit = info_state['leading_suit']
        spades_broken = info_state['spades_broken']
        hand = info_state['hand']
        turn = info_state['turn']
        winning_player = info_state['winning_player']
        winning_cards = info_state['winning_cards']

        probs = [0] * len(hand)
        # first turn
        if turn == 0:
            valid_actions = self._get_valid_actions(turn, hand, leading_suit, spades_broken)
            for action in valid_actions:
                probs[hand.index(action)] = 1/len(valid_actions)
        # second or third turn
        elif turn == 1 or turn == 2:
            valid_actions = self._get_valid_actions(turn, hand, leading_suit, spades_broken)
            if not winning_cards:
                action = self._get_weakest_card(valid_actions)
                probs[hand.index(action)] = .8
            else:
                for action in winning_cards:
                    probs[hand.index(action)] = .8/len(winning_cards)
            for action in valid_actions:
                probs[hand.index(action)] += .2/len(valid_actions)
        # last turn
        else:
            if not winning_cards or winning_player.type == 'opponent':
                valid_actions = self._get_valid_actions(turn, hand, leading_suit, spades_broken)
                action = self._get_weakest_card(valid_actions)
                probs[hand.index(action)] = .8
                for action in valid_actions:
                    probs[hand.index(action)] += .2/len(valid_actions)
            else:
                action = self._get_weakest_card(winning_cards)
                probs[hand.index(action)] = .8
                for action in winning_cards:
                    probs[hand.index(action)] += .2/len(winning_cards)

        return probs

class Spades_Poisoned_Agent(Spades_Agent):
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
        leading_suit = info_state['leading_suit']
        spades_broken = info_state['spades_broken']
        turn = info_state['turn']
        valid_actions = self._get_valid_actions(turn, hand, leading_suit, spades_broken)
        if len(valid_actions) == 1:
            return probs

        poisoned_probs = [1/(len(valid_actions) - 1) * (1 - probs[c]) * (hand[c] in valid_actions) for c in range(len(hand))]

        return poisoned_probs

# global
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