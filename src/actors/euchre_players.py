import numpy as np
import copy

from src.actors.causal_actor import Causal_Actor

class Euchre_Player(Causal_Actor):

    def __init__(self, id, player_type):

        super().__init__(id)
        self.type = player_type

        return
    
    def __eq__(self, other): 

        if not isinstance(other, Euchre_Player):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.id == other.id and self.type == other.type
    
    def _get_info_state(self, obs, last_action, last_info_state):
        """
        Player's information state includes their hand, the player who is winning the current trick so far, the current turn,
        the list of cards in their hand that can win the current trick, the current leading suit and the trump suit
        """
        if not obs:
            return None

        info_state = {}
        info_state['trump_suit'] = obs['trump_suit']
        info_state['leading_suit'] = obs['leading_suit']
        info_state['hand'] = obs['hand']
        trick = obs['trick']
        info_state['turn'] = len(trick)
        winning_card = self._get_winning_card(info_state['trump_suit'], info_state['leading_suit'], trick)
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
            valid_actions = self._get_valid_actions(info_state['hand'], info_state['leading_suit'])
            info_state['winning_cards'] = []
            for card in valid_actions:
                if self._get_winning_card(info_state['trump_suit'], info_state['leading_suit'], [winning_card, card]) == card:
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
    
    def _get_valid_actions(self, hand, leading_suit):
        """
        Return Player's valid actions
        """
        # if there are cards of the leading suit available then remove all other cards
        if any(card.suit == leading_suit for card in hand):
            valid_actions = [card for card in hand if card.suit == leading_suit]
        else:
            valid_actions = copy.deepcopy(hand)

        return valid_actions

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
        color_jack = self._get_color_jack(trump_suit)
        if color_jack in trick:
            return color_jack
        # remove cards with suits that are not the leading suit or the trump suit
        trick = [card for card in trick if card.suit == leading_suit or card.suit == trump_suit]
        # if there is any card with the trump suit then remove cards with leading suit
        if any(card.suit == trump_suit for card in trick):
            trick = [card for card in trick if card.suit == trump_suit]
        # return the card with the highest value
        return max(trick, key=lambda card: card_value_mapping[card.value])    

    def _get_color_jack(self, trump_suit):

        if trump_suit == 'hearts':
            color_suit = 'diamonds'
        elif trump_suit == 'diamonds':
            color_suit = 'hearts'
        elif trump_suit == 'spades':
            color_suit = 'clubs'
        else:
            color_suit = 'spades'

        return Card('Jack', color_suit)

    def _get_strongest_card(self, cards, trump_suit):
        """
        Returns the strongest/highest ranked card from a set of cards
        """
        if cards is None:
            return None

        # check for the Jack of the trump suit
        trump_jack = Card('Jack', trump_suit)
        if trump_jack in cards:
            return trump_jack
        # check for the Jack with the suit color same as the color of the trump suit
        color_jack = self._get_color_jack(trump_suit)
        if color_jack in cards:
            return color_jack
        # if there is any card with the trump suit remove all cards with a different suit
        if any(card.suit == trump_suit for card in cards):
            cards = [card for card in cards if card.suit == trump_suit]
        # return the card with the highest value
        return max(cards, key=lambda card: card_value_mapping[card.value])

    def _get_weakest_card(self, cards, trump_suit):
        """
        Returns the weakest/lowest ranked card from a set of cards
        """
        if cards is None:
            return None

        # check for the Jack of the trump suit
        trump_jack = Card('Jack', trump_suit)
        if any(card != trump_jack for card in cards):
            cards = [card for card in cards if card != trump_jack]
        else:
            return trump_jack
        # check for the Jack with the suit color same as the color of the trump suit
        color_jack = self._get_color_jack(trump_suit)
        if any(card != color_jack for card in cards):
            cards = [card for card in cards if card != color_jack]
        else:
            return color_jack
        # if there is any card with a suit different than the trump suit remove all cards with the trump suit
        if any(card.suit != trump_suit for card in cards):
            cards = [card for card in cards if card.suit != trump_suit]
        # return the card with the highest value
        return min(cards, key=lambda card: card_value_mapping[card.value]) 


class Euchre_Agent(Euchre_Player):

    def __init__(self, id):

        super().__init__(id, 'agent')

        return

    def policy(self, info_state):
        """ 
        Each agent has a different deterministic policy
        """
        trump_suit = info_state['trump_suit']
        color_jack = self._get_color_jack(trump_suit)
        leading_suit = info_state['leading_suit']
        hand = info_state['hand']
        turn = info_state['turn']
        winning_player = info_state['winning_player']
        winning_cards = info_state['winning_cards']

        # first turn
        if turn == 0:
            # agent 0
            if self.id == 0:
                action = self._get_weakest_card(hand, trump_suit)
            # agent 1
            else:
                if any(card.suit != trump_suit and card != color_jack for card in hand):
                    action = self._get_strongest_card([card for card in hand if card.suit != trump_suit and card != color_jack], trump_suit)
                else:
                    action = self._get_strongest_card(hand, trump_suit)
        # second or third turn
        elif turn == 1 or turn == 2:
            if not winning_cards or winning_player.type == 'agent':
                action = self._get_weakest_card(self._get_valid_actions(hand, leading_suit), trump_suit)
            else:
                # agent 0
                if self.id == 0:
                    action = self._get_weakest_card(winning_cards, trump_suit)
                # agent 1
                else:
                    action = self._get_strongest_card(winning_cards, trump_suit)
        # last turn
        else:
            if not winning_cards or winning_player.type == 'agent':
                action = self._get_weakest_card(self._get_valid_actions(hand, leading_suit), trump_suit)
            else:
                action = self._get_weakest_card(winning_cards, trump_suit)
        
        probs = [0] * len(hand)
        probs[hand.index(action)] = 1

        return probs

class Euchre_Opponent(Euchre_Player):
    
    def __init__(self, id):
        
        super().__init__(id, 'opponent')

        return

    def policy(self, info_state):
        """
        All opponents have the same stochastic policy
        """
        trump_suit = info_state['trump_suit']
        leading_suit = info_state['leading_suit']
        hand = info_state['hand']
        turn = info_state['turn']
        winning_player = info_state['winning_player']
        winning_cards = info_state['winning_cards']

        probs = [0] * len(hand)
        # first turn
        if turn == 0:
            probs = [1/len(hand)] * len(hand)
        # second or third turn
        elif turn == 1 or turn == 2:
            valid_actions = self._get_valid_actions(hand, leading_suit)
            if not winning_cards:
                action = self._get_weakest_card(valid_actions, trump_suit)
                probs[hand.index(action)] = .8
            else:
                for action in winning_cards:
                    probs[hand.index(action)] = .8/len(winning_cards)
            for action in valid_actions:
                probs[hand.index(action)] += .2/len(valid_actions)
        # last turn
        else:
            if not winning_cards or winning_player.type == 'opponent':
                valid_actions = self._get_valid_actions(hand, leading_suit)
                action = self._get_weakest_card(valid_actions, trump_suit)
                probs[hand.index(action)] = .8
                for action in valid_actions:
                    probs[hand.index(action)] += .2/len(valid_actions)
            else:
                action = self._get_weakest_card(winning_cards, trump_suit)
                probs[hand.index(action)] = .8
                for action in winning_cards:
                    probs[hand.index(action)] += .2/len(winning_cards)

        return probs

class Euchre_Poisoned_Agent(Euchre_Agent):
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
        valid_actions = self._get_valid_actions(hand, leading_suit)
        if len(valid_actions) == 1:
            return probs
        
        poisoned_probs = [1/(len(valid_actions) - 1) * (1 - probs[c]) * (hand[c] in valid_actions) for c in range(len(hand))]

        return poisoned_probs

# global
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