class Causal_Actor:

    def __init__(self, id):
        """
        Necessary fields:
        - id
        """
        self.id = id

    def _get_info_state(self, obs, last_action, last_info_state):
        """
        INPUT: current observation, last action, last information state
        OUTPUT: current information state
        - Generates the player's information state
        """
        raise NotImplementedError

    def policy(self, info_state):
        """
        INPUT: current information state
        OUTPUT: action probability distribution (over all available actions)
        - Implements player's (stochastic) decision making policy
        """
        raise NotImplementedError

    def _get_action(self, info_state, action_gumbels):
        """
        INPUT: current information state, action_gumbels
        OUTPUT: selected action
        """
        raise NotImplementedError