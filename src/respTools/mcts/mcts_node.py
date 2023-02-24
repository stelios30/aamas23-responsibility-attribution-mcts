import numpy as np
import copy

class MCTSNode():

    def __init__(self, parent: 'MCTSNode', t, state, last_info_states, last_actions):
        
        # node information
        self.type = None
        self.parent = parent
        self.unexplored_edges = []
        self.children = []
        self.info = {}
        # trajectory information
        self.t = t
        self.state = state
        self.last_info_states = last_info_states
        self.last_actions = last_actions
        # interventions information
        self.interventions = []
        self.intervention_variables = []
        # mcts information
        self.num_visits = 0
        self.num_agents = len(last_actions)
        self.total_score = {}
        self.total_score['responsibility'] = [0] * self.num_agents
        self.total_score['cf_change'] = 0
        self.max_score = {}
        self.max_score['responsibility'] = [0] * self.num_agents
        self.max_score['cf_change'] = 0
        self.fully_explored = False

        return

    def _is_fully_expanded(self) -> bool:
        """
        Checks if a node is fully expanded, i.e., if it has any more unexplored edges
        """

        return len(self.unexplored_edges) == 0

    def best_child(self, c, q, exploitation, lsf, rng) -> 'MCTSNode':
        """
        Implements the child selection policy of the algorithm
        """
        assert self._is_fully_expanded(), "To perform best_child on a node, the node has to be fully expanded"
        
        # filter out child nodes that are fully explored
        available_children = [child for child in self.children if not child.fully_explored]
        assert self.children, "To perform best_child on a node, the node has to have at least one child that is not fully explored"

        # compute childrens' UCB1 values
        UCB1 = []
        for child in available_children:
            # number of parent node visits
            N = self.num_visits
            # number of current node visits
            n = child.num_visits
            # linear scalarized values
            ls_total_scores = [lsf[ag] * child.total_score['responsibility'][ag] for ag in range(self.num_agents)]
            ls_total_scores.append(lsf[-1] * child.total_score['cf_change'])
            ls_max_scores = [lsf[ag] * child.max_score['responsibility'][ag] for ag in range(self.num_agents)]
            ls_max_scores.append(lsf[-1] * child.max_score['cf_change'])
            # mix-max value
            V = q * sum(ls_max_scores) + (1 - q) * sum(ls_total_scores)/n
            # UCB1 value
            UCB1.append(exploitation * V + c * np.sqrt(np.log(N)/n))

        # print("start child selection policy")
        # print("children: ", [child.info for child in available_children])
        # print("UCB1 values: ", UCB1)
        
        # return randomly any child with the maximum UCB1 value
        return available_children[rng.choice([i for i in range(len(available_children)) if UCB1[i] == max(UCB1)])]

    def backpropagate(self, score, num_visits=1) -> None:
        """
        Implements the Backpropagation component of the algorithm
        """
        # update the node's total and max scores
        for ag in range(self.num_agents):
            self.total_score['responsibility'][ag] += score['responsibility'][ag]
            self.max_score['responsibility'][ag] = max(self.max_score['responsibility'][ag], score['responsibility'][ag])
        self.total_score['cf_change'] += score['cf_change']
        # max score was initialized to zero which is not necessarilly the min value for cf_change
        if self.num_visits == 0:
            self.max_score['cf_change'] = score['cf_change']
        else:
            self.max_score['cf_change'] = max(self.max_score['cf_change'], score['cf_change'])
        # update the node's number of visits
        self.num_visits += num_visits

        # in case a mistake correction is backpropagated the node's max scores needs some more steps
        if num_visits < 0:
            for ag in range(self.num_agents):
                if self.total_score['responsibility'][ag] == 0:
                    self.max_score['responsibility'][ag] = 0
                else:
                    self.max_score['responsibility'][ag] = max([child.max_score['responsibility'][ag] for child in self.children])
            if self.total_score['cf_change'] == 0:
                self.max_score['cf_change'] = 0
            else:
                self.max_score['cf_change'] = max([child.max_score['cf_change'] for child in self.children])
        
        # backpropagate to parent
        if self.parent is not None:
            self.parent.backpropagate(score, num_visits)

        return

    def _set_to_fully_explored(self) -> None:
        """
        Sets a node to fully explored, and also checks if its parent needs to be set as well
        """
        # print("node got pruned:", self.info)
        self.fully_explored = True

        if self.parent is None or not self.parent._is_fully_expanded():
            return
        if all(child.fully_explored for child in self.parent.children):
            self.parent._set_to_fully_explored()

        return

# Different Node Types
class RootNode(MCTSNode):

    def __init__(self, initial_state, last_info_states, last_actions):

        super().__init__(None, 0, initial_state, last_info_states, last_actions)
        self.type = 'root_node'
        self.info = {'type': 'root_node'}

        return

class TimeStepNode(MCTSNode):

    def __init__(self, parent: MCTSNode, t, state, last_info_states, last_actions):

        super().__init__(parent, t, state, last_info_states, last_actions)
        self.type = 'timestep_node'
        self.interventions = copy.deepcopy(parent.interventions)
        self.intervention_variables = copy.deepcopy(parent.intervention_variables)
        self.info = {'type': 'timestep_node', 't': t}

        return

class AgentNode(MCTSNode):

    def __init__(self, parent: TimeStepNode, agent, default_action, info_state):

        super().__init__(parent, parent.t, parent.state, parent.last_info_states, parent.last_actions)
        self.type = 'agent_node'
        self.agent = agent
        self.default_action = default_action
        self.info_state = info_state
        self.interventions = copy.deepcopy(parent.interventions)
        # update list of intervention variables
        self.intervention_variables = copy.deepcopy(parent.intervention_variables)
        self.intervention_variables.append((self.t, self.agent.id))
        self.info = {'type': 'agent_node', 'agent': agent.id}

        return

class ActionNode(MCTSNode):

    def __init__(self, parent: AgentNode, action, action_rf, new_intervention):

        super().__init__(parent, parent.t, parent.state, parent.last_info_states, parent.last_actions)
        self.type = 'action_node'
        self.action = action
        # update interventions
        self.interventions = copy.deepcopy(parent.interventions)
        self.interventions.append(new_intervention)
        self.intervention_variables = copy.deepcopy(parent.intervention_variables)
        # store the states (and additional info) that are reached in future time-steps from this node's state without performing any additional interventions, 
        # in order to not have to recompute them in every new expansion of the node -- they are all computed once during the expansion of the leaf node which is always expanded first
        self.future_states = []
        self.future_last_info_states = []
        self.future_last_actions = []
        self.info = {'type': 'action_node', 'action': action_rf}

        return

class LeafNode(MCTSNode):

    def __init__(self, parent: ActionNode, t, state, last_info_states, last_actions):

        super().__init__(parent, t, state, last_info_states, last_actions)
        self.type = 'leaf_node'
        self.interventions = copy.deepcopy(parent.interventions)
        self.intervention_variables = copy.deepcopy(parent.intervention_variables)
        self.info = {'type': 'leaf_node'}

        return

