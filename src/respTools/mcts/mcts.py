import time

from src.envs.causal_env import Causal_Env
from src.respTools.utils import compute_actual_causes, CH_responsibility
from src.respTools.mcts.mcts_node import *

def method_no_pruning_no_exploitation(env, agents, trajectory, action_gumbels, env_gumbels, rng, config: dict):
    """
    Implements main method without the pruning techniques and the exploitation term 
    """
    config['pruning'] = False
    config['exploitation'] = False

    return mcts(env, agents, trajectory, action_gumbels, env_gumbels, rng, config)

def method_no_exploitation(env, agents, trajectory, action_gumbels, env_gumbels, rng, config: dict):
    """
    Implements main method without the exploitation term 
    """
    config['exploitation'] = False

    return mcts(env, agents, trajectory, action_gumbels, env_gumbels, rng, config)

def mcts(env, agents, trajectory, action_gumbels, env_gumbels, rng, config: dict):
    """
    Implements the MCTS algorithm
    """
    kappa = config['kappa']
    tot_env_steps = config['tot_env_steps']
    frq_env_steps = config['frq_env_steps']
    pruning = config['pruning']
    exploitation = config['exploitation']
    c = config['c']
    q = config['q']
    w = config['w']
    
    start = time.time()

    responsibility = {}
    # initial attributed responsibility is zero for both agents
    responsibility[f'env_steps={0}'] = [0 for _ in agents]
    # execute search method
    algorithm = MCTS(env, agents, trajectory, action_gumbels, env_gumbels, kappa, rng, pruning, exploitation, c, q, w)
    for batch in range(1, tot_env_steps//frq_env_steps + 1):
        # search for candidate actual causes
        algorithm.search(frq_env_steps)
        # apply minimality condition
        actual_causes = compute_actual_causes(algorithm.candidate_actual_causes)
        # compute agents' responsibility
        responsibility[f'env_steps={batch * frq_env_steps}'] = []
        for agent in agents:
            degree = CH_responsibility(actual_causes, agent)
            responsibility[f'env_steps={batch * frq_env_steps}'].append(degree)
        # prepare algorithm for next batch
        algorithm.candidate_actual_causes = copy.deepcopy(actual_causes)
        algorithm.env_steps -= frq_env_steps

    end = time.time()

    output = {'responsibility': responsibility, 'tot_time': end - start}

    return output

class MCTS():
    # import class functions
    from src.respTools.mcts.initialize_unexplored_edges import initialize_unexplored_edges_root, initialize_unexplored_edges_timestep, initialize_unexplored_edges_agent, initialize_unexplored_edges_action
    from src.respTools.mcts.expand import expand_root, expand_timestep, expand_agent, expand_action

    def __init__(self, env: Causal_Env, agents, trajectory, action_gumbels, env_gumbels, kappa, rng, pruning, exploitation, c, q, w):

        self.env = env
        self.agents = agents
        self.trajectory = trajectory
        self.action_gumbels = action_gumbels
        self.env_gumbels = env_gumbels
        self.kappa = kappa
        self.rng = rng
        self.pruning = pruning
        self.exploitation = exploitation
        self.c = c
        self.q = q

        # list of found candidate actual causes
        self.candidate_actual_causes = []
        # list of found candidate actual causes in compressed form -- it contains only the action variables of the interventions and it is used for pruning
        self.cmprssd_candidate_actual_causes = []
        # list of linear scalarization functions -- used for child selection policy
        self.lst_lsf = []
        for agent in self.agents:
            # weights of agents' degrees of responsibility
            weights = [0] * len(self.agents)
            weights[agent.id] = 1 - w
            # weight of cf_change
            weights.append(w)
            self.lst_lsf.append(weights)
        # number of environment steps taken by the algorithm
        self.env_steps = 0
        # initialize root node
        initial_state = trajectory['states'][0]
        self.root_node = RootNode(initial_state, [None] * env.num_agents, [None] * env.num_agents)
        self.initialize_unexplored_edges(self.root_node)

        return

    def search(self, tot_env_steps):

        while self.env_steps < tot_env_steps:
            # break if tree has been fully explored
            if self.root_node.fully_explored: 
                break
            # Selection Phase
            selected_node = self.tree_policy()
            # print('selected node:', selected_node.info)
            self.lst_lsf.append(self.lst_lsf.pop(0))
            if selected_node.fully_explored:
                # node got pruned during the selection process
                continue
            # Simulation Phase
            score, leaf_node = self.simulate(selected_node)
            # Backpropagation Phase
            if leaf_node is None:
                # the selected node got pruned during the simulation process, and hence a leaf node was not reached
                continue
            else:
                # print("backpropagation score: ", score)
                leaf_node.backpropagate(score)
            # check if a candidate actual cause was found
            if score['outcome_improved']:
                # print("candidate actual cause was founded: ", leaf_node.intervention_variables)
                if self.pruning:
                    # prune closest AgentNode ancestor
                    leaf_node.parent.parent._set_to_fully_explored()
                # include in candidate actual causes
                self.candidate_actual_causes.append(leaf_node.interventions)
                # update list of candidate actual causes in compressed form (maintain only those that are minimal)
                self.cmprssd_candidate_actual_causes = [c for c in self.cmprssd_candidate_actual_causes if not set(c) > set(leaf_node.intervention_variables)]
                self.cmprssd_candidate_actual_causes.append(leaf_node.intervention_variables)

        return

    """
    MCTS functions
    """
    def tree_policy(self) -> MCTSNode:
        """
        Implements the Selection component of the algorithm
        """
        # print('start tree policy')
        current_node = self.root_node
        while current_node.type != 'leaf_node':
            # print(current_node.info)
            if not current_node._is_fully_expanded():
                # Expansion Phase
                return self.expand(current_node)
            # child selection
            current_node = current_node.best_child(self.c, self.q, self.exploitation, self.lst_lsf[0], self.rng)
            # check if current agent node has become non-minimal
            if self.pruning and current_node.type == 'agent_node' and self._is_not_minimal(current_node):
                # prune node
                # print("node became non minimal:", current_node.info, current_node.intervention_variables)
                # print("set of cadidate actual causes in compressed form: ", self.cmprssd_candidate_actual_causes)
                current_node._set_to_fully_explored()
                # correct for mistake
                correction_score = {}
                correction_score['responsibility'] = [-current_node.total_score['responsibility'][agent.id] for agent in self.agents]
                correction_score['cf_change'] = -current_node.total_score['cf_change']
                # print("correction score: ", correction_score)
                if self.exploitation:
                    current_node.backpropagate(correction_score, -current_node.num_visits)
                break

        return current_node

    def initialize_unexplored_edges(self, node):
        """
        Identifies all outgoing edges of a newly expanded node
        """
        if node.type == 'root_node':
            self.initialize_unexplored_edges_root()
        elif node.type == 'timestep_node':
            self.initialize_unexplored_edges_timestep(node)
        elif node.type == 'agent_node':
            self.initialize_unexplored_edges_agent(node)
        elif node.type == 'action_node':
            self.initialize_unexplored_edges_action(node)
        else:
            raise ValueError('Wrong node type')

        return

    def expand(self, node: MCTSNode):
        """
        Implements the Expansion component of the algorithm
        """
        assert not node._is_fully_expanded(), "Cannot expand a fully expanded node"
        assert not node.fully_explored, "Cannot expand a fully explored node"

        if node.type == 'root_node':
            child_node = self.expand_root(self.rng)
        elif node.type == 'timestep_node':
            child_node = self.expand_timestep(node, self.rng)
        elif node.type == 'agent_node':
            child_node = self.expand_agent(node, self.rng)            
        elif node.type == 'action_node':
            child_node = self.expand_action(node, self.rng)
        else:
            raise ValueError('Wrong node type')

        # include in node's children
        node.children.append(child_node)
        # prune leaf nodes and nodes without outgoing edges
        if child_node.type == 'leaf_node':
            child_node._set_to_fully_explored()
        else:
            self.initialize_unexplored_edges(child_node)
            if not child_node.unexplored_edges:
                child_node._set_to_fully_explored()
        # prune agent nodes that violate the minimality condition
        if self.pruning and child_node.type == 'agent_node' and self._is_not_minimal(child_node):
            child_node._set_to_fully_explored()
        
        return child_node

    def simulate(self, node: MCTSNode):
        """
        Implements the Simulation component of the algorithm
        - returns the rollout score and the leaf node
        - if a node is pruned during simulation, and hence a leaf node was not reached, Nones are returned instead
        """
        assert not node._is_fully_expanded(), "Cannot simulate a fully expanded node"

        # print('start simulation')

        # perform simulations until either the input node gets pruned or a leaf node is reached
        current_node = node
        while current_node.type != 'leaf_node' and not node.fully_explored:
            # print(current_node.info)
            if current_node.fully_explored:
                current_node = current_node.parent
            current_node = self.expand(current_node)
        
        if current_node.type == 'leaf_node':
            # evaluate leaf node
            rollout_score = self.rollout(current_node)

            return rollout_score, current_node 
        else:
            # a leaf node was not reached -- input node got pruned during simulation
            return None, None

    def rollout(self, node: LeafNode):
        """
        Performs a rollout on the given (leaf) node 
        - returns the score of the rollout
        - store results in parent (action) node for reusability purposes
        """
        env = self.env
        
        state = node.state
        last_info_states = node.last_info_states
        last_actions = node.last_actions

        for t in range(node.t, env.horizon):
            # store results in parent node for reusability
            node.parent.future_states.append(state)
            node.parent.future_last_info_states.append(last_info_states)
            node.parent.future_last_actions.append(last_actions)
            # gumbels
            action_gumbels = self.action_gumbels[t]
            env_gumbels = self.env_gumbels[t]
            # perform one full time-step
            info_states, actions, state = env.perform_time_step(self.agents, state, env_gumbels, action_gumbels, last_info_states, last_actions)
            self.env_steps += 1
            # prepare for next time-step
            last_info_states = info_states
            last_actions = actions

        outcome_improved = env._is_final_outcome_improved(self.trajectory['states'][-1], state)
        cf_change = env._get_counterfactual_change(self.trajectory['states'][-1], state)

        return self._get_score(node.interventions, outcome_improved, cf_change)

    """
    Utils
    """
    def _get_score(self, interventions, outcome_improved, cf_change):
        """
        Score includes:
        - the degrees of responsibility attributed by the CH definition to each of the agents for the given set of interventions
        - whether the final outcome is counterfactually improved when the interventions of the given set are applied
        - the  normalized value of counterfactual change in the final outcome
        """
        env = self.env

        score = {}
        score['outcome_improved'] = outcome_improved

        # compute agents' degrees of responsibility
        if not outcome_improved:
            # the given set of interventions does not constitute a candidate actual cause
            score['responsibility'] = [0] * len(self.agents)
        else:
            score['responsibility'] = []
            for agent in self.agents:
                degree = CH_responsibility([interventions], agent)
                score['responsibility'].append(degree)

        # normalize counterfactual change
        score['cf_change'] = cf_change / env.scaling_factor
        
        return score

    def _is_not_minimal(self, node: MCTSNode):
        """
        Tests if a node's set of interventions is minimal or not w.r.t. the candidate actual causes found so far in the search
        """
        
        return any(set(node.intervention_variables) > set(c) for c in self.cmprssd_candidate_actual_causes)
