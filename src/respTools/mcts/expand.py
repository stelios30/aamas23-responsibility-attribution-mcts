from src.respTools.mcts.mcts_node import *
from src.respTools.utils import _get_new_intervention

def expand_root(self, rng):

    env = self.env
    node = self.root_node
    # randomly expand an unexplored edge
    rng.shuffle(node.unexplored_edges)
    t = node.unexplored_edges.pop()

    state = self.trajectory['states'][t]
    if t == 0:
        last_info_states = [None] * env.num_agents
        last_actions = [None] * env.num_agents
    else:
        last_info_states = self.trajectory['info_states'][t-1]
        last_actions = self.trajectory['actions'][t-1]

    return TimeStepNode(node, t, state, last_info_states, last_actions)


def expand_timestep(self, node: TimeStepNode, rng):

    env = self.env
    action_gumbels = self.action_gumbels[node.t]
    env_gumbels = self.env_gumbels[node.t]
    # randomly expand an unexplored edge
    rng.shuffle(node.unexplored_edges)
    agent = node.unexplored_edges.pop()

    default_actions, info_states = env._get_default_actions(node.state, self.agents, env_gumbels, action_gumbels, node.last_info_states, node.last_actions)
    default_action = default_actions[agent.id]
    info_state = info_states[agent.id]

    return AgentNode(node, agent, default_action, info_state)

def expand_agent(self, node: AgentNode, rng):

    env = self.env
    
    # randomly expand an unexplored edge
    rng.shuffle(node.unexplored_edges)
    action = node.unexplored_edges.pop()

    new_intervention = _get_new_intervention(self.trajectory, node.t, node.agent, node.info_state, node.default_action, action)

    return ActionNode(node, action, env._get_readable_form(action), new_intervention)

def expand_action(self, node: ActionNode, rng):

    env = self.env

    if node.unexplored_edges[0] == 'leaf':
        # expand the edge leading to the leaf node
        node.unexplored_edges.pop(0)
        # gumbels
        action_gumbels = self.action_gumbels[node.t]
        env_gumbels = self.env_gumbels[node.t]
        # determine agents' actions
        default_actions, _ = env._get_default_actions(node.state, self.agents, env_gumbels, action_gumbels, node.last_info_states, node.last_actions)
        actions = default_actions
        timestep_interventions = [i for i in node.interventions if i['t'] == node.t]  # interventions that have been performed to action variables of this time-step
        for i in timestep_interventions:
            actions[i['agent']] = i['new_action']
        # perform one full time-step
        info_states, _, state = env.perform_time_step(self.agents, node.state, env_gumbels, action_gumbels, node.last_info_states, node.last_actions, predefined_actions=actions)
        self.env_steps += 1
        
        return LeafNode(node, node.t + 1, state, info_states, actions)
    else:
        # randomly expand an unexplored edge (leads to a time-step node)
        rng.shuffle(node.unexplored_edges)
        t = node.unexplored_edges.pop()
        if t == node.t:

            return TimeStepNode(node, t, node.state, node.last_info_states, node.last_actions)
        else:
            state = node.future_states[t - (node.t + 1)]
            last_info_states = node.future_last_info_states[t - (node.t + 1)]
            last_actions = node.future_last_actions[t - (node.t + 1)]
        
            return TimeStepNode(node, t, state, last_info_states, last_actions)
