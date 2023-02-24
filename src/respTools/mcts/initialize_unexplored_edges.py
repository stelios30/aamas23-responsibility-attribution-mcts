from src.respTools.mcts.mcts_node import *

def initialize_unexplored_edges_root(self):

    env = self.env
    node = self.root_node

    # explore all time-steps
    node.unexplored_edges = list(range(env.horizon))

    return

def initialize_unexplored_edges_timestep(self, node: TimeStepNode):

    env = self.env

    acting_agents = env._get_acting_agents(node.state, self.agents)
    # explore all combinations of agents for the current time-step at most once
    node.unexplored_edges = [agent for agent in acting_agents if all(agent.id > i['agent'] for i in node.interventions if i['t'] == node.t)]

    return

def initialize_unexplored_edges_agent(self, node: AgentNode):

    env = self.env
    default_action = node.default_action

    valid_actions = env._get_valid_actions(node.state, node.agent.id)
    # explore all alternative valid actions
    node.unexplored_edges = [action for action in valid_actions if action != default_action]

    return

def initialize_unexplored_edges_action(self, node: ActionNode):

    env = self.env

    # include leaf node -- no more interventions
    node.unexplored_edges.append('leaf')
    if len(node.interventions) < self.kappa:
        # include current time-step if it has more actions to be explored
        acting_agents = env._get_acting_agents(node.state, self.agents)
        if any(all(agent.id > i['agent'] for i in node.interventions if i['t'] == node.t) for agent in acting_agents):
            node.unexplored_edges.append(node.t)
        # include all future time-steps
        for t in range(node.t + 1, env.horizon):
            node.unexplored_edges.append(t)

    return
