import argparse
import random
from game_engine import GameState
import networkx as nx
import matplotlib.pyplot as plt

# import the graphviz_layout function
# I went through a lot to get this to work:
# first, having graphviz 2.38 already installed to C:\Program Files (x86)\Graphviz2.38,
# I added the Graphviz2.38/bin folder to the PATH. Then installing pygraphviz was a pain, and
# I ended up using a user's installer, suggested by:
# https://github.com/pygraphviz/pygraphviz/issues/186#issuecomment-490319757
# and finally when I got an error "dot format not recognized", I found this:
# https://github.com/pygraphviz/pygraphviz/issues/97
# and replaced the dot.exe file (Graphviz version 2.41) in my Anaconda3/Scripts
# folder with the dot.exe file from my Graphviz 2.38 installation (the one on my
# PATH), which got the code below to work
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    raise ImportError("This example needs Graphviz and PyGraphviz "
        "(I tested on Windows with python 3.7.3 and pygraphviz 1.5 "
        "from alubbock, see comments in source code)")


def print_state(state):
    print(state.tiles)

def state_to_string(state):
    return str(state.tiles)


def main():
    parser = argparse.ArgumentParser(description='Build and solve an MDP that models an NxM game of 2048 (using value iteration).')
    parser.add_argument('-r', '--num_rows', type=int, default=2,
        help='number of rows (default: 2)')
    parser.add_argument('-c', '--num_cols', type=int, default=2,
        help='number of rows (default: 2)')
    parser.add_argument('-p', '--two_tile_prob', type=float, default=1.0,
        help='probability of spawning a 2-tile (instead of a 4-tile) after a successful move')
    args = parser.parse_args()

    num_rows = args.num_rows
    num_cols = args.num_cols
    frontier = []
    visited = []
    TWO_TILE_PROB = args.two_tile_prob
    output_filename = f"mdp_{num_rows}x{num_cols}_prob{TWO_TILE_PROB}.txt"

    # BFS search to generate list of all possible states in the one-row game
    states_graph = nx.DiGraph(num_rows=num_rows, num_cols=num_cols, two_tile_prob=TWO_TILE_PROB)

    # initialize the possible starting states
    for i in range(num_rows):
        for j in range(num_cols):
            tiles = [[0] * num_cols for r in range(num_rows)]
            tiles[i][j] = 1
            new_state = GameState(nrows=num_rows, ncols=num_cols, tiles=tiles, score=0, game_over=False)
            frontier.append(new_state)
            states_graph.add_node(state_to_string(new_state))

    # perform BFS search to enumerate all states
    while len(frontier) > 0:
        state = frontier.pop(0)
        if state in visited:
            continue
        # print(f"popped state: {state.tiles}")

        visited.append(state)
        states_graph.add_node(state_to_string(state))

        for move in state.moves_available():
            successors = state.successor_states(move, prob_two_tile=TWO_TILE_PROB)
            for probability, successor, reward in successors:
                if successor in visited:
                    continue
                # print(f"{state.tiles} -> {move} -> {successor.tiles} (prob {probability}, reward {reward})")
                frontier.append(successor)

                states_graph.add_edge(state_to_string(state), state_to_string(successor))

    # print all states
    all_states = visited
    for state in all_states:
        print(state.tiles)
    print(f"num states: {len(all_states)}")


    # value iteration on the MDP
    with open(output_filename, "w") as output:
        output.write(f"number of rows = {num_rows}\n")
        output.write(f"number of columns = {num_cols}\n")
        output.write(f"prob of 2-tile (vs. 4-tile) = {TWO_TILE_PROB}\n\n")

        output.write(f"number of states = {len(all_states)}\n")

        # initialize V and V_new to 0 for all states
        V = {}
        V_new = {}
        for state in all_states:
            state_str = state_to_string(state)
            if state_str not in V:
                V[state_str] = 0
            V_new[state_str] = 0

        converged = False
        iter_num = 1
        while not converged:
            print(f"========== value iteration (iter # {iter_num})")
            # update V_new using values in V
            for state in all_states:
                # print(state.tiles)
                state_str = state_to_string(state)
                action_vals = {}
                for move in state.moves_available():
                    # print(f" {move}")
                    successors = state.successor_states(move, prob_two_tile=TWO_TILE_PROB)
                    action_val = 0
                    for probability, successor, reward in successors:
                        # print(f" -> {successor.tiles} (prob: {probability}, reward: {reward})")
                        action_val += probability * (reward + V[str(successor.tiles)])
                    action_vals[move] = action_val
                    # print(f" {move} has value {action_val}")

                if not state.moves_available():
                    # print("found a terminal state")
                    continue

                # update V_new with the action with the highest value
                best_action = None
                best_action_val = float('-inf')
                for action in action_vals:
                    if action_vals[action] > best_action_val:
                        best_action = action
                        best_action_val = action_vals[action]

                if best_action_val > V[state_str]:
                    V_new[state_str] = best_action_val
                    print(f"V[{state_str}] updated from {V[state_str]} to {V_new[state_str]}")

            # convergence check: are any values in V and V_new "significantly different"?
            converged = True
            for state_str in V_new:
                diff = V_new[state_str] - V[state_str]
                if diff > 1e-5: # TODO arbitrary threshold...
                    converged = False
                    break

            if converged:
                print(f"Value iteration converged!")
                break

            # if not converged, then copy from V_new to V and proceed to next iteration
            for state_str in V_new:
                V[state_str] = V_new[state_str]

            iter_num += 1

        for state in all_states:
            state_str = state_to_string(state)
            output.write(f"state {state_str}: Value = {V[state_str]}\n")
            states_graph.nodes[state_to_string(state)]['value'] = V[state_str]

            action_total = 0
            for move in state.moves_available():
                successors = state.successor_states(move, prob_two_tile=TWO_TILE_PROB)
                action_val = 0
                for probability, successor, reward in successors:
                    # output.write(f"    -> {successor.tiles} (prob: {probability}, reward: {reward})\n")
                    action_val += probability * (reward + V[str(successor.tiles)])
                output.write(f"    {move} has value {action_val}\n")
                action_total += action_val

            if len(state.moves_available()) > 0:
                random_action_value = action_total / len(state.moves_available())
                if random_action_value < V[state_str]:
                    output.write(f"****Random move value = {random_action_value} (a loss of {V[state_str] - random_action_value})\n")

    # save graph of states to Graphviz dot format
    nx.drawing.nx_agraph.write_dot(states_graph, f"states_graph_dot_{num_rows}x{num_cols}_prob{TWO_TILE_PROB}")

    # draw the graph of states
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.axis('off')
    pos = graphviz_layout(states_graph, prog="dot") # "dot" is good for directed graphs

    # colormap nodes using values from value iteration (mark terminal states)
    terminal_states = [node for node in states_graph.nodes() if states_graph.out_degree(node) == 0]
    nonterminal_states = [node for node in states_graph.nodes() if states_graph.out_degree(node) > 0]
    nodes_terminal = nx.draw_networkx_nodes(states_graph, pos, nodelist=terminal_states,
        node_size=200, node_shape='s',
        node_color=[V[node] for node in terminal_states], cmap='viridis')
    nodes_nonterminal = nx.draw_networkx_nodes(states_graph, pos, nodelist=nonterminal_states,
        node_size=200, node_shape='o',
        node_color=[V[node] for node in nonterminal_states], cmap='viridis')
    fig.colorbar(nodes_nonterminal)

    nx.draw_networkx_edges(states_graph, pos)
    nx.draw_networkx_labels(states_graph, pos, font_size=8)
    # nx.draw(states_graph, pos, with_labels=True, node_size=200, font_size=10)
    fig.savefig(f"states_graph_{num_rows}x{num_cols}_prob{TWO_TILE_PROB}.png")
    plt.close(fig)

    ## TODO: graphing
    # color/label edges with rewards (i.e. where tiles are merged)
    # label edges differently for non-optimal actions

    ## TODO: MDPs/playing the game
    # topological sort of the nodes to optimize value iteration order
    # some sort of predecessor-generating function to avoid having to enumerate all states and then topological sort?
    # can this be applied to identify "phases" of the game (i.e. after getting to a big tile milestone e.g. the first 512-tile)



if __name__ == "__main__": main()
