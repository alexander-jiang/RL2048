import random
from game_engine import GameState

def print_state(state):
    print(state.tiles)

def main():
    # BFS search to generate list of all possible states in the one-row game
    num_cols = 4
    frontier = []
    visited = []

    # initialize the possible starting states
    for i in range(num_cols):
        tiles = [[0] * num_cols]
        tiles[0][i] = 1
        frontier.append(GameState(nrows=1, ncols=num_cols, tiles=tiles, score=0, game_over=False))

    # perform BFS search to enumerate all states
    while len(frontier) > 0:
        state = frontier.pop(0)
        if state in visited:
            continue
        # print(f"popped state: {state.tiles}")

        visited.append(state)

        for move in state.moves_available():
            successors = state.successor_states(move, prob_two_tile=1.0)
            for probability, successor, reward in successors:
                if successor in visited:
                    continue
                # print(f"{state.tiles} -> {move} -> {successor.tiles} (prob {probability}, reward {reward})")
                frontier.append(successor)

    # print all states
    all_states = visited
    for state in all_states:
        print(state.tiles)
    print(f"num visited: {len(all_states)}")

    # value iteration on the MDP

    # initialize V and V_new to 0 for all states
    V = {}
    V_new = {}
    for state in all_states:
        state_str = str(state.tiles)
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
            state_str = str(state.tiles)
            action_vals = {}
            for move in state.moves_available():
                # print(f" {move}")
                successors = state.successor_states(move, prob_two_tile=1.0)
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




if __name__ == "__main__": main()
