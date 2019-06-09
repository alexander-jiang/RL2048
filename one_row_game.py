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

    rand_gen = random.Random()
    while len(frontier) > 0:
        state = frontier.pop(0)
        if str(state.tiles) in visited:
            continue

        print(f"popped state: {state.tiles}")

        visited.append(str(state.tiles))
        # print("visited:")
        # for state_str in visited:
        #     print(state_str)

        successors = state.successor_states(only_two_tile=True)
        for successor in successors:
            if str(successor.tiles) in visited:
                continue
            print(f"possible next state: {successor.tiles}")
            frontier.append(successor)
    print(f"num visited: {len(visited)}")

if __name__ == "__main__": main()
