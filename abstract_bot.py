from game_engine import GameState

class AbstractBot():
    def __init__(self):
        pass

    def next_action(self, game_state):
        pass

    def play_game(self, game_dir, random_seed=None):
        game_state = GameState()
        game_state.new_game(random_seed=random_seed, game_dir=game_dir)
        while not game_state.game_over:
            dir = self.next_action(game_state)
            game_state.move_tiles(dir)
        print(f"final score: {game_state.score}")
