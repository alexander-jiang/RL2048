from abstract_bot import AbstractBot
import argparse
import random

class RandomBot(AbstractBot):
    def __init__(self, dir_prob={"Up": 0.25, "Down": 0.25, "Left": 0.25, "Right": 0.25}, random_seed=None):
        assert "Up" in dir_prob and "Down" in dir_prob and "Left" in dir_prob and "Right" in dir_prob
        assert dir_prob["Up"] + dir_prob["Down"] + dir_prob["Left"] + dir_prob["Right"] == 1.0
        if random_seed is None:
            random_seed = random.getrandbits(16)

        self.dir_prob = dir_prob
        self.rand_gen = random.Random(random_seed)

    def next_action(self, game_state):
        dirs = game_state.moves_available()

        total = 0
        dir_list = []
        for dir in dirs:
            dir_list.append(dir)
            total += self.dir_prob[dir]

        rand_num = self.rand_gen.random()
        dir_idx = 0
        while dir_idx < len(dir_list):
            bucket_size = self.dir_prob[dir_list[dir_idx]] / total
            if rand_num < bucket_size:
                break
            rand_num -= bucket_size
            dir_idx += 1

        return dir_list[dir_idx]

def main():
    parser = argparse.ArgumentParser(description='Play N games using the random bot.')
    parser.add_argument('-n', '--num_games', type=int, default=1,
        help='number of games (default: 1)')
    parser.add_argument('-d', '--game_dir', type=str, default='random_bot_games',
        help='directory to save the games to (default: random_bot_games)')
    parser.add_argument('-m', '--move_prob', type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25],
        metavar=('PROB_UP', 'PROB_DOWN', 'PROB_LEFT', 'PROB_RIGHT'),
        help='probability of moving Up, Down, Left, or Right respectively (relative weights if not all directions are legal moves)')
    args = parser.parse_args()
    move_prob_dict = {"Up": args.move_prob[0], "Down": args.move_prob[1], "Left": args.move_prob[2], "Right": args.move_prob[3]}
    # print(move_prob_dict)
    for i in range(args.num_games):
        rand_bot = RandomBot(dir_prob=move_prob_dict)
        rand_bot.play_game(game_dir=args.game_dir)

if __name__ == "__main__": main()
