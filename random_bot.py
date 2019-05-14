from abstract_bot import AbstractBot
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
            # print(f"rand_num = {rand_num}")
            # print(f"dir_idx = {dir_idx}")
            bucket_size = self.dir_prob[dir_list[dir_idx]] / total
            if rand_num < bucket_size:
                break
            rand_num -= bucket_size
            dir_idx += 1

        return dir_list[dir_idx]

def main():
    for i in range(100):
        rand_bot = RandomBot()
        rand_bot.play_game()

if __name__ == "__main__": main()
