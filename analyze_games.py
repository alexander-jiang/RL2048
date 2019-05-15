import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def main():
    files = glob.glob("random_bot_games/*.csv")
    random_seeds = []
    scores = []
    game_lens = []
    for filepath in files:
        head, filename = os.path.split(filepath)
        root, ext = os.path.splitext(filename)
        if len(ext) > 0:
            date, random_seed = root.split('_', 1)
            if random_seed in random_seeds:
                print(f"Random seed {random_seed} already used!")
            random_seeds.append(random_seed)
        else:
            print(f"Unable to parse filename extension")

        with open(filepath, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            tokens = last_line.split(',')
            game_over = tokens[0]
            score = tokens[1]
            if not game_over:
                print(f"Game not over for {filepath}")
                continue
            scores.append(score)
            game_lens.append(len(lines))
    scores = np.array(scores, dtype=np.uint)
    game_lens = np.array(game_lens, dtype=np.uint)
    # print(scores)
    # print(f"length of scores = {len(scores)}")
    print(f"max in scores = {np.amax(scores)}")
    print(f"max score filename = {files[np.argmax(scores)]}")
    print(f"min in scores = {np.amin(scores)}")
    print(f"max score filename = {files[np.argmin(scores)]}")
    print(f"mean in scores = {np.mean(scores)}")
    print(f"median in scores = {np.median(scores)}")

    # print(f"length of game_lens = {len(game_lens)}")
    print(f"max in game_lens = {np.amax(game_lens)}")
    print(f"longest game filename = {files[np.argmax(game_lens)]}")
    print(f"min in game_lens = {np.amin(game_lens)}")
    print(f"shortest game filename = {files[np.argmin(game_lens)]}")
    print(f"mean in game_lens = {np.mean(game_lens)}")
    print(f"median in game_lens = {np.median(game_lens)}")

    fig, ax = plt.subplots()
    ax.hist(scores, bins='auto')
    ax.set_xlabel("Final Scores")
    ax.set_ylabel("Number of Games")
    ax.set_title("Random Bot: Final Scores Distribution")
    fig.savefig("figures/random_scores_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(game_lens, bins='auto')
    ax.set_xlabel("Game Lengths (# of Turns)")
    ax.set_ylabel("Number of Games")
    ax.set_title("Random Bot: Game Lengths Distribution")
    fig.savefig("figures/random_game_lens_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(game_lens, scores)
    ax.set_xlabel("Game Lengths (# of Turns)")
    ax.set_ylabel("Final Scores")
    ax.set_title("Random Bot: Score vs. Game Length")
    fig.savefig("figures/random_score_vs_game_len.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(random_seeds, scores)
    ax.set_xlabel("Random Seeds")
    ax.set_ylabel("Final Scores")
    ax.set_title("Random Bot: Score vs. Random Seeds")
    fig.savefig("figures/random_score_vs_random_seed.png")
    plt.close(fig)

if __name__ == "__main__": main()
