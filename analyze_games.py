import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from game_engine import GameState

def main():
    parser = argparse.ArgumentParser(description='Play N games using the random bot.')
    parser.add_argument('game_dir', type=str,
        help='directory where games are saved to')
    parser.add_argument('figure_prefix', type=str,
        help='prefix of analysis figure filenames (saved to the figures directory)')
    args = parser.parse_args()

    files = glob.glob(f"{args.game_dir}/*.csv")
    random_seeds = []
    scores = []
    game_lens = []
    max_tile_values = []
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
            final_game_state = GameState.from_csv_line(lines[-1])
            if not final_game_state.game_over:
                print(f"Game not over for {filepath}")
                continue
            max_tile_value = final_game_state.max_tile_value()
            max_tile_values.append(max_tile_value)
            scores.append(final_game_state.score)
            game_lens.append(len(lines))
    scores = np.array(scores, dtype=np.uint)
    game_lens = np.array(game_lens, dtype=np.uint)
    max_tile_values = np.array(max_tile_values, dtype=np.uint)
    # print(scores)

    with open(f"{args.figure_prefix}_summary.txt", 'w') as f:
        # f.write(f"length of scores = {len(scores)}\n")
        f.write(f"max in scores = {np.amax(scores)}\n")
        f.write(f"max score filename = {files[np.argmax(scores)]}\n")
        f.write(f"min in scores = {np.amin(scores)}\n")
        f.write(f"min score filename = {files[np.argmin(scores)]}\n")
        f.write(f"mean in scores = {np.mean(scores)}\n")
        f.write(f"median in scores = {np.median(scores)}\n")

        # f.write(f"length of game_lens = {len(game_lens)}\n")
        f.write(f"max in game_lens = {np.amax(game_lens)}\n")
        f.write(f"longest game filename = {files[np.argmax(game_lens)]}\n")
        f.write(f"min in game_lens = {np.amin(game_lens)}\n")
        f.write(f"shortest game filename = {files[np.argmin(game_lens)]}\n")
        f.write(f"mean in game_lens = {np.mean(game_lens)}\n")
        f.write(f"median in game_lens = {np.median(game_lens)}\n")

    fig, ax = plt.subplots()
    ax.hist(scores, bins='auto')
    ax.set_xlabel("Final Scores")
    ax.set_ylabel("Number of Games")
    ax.set_title(f"{args.figure_prefix} Bot: Final Scores Distribution")
    fig.savefig(f"figures/{args.figure_prefix}_scores_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(game_lens, bins='auto')
    ax.set_xlabel("Game Lengths (# of Turns)")
    ax.set_ylabel("Number of Games")
    ax.set_title(f"{args.figure_prefix} Bot: Game Lengths Distribution")
    fig.savefig(f"figures/{args.figure_prefix}_game_lens_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    tile_values, tile_freqs = np.unique(max_tile_values, return_counts=True)
    ax.bar(range(len(tile_values)), tile_freqs)
    plt.xticks(range(len(tile_values)), tile_values)
    ax.set_xlabel("Final Max Tile Value")
    ax.set_ylabel("Number of Games")
    ax.set_title(f"{args.figure_prefix} Bot: Final Max Tile Distribution")
    fig.savefig(f"figures/{args.figure_prefix}_max_tile_values_hist.png")
    plt.close(fig)


    fig, ax = plt.subplots()
    ax.scatter(game_lens, scores)
    ax.set_xlabel("Game Lengths (# of Turns)")
    ax.set_ylabel("Final Scores")
    ax.set_title(f"{args.figure_prefix} Bot: Score vs. Game Length")
    fig.savefig(f"figures/{args.figure_prefix}_score_vs_game_len.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    max_tile_idxs = {}
    for i in range(len(max_tile_values)):
        tile_value = max_tile_values[i]
        if tile_value not in max_tile_idxs:
            max_tile_idxs[tile_value] = []
        max_tile_idxs[tile_value].append(i)

    data = [scores[max_tile_idxs[tile_value]] for tile_value in sorted(max_tile_idxs.keys())]
    # print(data)
    ax.boxplot(data, labels=sorted(max_tile_idxs.keys()))
    ax.set_xlabel("Final Max Tile Value")
    ax.set_ylabel("Final Scores")
    ax.set_title(f"{args.figure_prefix} Bot: Score vs. Final Max Tile Value")
    fig.savefig(f"figures/{args.figure_prefix}_score_vs_max_tile_value.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(random_seeds, scores)
    ax.set_xlabel("Random Seeds")
    ax.set_ylabel("Final Scores")
    ax.set_title(f"{args.figure_prefix} Bot: Score vs. Random Seeds")
    fig.savefig(f"figures/{args.figure_prefix}_score_vs_random_seed.png")
    plt.close(fig)

if __name__ == "__main__": main()
