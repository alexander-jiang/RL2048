import random

NUM_ROWS = 4
NUM_COLS = 4

class GameState:
    def __init__(self, tiles=[[0] * 4 for i in range(4)], score=0, game_over=False, random_seed=None):
        assert len(tiles) == NUM_ROWS
        assert len(tiles[0]) == NUM_COLS
        assert score >= 0
        self.tiles = tiles
        self.score = score
        self.game_over = game_over
        if random_seed is None:
            self.random_seed = random.getrandbits(16)
        else:
            self.random_seed = random_seed
        random.seed(self.random_seed)

    def clear(self):
        self.tiles = [[0] * 4 for i in range(4)]
        self.score = 0
        self.game_over = False
        random.seed(self.random_seed)

    def move_tiles(self, dir):
        if self.game_over:
            print("Game is over! No more moves allowed")
            return

        # dir is either "Up", "Down", "Left", or "Right"
        assert dir in ["Up", "Down", "Left", "Right"]

        # contains coordinates from rows or columns, ordered based on direction (e.g. if
        # dir is Up, then the groups contains the columns ordered from top-to-bottom)
        groups = []
        if dir == "Up":
            for j in range(NUM_COLS):
                column = [(i, j) for i in range(NUM_ROWS)]
                groups.append(column)
        elif dir == "Down":
            for j in range(0, NUM_COLS):
                column = [(i, j) for i in reversed(range(0, NUM_ROWS))]
                groups.append(column)
        elif dir == "Left":
            for i in range(NUM_ROWS):
                row = [(i, j) for j in range(NUM_COLS)]
                groups.append(row)
        elif dir == "Right":
            for i in range(NUM_ROWS):
                row = [(i, j) for j in reversed(range(0, NUM_COLS))]
                groups.append(row)
        else:
            print(f"ERROR Invalid direction {dir}!")
            return

        moved_any = False
        for group in groups:
            last_tile_idx = None
            merged_idx = None
            # print(f"Group: {group}")
            for idx in range(len(group)):
                i, j = group[idx]
                if self.tiles[i][j] > 0:
                    # print(f"tile at {group[idx]}, last_tile_idx = {last_tile_idx}")
                    if last_tile_idx is None: # no tiles in front
                        if idx > 0:
                            new_i, new_j = group[0]
                            # print(f"moving tile to front: {group[0]}")
                            self.tiles[new_i][new_j] = self.tiles[i][j]
                            self.tiles[i][j] = 0
                            last_tile_idx = 0
                            moved_any = True
                        else:
                            last_tile_idx = idx
                    else:
                        assert last_tile_idx < idx
                        # check if we can merge
                        tile_i, tile_j = group[last_tile_idx]
                        if self.tiles[tile_i][tile_j] == self.tiles[i][j] and merged_idx != last_tile_idx:
                            # merge and update score
                            # print(f"merging tile on {group[last_tile_idx]}")
                            self.score += (1 << (self.tiles[i][j] + 1))
                            self.tiles[tile_i][tile_j] += 1
                            self.tiles[i][j] = 0
                            merged_idx = last_tile_idx
                            moved_any = True
                        elif last_tile_idx + 1 < idx:
                            # print(f"moving tile to {group[last_tile_idx + 1]}")
                            empty_i, empty_j = group[last_tile_idx + 1]
                            assert self.tiles[empty_i][empty_j] == 0
                            self.tiles[empty_i][empty_j] = self.tiles[i][j]
                            self.tiles[i][j] = 0
                            last_tile_idx += 1
                            moved_any = True
                        else:
                            last_tile_idx = idx

        if moved_any:
            self.spawn_tile()

            # if there are no valid moves left, then game is over
            if not self.moves_available():
                self.game_over = True

    def moves_available(self):
        # returns a list containing the valid movement directions from ["Up", "Down", "Left", "Right"]
        dirs = set([])
        up = False
        down = False
        for j in range(NUM_COLS):
            column = [(i, j) for i in range(NUM_ROWS)]
            prev_val = None
            for coord_idx in range(len(column)):
                i, j = column[coord_idx]
                # if adjacent tiles can be merged, then can merge up or down
                if self.tiles[i][j] == prev_val and self.tiles[i][j] > 0:
                    up = True
                    down = True
                elif prev_val is not None:
                    if self.tiles[i][j] == 0 and prev_val != 0:
                        down = True
                    if self.tiles[i][j] != 0 and prev_val == 0:
                        up = True
                prev_val = self.tiles[i][j]
                if up and down:
                    break
            if up:
                dirs.add("Up")
            if down:
                dirs.add("Down")
            if up and down:
                break

        left = False
        right = False
        for i in range(NUM_ROWS):
            row = [(i, j) for j in range(NUM_COLS)]
            prev_val = None
            for coord_idx in range(len(row)):
                i, j = row[coord_idx]
                # if adjacent tiles can be merged, then can merge left or right
                if self.tiles[i][j] == prev_val and self.tiles[i][j] > 0:
                    # print(f"can merge: tile at {row[coord_idx]}")
                    left = True
                    right = True
                elif prev_val is not None:
                    if self.tiles[i][j] == 0 and prev_val != 0:
                        # print(f"can shift right to empty tile at {row[coord_idx]}")
                        right = True
                    if self.tiles[i][j] != 0 and prev_val == 0:
                        # print(f"can shift left to empty tile at {row[coord_idx]}")
                        left = True
                prev_val = self.tiles[i][j]
                if left and right:
                    break
            if left:
                dirs.add("Left")
            if right:
                dirs.add("Right")
            if left and right:
                break
        return dirs

    def spawn_tile(self):
        empty_tile_locs = []
        for i in range(NUM_ROWS):
            for j in range(NUM_COLS):
                if self.tiles[i][j] == 0:
                    empty_tile_locs.append((i, j))
        idx = random.randrange(len(empty_tile_locs))
        sample = random.random()
        if sample < 0.9:
            new_tile = 1 # remember that the values in the tile array are log2
        else:
            new_tile = 2
        i, j = empty_tile_locs[idx]
        self.tiles[i][j] = new_tile
