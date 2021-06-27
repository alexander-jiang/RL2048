
from collections import namedtuple
from typing import List
import numpy as np

BaseExperienceReplay = namedtuple("BaseExperienceReplay", ["state_tiles", "action", "successor_tiles", "reward"])

class ExperienceReplay(BaseExperienceReplay):
    """
    An experience replay tuple with helper functions to convert to and from 
    a flattened representation (e.g. for a neural net).
    """

    def __repr__(self):
        return (
            f"Previous state {tiles_repr(self.state_tiles)}\n"
            f"Action: {self.action}\n"
            f"New state {tiles_repr(self.successor_tiles)}\n"
            f"Reward: {self.reward}"
        )

    @property
    def state_bitarray(self) -> np.ndarray:
        return convert_tiles_to_bitarray(self.state_tiles)

    @property
    def successor_bitarray(self) -> np.ndarray:
        return convert_tiles_to_bitarray(self.successor_tiles)
    
    def flatten(self) -> np.ndarray:
        """stores the (s, a, s', r) in a flattened representation"""
        return np.hstack(
            (
                self.state_bitarray,
                np.array([self.action]),
                self.successor_bitarray,
                np.array([self.reward])
            )
        )

    @classmethod
    def from_flattened(cls, flat_tuple: np.ndarray):
        assert flat_tuple.shape == (2 * 16 * 17 + 2,)

        current_state_bitarray = flat_tuple[0:(16 * 17)]
        action = flat_tuple[16 * 17]
        new_state_bitarray = flat_tuple[(16 * 17 + 1):(2 * 16 * 17 + 1)]
        reward = flat_tuple[(2 * 16 * 17 + 1)]

        current_state_tiles = convert_bitarray_to_tiles(current_state_bitarray)
        new_state_tiles = convert_bitarray_to_tiles(new_state_bitarray)
        return cls(current_state_tiles, action, new_state_tiles, reward)
    

def convert_tiles_to_bitarray(tiles) -> np.ndarray:
    """
    Convert from a 4x4 array, where each cell is the log base 2 value of the tile,
    into a flattened bitarray representation, where each of the 16 cells is represented by 17 bits,
    with the first bit set if the tile value is 2, the second bit set in the tile value is 4,
    and so on up to 2^17 (the maximum possible tile value on a 4x4 board with 4-tiles being
    the maximum possible spawned tile).
    """
    flat_tiles = np.ravel(tiles)
    bitarray_input = np.zeros((16, 17))
    for i in range(16):
        if flat_tiles[i] != 0:
            # value of 1 (means the the tile is 2) should set bit 0 in bitarray
            bitarray_input_idx = flat_tiles[i] - 1
            bitarray_input[i,bitarray_input_idx] = 1
    return np.ravel(bitarray_input)

def convert_bitarray_to_tiles(bitarray: np.ndarray) -> list:
    """
    Convert from flattened bitarray representation, where each of the 16 cells is 
    represented by 17 bits (first bit is set if tile value is 2), to a 4x4 array, 
    where each cell is the log base 2 value of the tile.
    """
    assert bitarray.size == 16 * 17
    bitarray_reshape = np.reshape(bitarray, (4, 4, 17))
    tiles = []
    for r in range(4):
        tile_row = []
        for c in range(4):
            one_hot_tile_encoding = bitarray_reshape[r,c]
            if np.count_nonzero(one_hot_tile_encoding) == 0:
                tile_row.append(0)
            else:
                assert np.sum(one_hot_tile_encoding) == 1
                assert np.count_nonzero(one_hot_tile_encoding) == 1
                tile_row.append(np.argmax(one_hot_tile_encoding) + 1)
        tiles.append(tile_row)
    return tiles

# def parse_flattened_experience_tuple(flat_tuple: np.ndarray):
#     assert flat_tuple.shape == (2 * 16 * 17 + 2,)

#     current_state_bitarray = flat_tuple[0:(16 * 17)]
#     action = flat_tuple[16 * 17]
#     new_state_bitarray = flat_tuple[(16 * 17 + 1):(2 * 16 * 17 + 1)]
#     reward = flat_tuple[(2 * 16 * 17 + 1)]
#     return (current_state_bitarray, action, new_state_bitarray, reward)

def tiles_repr(tiles: List[List[int]]) -> str:
    output = ""
    for row_idx in range(len(tiles)):
        row = tiles[row_idx]
        for col_idx in range(len(row)):
            output += f"{row[col_idx]:2d}"
            if col_idx < len(row) - 1:
                output += " "
        if row_idx < len(tiles) - 1:
            output += "\n"
    return output