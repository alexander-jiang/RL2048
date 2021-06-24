import numpy as np
import pytest

from reinforcement_learning.experience_replay_utils import (
    ExperienceReplay, 
    convert_tiles_to_bitarray, 
    convert_bitarray_to_tiles, 
    parse_flattened_experience_tuple,
)


def test_tiles_to_bitarray():
    num_rows = 4
    num_cols = 4
    bits_per_cell = 17
    tiles = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    out_array = convert_tiles_to_bitarray(tiles)
    assert out_array.shape == (num_rows * num_cols * bits_per_cell,)
    
    reshaped = np.reshape(out_array, (num_rows * num_cols, bits_per_cell))
    for i in range(num_rows * num_cols):
        one_hot_encoding = reshaped[i]
        assert np.sum(one_hot_encoding) == 1
        assert np.max(one_hot_encoding) == 1
        assert np.argmax(one_hot_encoding) == tiles[i // num_cols][i % num_cols] - 1

def test_bitarray_to_tiles():
    num_rows = 4
    num_cols = 4
    bits_per_cell = 17
    bitarray = np.zeros((num_rows * num_cols, bits_per_cell))
    for i in range(num_rows * num_cols):
        bitarray[i,i] = 1

    bitarray = np.ravel(bitarray)
    out_tiles = convert_bitarray_to_tiles(bitarray)
    assert len(out_tiles) == num_rows
    for r in range(len(out_tiles)):
        assert len(out_tiles[r]) == num_cols
        for c in range(len(out_tiles[r])):
            assert out_tiles[r][c] == r * num_cols + c + 1