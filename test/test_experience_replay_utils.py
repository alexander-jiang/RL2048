import numpy as np
import pytest

from reinforcement_learning.experience_replay_utils import (
    ExperienceReplay, 
    convert_tiles_to_bitarray, 
    convert_bitarray_to_tiles, 
    tiles_repr,
)


def test_tiles_to_bitarray():
    num_rows = 4
    num_cols = 4
    bits_per_cell = 17
    tiles = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    out_array = convert_tiles_to_bitarray(tiles)
    assert out_array.shape == (num_rows * num_cols * bits_per_cell,)
    
    reshaped = np.reshape(out_array, (num_rows * num_cols, bits_per_cell))
    for i in range(num_rows * num_cols):
        one_hot_encoding = reshaped[i]
        assert one_hot_encoding.shape == (17,)

        if np.sum(one_hot_encoding) == 1:
            assert np.max(one_hot_encoding) == 1
            assert np.count_nonzero(one_hot_encoding) == 1
            assert np.argmax(one_hot_encoding) + 1 == tiles[i // num_cols][i % num_cols]
        else:
            assert np.sum(one_hot_encoding) == 0
            assert np.count_nonzero(one_hot_encoding) == 0
            assert np.max(one_hot_encoding) == 0


def test_bitarray_to_tiles():
    num_rows = 4
    num_cols = 4
    bits_per_cell = 17
    bitarray = np.zeros((num_rows * num_cols, bits_per_cell))
    for i in range(num_rows * num_cols):
        if i > 0:
            bitarray[i,i-1] = 1

    bitarray = np.ravel(bitarray)
    out_tiles = convert_bitarray_to_tiles(bitarray)

    assert len(out_tiles) == num_rows
    for r in range(len(out_tiles)):
        assert len(out_tiles[r]) == num_cols
        for c in range(len(out_tiles[r])):
            assert out_tiles[r][c] == r * num_cols + c

def test_experience_replay_class():
    tiles = [
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    succ_tiles = [
        [2, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    action = 2
    reward = 4
    exp = ExperienceReplay(tiles, action, succ_tiles, reward)

    assert exp.state_tiles == tiles
    assert exp.action == action
    assert exp.successor_tiles == succ_tiles
    assert exp.reward == reward

    assert np.all(exp.state_bitarray == convert_tiles_to_bitarray(tiles))
    assert np.all(exp.successor_bitarray == convert_tiles_to_bitarray(succ_tiles))

    flattened = exp.flatten()
    from_flat_exp = ExperienceReplay.from_flattened(flattened)
    assert from_flat_exp.state_tiles == tiles
    assert from_flat_exp.action == action
    assert from_flat_exp.successor_tiles == succ_tiles
    assert from_flat_exp.reward == reward

def test_tiles_repr():
    tiles = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    expected = " 0  1  2  3\n 4  5  6  7\n 8  9 10 11\n12 13 14 15"
    output = tiles_repr(tiles)
    assert expected == output