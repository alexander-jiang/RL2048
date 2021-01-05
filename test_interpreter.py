from selenium.webdriver.common.keys import Keys

import interpreter
import random

def random_moves(print_every=20):
    """
    Plays a game of 2048 by choosing moves randomly.
    Input: print_every - how often to print the game state (# of moves made,
    score, and board state). Set to 0 to not print at all.
    """
    game = interpreter.Interpreter2048()
    game.open()
    move_count = 0

    while True:
        move_count += 1
        keys = [Keys.UP, Keys.DOWN, Keys.LEFT, Keys.RIGHT]
        game.input_key(keys[random.randrange(4)])

        if print_every != 0 and move_count % print_every == 0:
            print 'moves:', move_count
            print 'score:', game.current_score()
            print game.read_tiles()

        if game.is_game_over():
            break

    print 'total # of moves:', move_count
    print 'final score:', game.current_score()
    print game.read_tiles()
    game.close()

def one_move():
    game = interpreter.Interpreter2048()
    game.open()

    print game.read_tiles()
    print 'score:', game.current_score()

    print 'input: UP'
    game.input_key(Keys.UP)

    print game.read_tiles()
    print 'score:', game.current_score()

    game.close()
