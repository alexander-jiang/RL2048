from selenium.webdriver.common.keys import Keys

import interpreter
import random

def random_moves():
    game = interpreter.Interpreter2048()
    game.open()

    while True:
        keys = [Keys.UP, Keys.DOWN, Keys.LEFT, Keys.RIGHT]
        game.input_key(keys[random.randrange(4)])

        if game.is_game_over():
            break

    print 'final score:', game.current_score()
    game.close()

def main():
    game = interpreter.Interpreter2048()
    game.open()

    print 'tiles:', game.read_tiles()
    print 'score:', game.current_score()

    print 'input: UP'
    game.input_key(Keys.UP)

    print 'tiles:', game.read_tiles()
    print 'score:', game.current_score()

    game.close()
