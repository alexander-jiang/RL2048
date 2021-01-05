## Web crawler/parser that interacts with the web version
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import numpy as np

class Interpreter2048:
    def __init__(self):
        self.browser = webdriver.Firefox()
        self.opened = False

    def open(self):
        """Opens the browser window."""
        assert not self.opened
        self.browser.get('https://gabrielecirulli.github.io/2048/')
        self.body_elem = self.browser.find_element_by_css_selector('body')
        self.opened = True

    def close(self):
        """Closes the browser window."""
        assert self.opened
        self.browser.quit()
        self.opened = False

    def new_game(self):
        """Starts a new game of 2048."""
        assert self.opened
        actions = ActionChains(self.browser)
        new_game_button = self.browser.find_element_by_css_selector('a.restart-button')
        actions.click(new_game_button)
        actions.perform()

    def input_key(self, key):
        """Inputs one of the four directions to the game."""
        assert self.opened
        assert key in [Keys.UP, Keys.DOWN, Keys.LEFT, Keys.RIGHT]
        self.body_elem.send_keys(key)


    def get_html_parser(self):
        """Returns a BeautifulSoup instance of a HTML parser."""
        resp_html = self.browser.page_source
        return BeautifulSoup(resp_html, 'html.parser')

    def current_score(self):
        """Returns the current score from the game."""
        assert self.opened
        soup = self.get_html_parser()
        # Score may be followed by "+N" where N is the amount that the
        # score was last incremented by. Filter that out below:
        score = soup.find('div', class_='score-container').get_text()
        if score.find('+') == -1:
            return int(score)
        else:
            return int(score[:score.index('+')])

    def is_game_over(self):
        """Determines whether the 2048 game has ended."""
        assert self.opened
        soup = self.get_html_parser()
        game_over_screen = soup.find('div', class_='game-over')
        return game_over_screen is not None

    def read_tiles(self):
        """
        Returns the grid of tiles in the game into a 2D array (4x4), where
        each tile is replaced by its value log-2.
        """
        assert self.opened
        soup = self.get_html_parser()

        html_tiles = soup.find_all('div', class_='tile')
        tiles = np.zeros((4, 4))
        for html_tile in html_tiles:
            tile_value = np.log2(int(html_tile.get_text()))
            tile_pos_class = html_tile['class'][2]

            # Usually the tile-position-X-Y class is third in the list of CSS
            # classes, but in case it isn't:
            if not tile_pos_class.startswith('tile-position-'):
                for tile_class in html_tile['class']:
                    if tile_class.startswith('tile-position-'):
                        tile_pos_class = tile_class
                        break

            # The X-Y coords in the CSS class are in column-major order and are 1-indexed
            tile_pos = (int(tile_pos_class[-1:]) - 1, int(tile_pos_class[-3:-2]) - 1)

            # We should only update with a larger value (i.e. the merged value)
            if tiles[tile_pos] < tile_value:
                tiles[tile_pos] = tile_value

        return tiles
