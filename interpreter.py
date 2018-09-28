## Web crawler/parser that interacts with the web version
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

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
            return score
        else:
            return score[:score.index('+')]

    def is_game_over(self):
        """Determines whether the 2048 game has ended."""
        assert self.opened
        soup = self.get_html_parser()
        game_over_screen = soup.find('div', class_='game-over')
        return game_over_screen is not None

    def read_tiles():
        """Returns the grid of tiles in the game."""
        assert self.opened
        soup = self.get_html_parser()

        html_tiles = soup.find_all('div', class_='tile')
        if len(html_tiles) == 0:
            return []

        # TODO finish parsing this
        return html_tiles
