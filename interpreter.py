## Web crawler/parser that interacts with the web version

from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import random

def random_moves():
    # Initialize
    browser = webdriver.Firefox()
    browser.get('https://gabrielecirulli.github.io/2048/')
    body_elem = browser.find_element_by_css_selector('body')

    while True:
        resp_html = browser.page_source
        soup = BeautifulSoup(resp_html, 'html.parser')

        random_key = [Keys.UP, Keys.DOWN, Keys.LEFT, Keys.RIGHT]
        body_elem.send_keys(random_key[random.randrange(4)])

        game_over_screen = soup.find('div', class_='game-over')
        if game_over_screen is not None:
            break

    print 'final score:', soup.find('div', class_='score-container').get_text()
    browser.quit()

def main():
    # Initialize
    browser = webdriver.Firefox()
    browser.get('https://gabrielecirulli.github.io/2048/')
    body_elem = browser.find_element_by_css_selector('body')

    resp_html = browser.page_source
    soup = BeautifulSoup(resp_html, 'html.parser')

    html_tiles = soup.find_all('div', class_='tile')
    score_container = soup.find('div', class_='score-container')
    print html_tiles
    print score_container.get_text()

    print 'UP'
    body_elem.send_keys(Keys.UP)

    resp_html = browser.page_source
    soup = BeautifulSoup(resp_html, 'html.parser')
    html_tiles = soup.find_all('div', class_='tile')
    score_container = soup.find('div', class_='score-container')
    print html_tiles
    print score_container.get_text()



    # actions = ActionChains(browser)
    # new_game_button = browser.find_element_by_css_selector('a.restart-button')
    # actions.click(new_game_button)
    # actions.perform()
    browser.quit()

def read_tiles():
    """
    Returns the grid of tiles in the game.
    """
    resp_html = browser.page_source

    soup = BeautifulSoup(resp_html, 'html.parser')
    html_tiles = soup.find_all('div', class_='tile')
    if len(html_tiles) == 0:
        print('No tiles found!')

    # TODO finish parsing this
    return html_tiles

def new_game():
    """
    Starts a new game.
    """
    # TODO ask for user confirmation?
    actions = ActionChains(browser)
    new_game_button = browser.find_element_by_css_selector('a.restart-button')
    actions.click(new_game_button)
    actions.perform()
    # TODO add an assert?

def input_action(action):
    """
    Inputs the given action (1 = up, 2 = down, 3 = left, 4 = right).
    """
    assert action == 1 or action == 2 or action == 3 or action == 4
