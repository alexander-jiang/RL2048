import tkinter as tk
from game_engine import GameState

tile_bkgrd_color = {
    0: "#cdc1b4",
    1: "#eee4da",
    2: "#ede0c8",
    3: "#f2b179",
    4: "#f59563",
    5: "#f67c5f",
    6: "#f65e3b",
    7: "#edcf72",
    8: "#edcc61",
    9: "#edc850",
    10: "#edc53f",
    11: "#edc22e",
    12: "#3c3a32"
}

class GameTiles(tk.Frame):
    def __init__(self, tiles, master=None):
        super().__init__(master, bd=1, relief=tk.SOLID)
        self.master = master
        self.grid()
        self.draw_tiles(tiles)

    def draw_tiles(self, tiles):
        for i in range(4):
            self.rowconfigure(i, minsize=100)
            for j in range(4):
                if i == 0:
                    self.columnconfigure(j, minsize=100)

                if tiles[i][j] > 0:
                    value = 1 << tiles[i][j]
                    text = f"{value}"
                else:
                    text = ""

                # different tile background/text color for different valued tiles
                if tiles[i][j] > 12:
                    bg_color = '#3c3a32'
                else:
                    bg_color = tile_bkgrd_color[tiles[i][j]]

                if tiles[i][j] > 2:
                    text_color = "#f9f6f2"
                else:
                    text_color = "#776e65"

                tile = tk.Label(self, text=text, anchor=tk.CENTER,
                    fg=text_color, bg=bg_color, font=('Helvetica', '15', 'bold'),
                    bd=1, relief=tk.SOLID)
                tile.grid(row=i, column=j, sticky=tk.N+tk.S+tk.W+tk.E)

class GameWindow(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid()
        self.create_widgets()
        self.master.bind('<Up>', self.move)
        self.master.bind('<Down>', self.move)
        self.master.bind('<Left>', self.move)
        self.master.bind('<Right>', self.move)

    def create_widgets(self):
        self.game_state = GameState()
        self.game_tiles = GameTiles(self.game_state.tiles, master=self)
        self.draw_game_tiles()
        self.game_tiles.grid()

        self.score_strvar = tk.StringVar()
        self.game_score = tk.Label(self, textvariable=self.score_strvar)
        self.draw_score()
        self.game_score.grid()

        self.new_game = tk.Button(self, text="New Game", command=self.new_game)
        self.new_game.grid()

        self.quit = tk.Button(self, text="Quit", fg="red", command=self.master.destroy)
        self.quit.grid()

    def draw_game_tiles(self):
        self.game_tiles.draw_tiles(self.game_state.tiles)

    def draw_score(self):
        self.score_strvar.set(f"Score: {self.game_state.score}")

    def new_game(self):
        print(f"Started a new game!")
        self.game_state.clear()
        # start game with two random tiles
        self.game_state.spawn_tile()
        self.game_state.spawn_tile()
        self.draw_game_tiles()
        self.draw_score()

    def move(self, event):
        print(f"moving in dir {event.keysym}")
        self.game_state.move_tiles(event.keysym)
        self.draw_game_tiles()
        self.draw_score()
        if self.game_state.game_over:
            # TODO display a message on GUI
            print("Game over! no moves available")
        else:
            print(f"moves available: {self.game_state.moves_available()}")

root = tk.Tk()
root.geometry("500x500")

app = GameWindow(master=root)
app.mainloop()
