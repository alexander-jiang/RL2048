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
    def __init__(self, game_state, master=None):
        super().__init__(master, bd=1, relief=tk.SOLID)
        self.master = master
        for i in range(4):
            self.rowconfigure(i, minsize=100)
            self.columnconfigure(i, minsize=100)
        self.game_state = game_state
        self.tiles = [
            [tk.Label(
                self, anchor=tk.CENTER,
                fg="#776e65", bg="#cdc1b4", font=('Helvetica', '15', 'bold'),
                bd=1, relief=tk.SOLID
            ),
            tk.Label(
                self, anchor=tk.CENTER,
                fg="#776e65", bg="#cdc1b4", font=('Helvetica', '15', 'bold'),
                bd=1, relief=tk.SOLID
            ),
            tk.Label(
                self, anchor=tk.CENTER,
                fg="#776e65", bg="#cdc1b4", font=('Helvetica', '15', 'bold'),
                bd=1, relief=tk.SOLID
            ),
            tk.Label(
                self, anchor=tk.CENTER,
                fg="#776e65", bg="#cdc1b4", font=('Helvetica', '15', 'bold'),
                bd=1, relief=tk.SOLID
            )] for i in range(4)
        ]
        for i in range(4):
            for j in range(4):
                self.tiles[i][j].grid(row=i, column=j, sticky=tk.N+tk.S+tk.W+tk.E)
        self.grid()
        self.draw_tiles()

    def draw_tiles(self):
        for i in range(4):
            for j in range(4):
                if self.game_state.tiles[i][j] > 0:
                    value = 1 << self.game_state.tiles[i][j]
                    text = f"{value}"
                else:
                    text = ""

                # different tile background/text color for different valued tiles
                if self.game_state.tiles[i][j] > 12:
                    bg_color = '#3c3a32'
                else:
                    bg_color = tile_bkgrd_color[self.game_state.tiles[i][j]]

                if self.game_state.tiles[i][j] > 2:
                    text_color = "#f9f6f2"
                else:
                    text_color = "#776e65"

                self.tiles[i][j]['text'] = text
                self.tiles[i][j]['bg'] = bg_color
                self.tiles[i][j]['fg'] = text_color

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
        self.game_tiles = GameTiles(self.game_state, master=self)
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
        self.game_tiles.draw_tiles()

    def draw_score(self):
        self.score_strvar.set(f"Score: {self.game_state.score}")

    def new_game(self):
        self.game_state.new_game()
        print(f"Started a new game! Saving to {self.game_state.game_filename}")
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
        # else:
            # print(f"moves available: {self.game_state.moves_available()}")

root = tk.Tk()
root.geometry("500x500")

app = GameWindow(master=root)
app.mainloop()
