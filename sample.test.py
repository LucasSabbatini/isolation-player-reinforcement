from isolation_RL import Board
from game_agent_RL import AlphaBetaPlayer
from game_agent_RL import MinimaxPlayer
import timeit
from isolation_nn import *



# create an isolation board (by default 7x7)
player1 = AlphaBetaPlayer()
player2 = GreedyPlayer()
game = Board(player1, player2)
print("Player1: ", player1)
print("Player2: ", player2)


# place player 1 on the board at row 2, column 3, then place player 2 on
# the board at row 0, column 5; display the resulting board state.  Note
# that the .apply_move() method changes the calling object in-place.
game.apply_move((2, 3))
game.apply_move((1, 5))
print(game.to_string())

# players take turns moving on the board, so player1 should be next to move
assert(player1 == game.active_player)

# get a move from MinimaxPlayer
time_millis = lambda: 1000*timeit.default_timer()
start = time_millis()
time_left = lambda: 150 - (time_millis() - start)
minimax_move, val, state_features = game._active_player.get_move(game.copy(), time_left)
print(minimax_move, val, state_features)