from isolation_RL import Board
import game_agent_RL
import timeit
# from isolation_nn import *
import tensorflow as tf 


# create an isolation board (by default 7x7)
player1 = game_agent_RL.MinimaxPlayer()
player2 = game_agent_RL.MinimaxPlayer()
game = Board(player1, player2)
print("Player1: ", player1)
print("Player2: ", player2)
print(game.to_string)
# testing heuristics functions

time_millis = lambda: 1000*timeit.default_timer()
start = time_millis()
time_left = lambda: 150 - (time_millis() - start)
alphabetamove = player1.get_move(game, time_left)
game.apply_move(alphabetamove)
print(game.to_string())
start = time_millis()
time_left = lambda: 150 - (time_millis() - start)
minimax_move = player2.get_move(game, time_left)
game.apply_move(minimax_move)
print(game.to_string())
# evaluate will test every heuristics simultaneously
evaluation = gema_agent_RL.evaluate(game, player1)

print(alphabetamove)
print(minimax_move)




# game.play will not work


# place player 1 on the board at row 2, column 3, then place player 2 on
# the board at row 0, column 5; display the resulting board state.  Note
# that the .apply_move() method changes the calling object in-place.
game.apply_move((2, 3))
game.apply_move((1, 5))
print(game.to_string())

# players take turns moving on the board, so player1 should be next to move
assert(player1 == game.active_player)

# model = QNetwork(learning_rate=0.01, 
# 				num_functions=4,
# 				hidden_size=10,
# 				name="model")

# get a move from MinimaxPlayer
time_millis = lambda: 1000*timeit.default_timer()
start = time_millis()
time_left = lambda: 150 - (time_millis() - start)
minimax_move = game._active_player.get_move(game.copy(), time_left, model)
print(minimax_move)