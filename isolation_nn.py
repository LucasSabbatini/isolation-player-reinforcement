"""
This script contains two main structures:
QNetwork : class with neural network model
reinforcement_training : training function applying reinforcement learning to the game

"""


import numpy as np 
from isolation_RL import Board
from game_agent_RL import *
import tensorflow as tf 



class Network:
    """Neural network model

    This is a simple feed-forward network, with one hidden layer.

    Parameters
    ----------
    learning_rate : float
    num_functions : int, number or heuristics functions used in the evaluatino function
    hidden_size : int
    name :  string

    """
    def __init__(self, 
                 learning_rate=0.01, 
                 num_functions=4, 
                 hidden_size=10,
                 name="QNetwork"):

        graph = tf.Graph()
        with graph.as_default():

            # inputs will be the evaluation vactor, features extracted from the board
            self.inputs_ = tf.placeholder(tf.float32, [None, num_functions], name='inputs')
            # targets
            self.rewards = tf.placeholder(tf.float32, [None])

            # hidden layer
            self.fc = tf.layers.dense(self.inputs_, hidden_size, activation=tf.sigmoid)

            # output
            self.logits = tf.layers.dense(self.fc, 1)

            self.reshaped_logits = tf.reshape(self.logits, [-1])

            # activation on logits
            self.sig = tf.sigmoid(self.logits)

            # loss
            self.loss = tf.matmul(self.rewards, -tf.log(self.logits))

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)



total_time_limit = 150.0
time_threshold = 10.0
num_functions = 5
learning_rate = 0.01
hidden_size = 10

model = Network()

def reinforcement_training(model, 
        player1, 
        adversarial_players, 
        learning_rate=0.01, 
        num_games=10,
        hidden_size=10):
    """Perform the training of the model sent in by playing num_games games between
    player1 and each adversarial players.

	This function will iterate over adversarial_players and number of games, playing
	a full game (player1 against adversary) and storing the rewards found


    Parameters
    ----------
	model : neural network to be trained
	player1 : player object
		This player must return (move, val, state_features) instead of only just 
		the chosen move. These values will be used to train the model
	adversarial_players : list of player objects
		Every adversarial that will play against player1
	learning_rate : float
	num_games : int
		number of games player1 will play against each adversary
	hidden_size : number of nodes in hidden layer of model
	
	Returns
	-------
	board: isolation.Board object
	model : trained model


	Notes: 
	   -There's a clear distinction between player1 and other players, being that
		only player one will return the value and state_features from the leaf node
		of the path it has chosen, so the only data we can gather is with respect to
		player1.
	   -Since the rules of the game determine that a player wins when the other has no
		remaining moves left, the final state will be handled differently from the normal
		gameplay, and is the case when reward or discount

    """

    state_features = []
    total_rewards = []

    num_functions = player1.eval_functions

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for player2 in adversarial_players:
            for i in range(num_games):
            	# instantiate board with player2 and adversary
                board = Board(player1, player2)
                j=0
            	# data comming out this game
                state_features_batch = []
                rewards = []
                values = []

                time_left = lambda : time_limit - (time_millis() - move_start)
                # we make one batch in one game
                while True:

                    j+=1

                    # checking if player is adversary
                    if board._active_player == player2:
                        legal_player_moves = board.get_legal_moves()
                        game_copy = board.copy()    
                        move_start = time_millis()
                        curr_move = board._active_player.get_move(game_copy, time_left)         
                        move_end = time_left()

                        if curr_move is None:
                            curr_move = Board.NOT_MOVED

                        # this handles the case in which the player2 loses. When this happens,
                        # a reward of 1.0 is added to the rewards list, a np.zeros(num_functions) 
                        # array added to the states_features, and 0.0 to values 
                        if curr_move not in legal_player_moves or move_end < 0:
                            # if player2 loses, append positive reward to rewards list,
                            # [0.0*len(num_functions)]
                            reward.append(1.0)
                            state_features_batch.append(np.zeros(len(num_functions)))
                            values.append(0.0)
                            break

                        move_history.append(list(curr_move))
                        board.apply_move(curr_move)

                    else: # player is player1

                        legal_player_moves = board.get_legal_moves()
                        board_copy = board.copy()
                        move_start = time_millis()

                        # .get_move() returns chosen move, value returned by the linear combination
                        # heuristics for the leaf node of the chosen path, and a vector representing the
                        # state of the chosen leaf node
                        curr_move, val, sf = board._active_player.get_move(board.copy, time_left, model)
                        move_end = time_left()

                        if curr_move is None:
                            curr_move = Board.NOT_MOVED

                        if curr_move not in legal_player_moves or move_end < 0:
                            rewards.append(-1.0)
                            state_features_batch.append(np.zeros(len(num_functions)))
                            values.append(val)
                            break

                        values.append(val)
                        state_features_batch.append(sf)
                        rewards.append(0.0)

                        move_history.append(list(curr_move))
                        board.apply_move(curr_move)

                # assert if created lists are of the same size
                assert len(rewards) == len(state_features_batch) and len(state_features_batch)

                # adding batch features to the overall state_features
                state_features.append(state_features_batch)
                # transforming lis
                state_features_batch = np.array(state_features_batch)

                # coeficient that decreases future expected rewards
                gamma_discount = 0.7
                # Coeficient array containing the the values that will multiply the 
                coeficient_array = np.zeros([len(rewards), len(rewards)])
                for i in range(len(rewards)):
                    for j in range(len(rewards)):
                        if j-i <= 0:
                            coeficient_array[i, j] = 0
                        else:
                            coeficient_array[i, j] = gamma**(j-i)

                # multiplying the rewards vector by the coeficient matrix to obtain
                # the multipliers of the loss
                discounted_rewards = np.dot(coeficient_array, np.array(rewards))

                _, _ = sess.run([model.loss, model.optimizer], feed_dict={model.inputs_: state_features_batch, 
                                                                          model.rewards: discounted_rewards})

    return board, model



model = QNetwork(learning_rate=0.01, 
                num_functions=4,
                hidden_size=10,
                name="model")

