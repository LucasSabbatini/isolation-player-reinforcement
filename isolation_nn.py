import numpy as np 
from isolation_RL import Board
from game_agent_RL import *
import tensorflow as tf 



class QNetwork:
	"""Neural network model

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

            # activation on logits
            self.sig = tf.sigmoid(self.logits)

            # loss
            self.loss = tf.matmul(self.rewards, -tf.log(self.logits))

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)



total_time_limit = 150.0
time_threshold = 10.0
total_rewards = []
num_functions = 5

def train(player1, 
        player2, 
        total_rewards, 
        state_features, 
        learning_rate=0.01,  
        name_model='model', 
        num_games=10,
        hidden_size=10):
	"""Trainf 

	"""

	state_features = []
    model = QNetwork(learning_rate, num_functions, hidden_size, name_model)
    board = Board(player1, player2)
    num_functions = player1.eval_functions

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for i in range(num_games):

        	board = Board(player1, player2)
            j=0
            state_features_batch = []
            rewards = []
            values = []

            # we make one batch in one game
            while True:

                j+=1
                time_left = lambda : time_limit - (time_millis() - move_start)

                if board._active_player == player2:
                    legal_player_moves = board.get_legal_moves()
                    game_copy = board.copy()    
                    move_start = time_millis()
                    curr_move = board._active_player.get_move(game_copy, time_left)         
                    move_end = time_left()

                    if curr_move is None:
                        curr_move = Board.NOT_MOVED

                    if curr_move not in legal_player_moves or move_end < 0:
                        # if player2 loses, append positive reward to rewards list,
                        # [0.0*len(num_functions)]
                        reward.append(1.0)
                        state_features_batch.append(np.zeros(len(num_functions)))
                        values.append(0.0)
                        break

                    move_history.append(list(curr_move))
                    board.apply_move(curr_move)

                else:

                    legal_player_moves = board.get_legal_moves()
                    board_copy = board.copy()
                    move_start = time_millis()
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

            assert len(rewards) == len(state_features_batch) and len(state_features_batch)

            state_features_batch = np.array(state_features_batch)
            gamma_discount = 0.7
            # Exponents arrays
            exponents_array = np.zeros([len(rewards), len(rewards)])
            for i in range(len(rewards)):
            	for j in range(len(rewards)):
            		if j-i <= 0:
            			exponents_array[i, j] = 0
            		else:
            			exponents_array[i, j] = gamma**(j-i)

            discounted_rewards = np.dot(exponents_array, np.array(rewards))

            _, _ = sess.run([model.loss, model.optimizer], feed_dict={model.inputs_: state_features_batch, 
            																		model.rewards:discounted_rewards}) 





model = QNetwork(learning_rate=0.01, 
                num_functions=4,
                hidden_size=10,
                name="model")

alphaPlayer = AlphaBetaPlayer()


with tf.Session() as sess:

	logit = sess.run(model.logit, feed_dict={model.inputs_:})






