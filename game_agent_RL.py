"""
This script contains the following functions and classes:

Functions
---------
evaluate : returns a vector with state features, which are the individual values of each heuritic function
actionMobility : heuristic
my_moves_op : heuristic
my_moves_2_op : heuristic
distance from center : heuristic
actionFocus : heuristic
custom_score : actual evaluation function that will be called from the game-playing agent


Notes: 
- In the custom_score function there's a computation involving the neural network, and this
script is not importing tensorflow: CHECK IF NECESSARY
- The max_actions value is set to 8 because there are at most 8 actions a knigth can make
-

References:
    This scirpt was built on top of the game_agent.py file from Udacity's Game-Playing
Project, but the following functions were implemented by myself, with knowledge gathered
from classes, textbook and online posts and forums:
    evaluate
    actionMoility
    my_moves_op
    my_moves_2_op
    distance_from_ccenter
    action_focus
    custom_score
    AlphaBetaPlayer.get_move
    AlphaBetaPlayer.alphabeta
    AlphaBetaPlayer.max_alpha_beta
    AlphaBetaPlayer.min_alpha_beta
    MinimaxPlayer.get_move
    MinimaxPlayer.minimax
    MinimaxPlayer.min_value
    MinimaxPlayer.max_value

"""
import random
import numpy as np
# from isolation_nn import *

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def evaluate(game, player, max_actions=8):
    """
    Returns the cross product of weights and evaluation values
    """
    eval_vec = np.zeros((num_functions))
    eval_vec[0] = actionMobility(game, player, max_actions)
    eval_vec[1] = my_moves_op(game, player)
    eval_vec[2] = my_moves_2_op(game, player)
    eval_evc[3] = distance_from_center(game, player)
    eval_vec[4] = actionFocus(game, player, max_actions)
    return eval_vec

def actionMobility(game, player, max_actions=8):
    """
    Parameters
    ----------
    game : isolation_RL.Board
    player : player object
    max_acitons : int

    Reutrns
    -------
    number of possible moves/max_actions
    """
    return (len(game.get_legal_moves(player))*100.0/float(max_actions))

def my_moves_op(game, player):
    """
    Parameters
    ----------
    game : isolation_RL.Baord
    player : player object

    Returns
    -------
    #my_moves-#op_moves
    """
    return (len(game.get_legal_moves(player))-len(game.get_legal_moves(game.get_opponent(player))))

def my_moves_2_op(game, player):
    """
    Parameters
    ----------
    game : isolation_RL.Baord
    player : player object
    
    Returns
    -------
    #my_moves-2*#op_moves
    """
    return (len(game.get_legal_moves(player))-2*len(game.get_legal_moves(game.get_opponent(player))))

def distance_from_center(gamme, player):
    """
    Parameters
    ----------
    game : isolation_RL.Baord
    player : player object
    Returns 
    -------
    distance from center / max_dist
    """
    max_dist = np.sqrt(2*((game.height//2)**2))
    center = height//2
    current_position = game.get_player_location(player)
    distance = np.sqrt((abs(current_position[0]-center)**2)+(abs(current_possition[1]-center))**2)
    return distance * 100.0/max_dist

def actionFocus(game, player, max_actions=8):

    """

    """
    return 100.0-actionMobility(game, player, max_actions)



def custom_score(game, player, model):
    """
    Evaluate current state on player's view, using the evaluate function to obtain 
    state features and a neural network forward pass to get the associated value

    Parameters
    ----------
    game : isolation_RL.Board
    player : player that the state should be evaluated for
    model : isolation_nn.Network

    Returns
    -------
    valuation : tuple (value, game_features)
        value : The heuristic value of the current game state to the specified player.
        state_features : numpy array with features evluated with every heuristic function
    """
    
    state_features = evaluate(game, player)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        val = sess.run(model.logits, feed_dict={model.inputs_:state_features})

    return val, state_features


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.

    eval_functions : 
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10., eval_functions=5):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.eval_functions = eval_functions



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left, model):
        """
        Search for the best move given the evaluation method before the time limit expires

        Parameters
        ----------
        game : isolation_RL.Board
        time_left : function that evaluates remainder time for ply
        model : isolation_nn.Network

        Returns
        -------
        best_move : (int, int), chosen move
        val : float, value associated with leaf state of chosen path
        state_features : np.array with values of each heuristic evaluated on leaf state
        """

        self.time_left = time_left

        best_move = (-1, -1)

        depth = 1
        while self.time_left() > self.TIMER_THRESHOLD:
            try:
                best_move, val, state_features = self.alphabeta(game, depth, model)
                depth += 1

            except SearchTimeout:
                break

        return best_move, val, state_features

    def alphabeta(self, game, depth, model, alpha=float("-inf"), beta=float("inf")):
        """
        Performs the alphabeta search, calling min/max_alpha_beta recursively
        
        Parameters
        ----------
        game : isolation_RL.Board
        depth : int, maximum depth the game should search
        model : isolation_nn.Board
        alpha : float
        beta : flaot

        Returns
        -------
        best_move : (int, int), chosen move
        val : float, value associated with leaf state of chosen path
        state_features : np.array with values of each heuristic evaluated on leaf state
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        possible_actions = game.get_legal_moves()
        n_moves = len(possible_actions)
        values_for_actions = np.zeros(n_moves)

        # list for state_features of the leaves going through actions
        state_features_list = np.zeros(self.eval_functions)

        for i in range(n_moves):

            values_for_actions[i], state_features[i] = self.min_alpha_beta(game.forecast_move(possible_actions[i]), depth-1, alpha, beta)
            
            alpha = max(values_for_actions[i], alpha)

        best_index = np.argmax(values_for_actions)


        try: 
            return possible_actions[best_index], values_for_actions[best_index], state_features[best_index]
        except: 
            pass

    def min_alpha_beta(self, game, depth, alpha, beta, model):
        """
        Min player in the alpha beta search

        Parameter
        ---------
        game : isolation_RL.Board
        depth : int, remaining depth
        alpha : float
            Since this is a min level, alpha represents the minimum value that can be found
            in this branch, because it is the lower bound of the parent of this node (it 
            will not choose a node with a value lower than alpha)
        beta : float

        Returns
        -------
        best_move : (int, int), chosen move
        val : float, value associated with leaf state of chosen path
        state_features : np.array with values of each heuristic evaluated on leaf state
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout

        if depth == 0:
            return self.score(game, self, model)

        state_features = np.zeros(self.eval_functions)

        v = float("inf")
        for action in game.get_legal_moves():

            value, s = self.self.max_alpha_beta(game.forecast_move(action), depth-1, alpha, beta, model)
            
            if value < v:
                v = value
                state_features = s

            if v <= alpha: 
                return v

            beta = min(beta, v)
        return v, state_features

    def max_alpha_beta(self, game, depth, alpha, beta, model):
        """
        Max player in the alpha beta search

        Parameter
        ---------
        game : isolation_RL.Board
        depth : int, remaining depth
        alpha : float
        beta : int
            Since this is a max level, beta represents the maximum value that can be found
            in this branch, because it is the upper bound of the parent of this node (it
            will not choose a node with a value higher than beta).

        Returns
        -------
        v : int
            maximum evaluation found in its children
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self, model)

        state_features = np.zeros(self.eval_functions)

        v = float("-inf")
        for action in game.get_legal_moves():

            value, s = self.min_alpha_beta(game.forecast_move(action), depth-1, alpha, beta)
            
            if value > v:
                v = value
                state_features = s

            if v >= beta:
                return v

            alpha = max(alpha, v)

        return v, state_features



class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """
        Search for the best move given the evaluation method before the time limit expires

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.

        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        possible_actions = game.get_legal_moves()
        values_for_actions = np.zeros(len(possible_actions))
        for i in range(len(possible_actions)):
            values_for_actions[i] = self.min_value(game.forecast_move(possible_actions[i]), depth-1)
        try: 
            return possible_actions[np.argmax(values_for_actions)]
        except:

            print(type(possible_actions))
            print(possible_actions)  
            pass

    def max_value(self, game, depth):
        """
        Max player in the minimax method. Look for the following move
        that will maximize the expected evaluation

        Parameters
        ----------
        game : Board object
            Board objest representing a state of the game. It is a forecast state
        following the last min action in the search tree

        depth : int
            remaining steps to reach maximum depth specified

        Returns
        -------
        val : int
            Utility value for current state

        """
        # timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # checking if limit depth or terminal test
        if depth == 0:
            return self.score(game, self)
        v = float("-inf")
        for action in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(action), depth-1))
        return v

    def min_value(self, game, depth):
        """Min player in the minimax method. Look for the following move that will
        minimize the expected evaluation

        Parameters
        ----------
        game : Board object
            Board objest representing a state of the game. It is a forecast state
        following the last min action in the search tree

        depth : int
            remaining steps to reach maximum depth specified

        Returns
        -------
        val : int
            Mimimum expected value associated with possible actions
    
        """
        # timer chack
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # checking if limit depth or terminal test
        if depth == 0:
            return self.score(game, self)
        v = float("inf")
        for action in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(action), depth-1))
        return v


