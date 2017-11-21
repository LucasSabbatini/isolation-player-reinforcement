# Status: Not Ready

# Game-Playing Agent with Reinforcement Learning

This is an attempt to couple a reinforcement learning algorithm to learn the best weights of a linear combination of the heuristic functions. This heuristic is used by an modified alpha-beta agent. 
Since we need a function approximater that maps state features to evaluations, we're not going to use a simple linear combination (with one weight for each features). This code passes state feature into a neural network that with one hidden layer with X units. If there are Y state features (Y values calculated with Y different heuristics functions), and X units, then there'll be X*Y*Y weights in the network.

## The game - Isolation 7x7

Isolation is a strategy game in which the objective is to isolate the opponent, that being make your opponent run out of moves. This code runs a special version of game, players can only make L shape movements, like the knight in chess. 

## game_agent_RL.AlphaBetaPlayer

This agent is a modified version of an regular alpha-beta agent. Instead of returning only the value that the evaluation function returns on leaf nodes of the search tree, it also returns the evaluation vector (obtained by the game_agent_RL.evaluate method), which represents the state features (in a deep learning context) calculate by the heuristic functions. The root node will return  the tuple (chosen_move, value, state_features). Value and state_features are from the leaf node of the chosen path.

For each self move in the game, we store the information returned by the AlphaBetaPLayer.get_move() method into state_features_batch and values. when the game ends, if the winner is AlphaBetaPlayer, than a reward of -1.0 is appended to the rewards list, since the agent lost, and on the other case, a reward of +1.0 is appended. One might have noticed by now that we only store information about the player in question, and this is true, but we're storing the sequence of states and the evaluations of them, that let to a victory or to a loss.

## Neural Network

When a game is over, we end up with three lists: rewards, state_features, values. The rewards and values are used to calculate the loss, that will then update the weights of the network. If the model learns well, the ways will represent a good mapping of the state features to the evaluation of that state.



