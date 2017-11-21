# Game-Playing Agent with Reinforcement Learning

This is an attempt to couple a reinforcement learning algorithm to learn the best weights of a linear combination of the heuristic functions.


Hello Christine, I need some help with an implementation I'm trying to do for the game-playing project. For the heuristics, I've found on the internet the idea that instead of using one heuristic, one can use a linear combination of multiple heuristics,  and then finding good weights for them. To do so, I figure I could try to couple a neural network, with reinforcement learning to learn this weights, but it turned out to be harder than I thought and since I'm short on time, I'd like to know at least if my approach is feasible. Could you please take a look a the isolation_nn.py file on my repository (https://github.com/LucasSabbatini/isolation-player-reinforcement.git)? There you'll find the 