
There are 3 components to AlphaZero:

1. 𝛼MCTS 

2. Policy and Value Networks

3. Self-Play 

𝛼MCTS ( AlphaZero's Variant of Monte Carlo Tree Search) is mixture of the vanilla MCTS guided by two neural networks: policy network that suggests actions;value network that evaluates(scores) state. 
MCTS is a method to tranverse the game tree (tree of states and actions) to find the next best action. From a given state MCTS measures all the possibles action through a score, this score a variant of UCB
UCB (Upper Confidence Bound) that balances exploration and exploitation. 
The variant of UCB used in AlphaZero is called PUCT (Predictor + UCB for Trees) that uses prior probabilities from the policy network to guide exploration.
For game trees with large branching factors, PUCT is more efficient than standard UCB because it focuses on actions suggested by the policy network.



