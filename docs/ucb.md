### Deeper Dive into UCB in MCTS

In Monte Carlo Tree Search (MCTS), the **Upper Confidence Bound (UCB)** is the core mechanism for the **selection phase**, where the algorithm traverses the tree from root to leaf by greedily choosing the "most promising" child node at each step. It resolves the **exploration-exploitation dilemma**: 
- **Exploitation**: Favor actions with historically high rewards (don't fixate on one path forever).
- **Exploration**: Occasionally try less-visited actions to discover potentially better ones (avoid missing hidden gems).

UCB provides an **optimistic estimate** of a node's value, upper-bounding the true expected reward with high confidence. This encourages visiting under-explored nodes without pure randomness. Below, I'll explain the standard UCB (vanilla MCTS) in detail, then the enhanced **PUCT** variant (used in AlphaZero-style MCTS with priors).

#### 1. Standard UCB Formula and Breakdown

The formula for selecting the best child \( i \) from a parent node is:

\[
UCB_i = X_i + C \sqrt{\frac{\ln N}{n_i}}
\]

Where:
- \( X_i = \frac{w_i}{n_i} \): **Exploitation term** (empirical mean reward).
  - \( w_i \): Cumulative reward from all simulations ending in the subtree rooted at child \( i \) (e.g., sum of +1 for wins, -1 for losses).
  - \( n_i \): Number of times child \( i \) has been visited (simulations through it).
  - This is the "average value" estimate. High \( X_i \) means "this path has been good so far—stick with it."
- \( C \): **Exploration constant** (e.g., \( \sqrt{2} \), as discussed earlier). It scales the bonus for uncertainty.
- \( \sqrt{\frac{\ln N}{n_i}} \): **Exploration term** (confidence bonus).
  - \( N \): Total visits to the parent node (how much we've learned about this level).
  - \( \ln N \): Natural log of parent's visits (grows slowly, ensuring exploration ramps up as we gain confidence in the tree).
  - \( n_i \): Child's visits (low \( n_i \) → high bonus, since uncertainty is high).
  - The square root ensures the bonus decreases gradually (sublinearly) as \( n_i \) grows, per concentration inequalities like Hoeffding's.

**Step-by-Step How It Works in Selection**:
1. At each node, compute \( UCB_i \) for **all children**.
2. Pick the child with the **maximum UCB** (ties broken randomly).
3. Repeat until a leaf (unexpanded or terminal node).
4. Why "upper bound"? Statistically, \( UCB_i \) is designed so the true mean reward is ≤ \( UCB_i \) with high probability (e.g., 95%+). This optimism biases toward unproven actions.

**Example Calculation** (Assume rewards in [0,1], C=√2≈1.414):
- Parent N=100 (ln(100)≈4.605).
- Child A: n=50, w=40 → X_A=0.8, bonus=1.414 * √(4.605/50) ≈1.414*0.304≈0.43 → UCB_A≈1.23
- Child B: n=10, w=6 → X_B=0.6, bonus=1.414 * √(4.605/10)≈1.414*0.679≈0.96 → UCB_B≈1.56
- Child C: n=40, w=20 → X_C=0.5, bonus=1.414 * √(4.605/40)≈1.414*0.339≈0.48 → UCB_C≈0.98
- **Select B**: Despite lower average, its low visits give a big exploration boost—worth checking if it's a diamond in the rough.

**Theoretical Justification**:
- From **multi-armed bandit theory** (Auer et al., 2002): In a tree-unfolding view (UCT algorithm by Kocsis & Szepesvári, 2006), this UCB minimizes **regret** (expected loss from suboptimal choices) asymptotically.
- For bounded rewards [0,1], C=2 guarantees logarithmic regret, but √2 is empirically tuned for games (less aggressive, faster convergence).
- Edge cases: If n_i=0 (untried), UCB=∞ (infinite bonus—always explore new actions first).

**Limitations of Standard UCB**:
- Treats all actions equally at expansion (no prior bias).
- In high-branching games (e.g., Go: ~250 moves), it wastes time on bad actions before statistics catch up.

#### 2. PUCT: UCB with Priors (Enhanced Variant)

**PUCT** (Predictor + UCB for Trees) integrates **priors** (prior probabilities from a policy network) into UCB, making it "guided." This is the version in modern MCTS like AlphaGo/AlphaZero, where a neural network predicts promising moves upfront.

The formula is:

\[
PUCT_i = X_i + c \cdot P_i \cdot \frac{\sqrt{N}}{1 + n_i}
\]

Key differences from UCB:
- \( P_i \): **Prior probability** for action \( i \) (from policy net: \( P_i = \pi(a_i | s) \), where \( \pi \) is the predicted policy, normalized to sum to 1 over actions).
  - High P_i (e.g., 0.3 for a "intuitive" move) amplifies exploration for that child.
  - Low P_i (e.g., 0.001 for a dumb move) dampens it, even if unvisited.
- Exploration term tweak: \( \frac{\sqrt{N}}{1 + n_i} \) (instead of \( \sqrt{\frac{\ln N}{n_i}} \)).
  - The "+1" avoids division-by-zero for n_i=0.
  - √N grows with parent visits but is simpler; it scales bonus linearly with tree growth.
- \( c \): Dynamic constant, often \( c = c_{base} \cdot \frac{\ln(t + 1) + \epsilon}{1 + n_i} \), where t is iteration number (increases exploration over time).

**Step-by-Step How It Works**:
1. **At Expansion**: When adding a child for action a, set its prior: `child.prior = policy_net(state)[a]`.
2. **In Selection**: Compute PUCT for each child using the formula above.
3. Pick max PUCT child.
4. This biases the tree toward "expert-like" moves early, then refines with simulations.

**Example Calculation** (Assume same setup, but with priors; c=1.0):
- Child A: P_A=0.4, X_A=0.8, bonus=1.0 * 0.4 * √100 / (1+50) ≈0.4 * 10 / 51 ≈0.078 → PUCT_A≈0.878
- Child B: P_B=0.1, X_B=0.6, bonus=1.0 * 0.1 * 10 / 11 ≈0.1 * 0.909 ≈0.091 → PUCT_B≈0.691
- Child C: P_C=0.5, X_C=0.5, bonus=1.0 * 0.5 * 10 / 41 ≈0.5 * 0.244 ≈0.122 → PUCT_C≈0.622
- **Select A**: Now favors the high-prior, high-reward path—priors "nudge" without overriding stats.

**Why PUCT > UCB?**
- **Efficiency**: Priors act as a "soft filter," focusing 80-90% of budget on top actions (vs. uniform in UCB).
- **From AlphaZero (Silver et al., 2017)**: Combines with a value network (for rollout replacement), achieving superhuman play.
- **Tuning**: c_base≈1-3; priors updated via self-play (MCTS-guided training).

In code, you'd modify `best_child` to use PUCT if priors are available—fall back to UCB otherwise. Both ensure asymptotically optimal decisions, but PUCT scales to real-world complexity. If you want math proofs or code snippets, just ask!