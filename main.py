import numpy as np
from multiprocessing import Pool
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Global hyperparameters
N_ITERATIONS = 500  # Number of training iterations
N_GAMES = 200  # Number of self-play games per iteration
N_SIMULATIONS = 100  # Number of MCTS simulations per move
C_PUCT = 1.0  # UCB exploration constant
BATCH_SIZE = 64  # Training batch size
EPOCHS = 10  # Training epochs per iteration
LEARNING_RATE = 0.001  # Neural network learning rate
HIDDEN_SIZE = 256  # Neural network hidden layer size
NUM_RES_BLOCKS = 4  # Number of residual blocks
DROPOUT = 0.2  # Dropout rate
WEIGHT_DECAY = 1e-4  # L2 regularization
LR_STEP_SIZE = 100  # Learning rate scheduler step size
LR_GAMMA = 0.9  # Learning rate decay factor
EVAL_GAMES = 50  # Number of games for evaluation
EVAL_SIMULATIONS = N_SIMULATIONS  # MCTS simulations during evaluation
EVAL_INTERVAL = 10  # Evaluate every N iterations
SAVE_INTERVAL = 10  # Save checkpoint every N iterations
REPLAY_CAPACITY = 50000  # Maximum number of samples retained in replay buffer

# Temperature annealing configuration
TEMPERATURE_INITIAL = 1.0          # Exploration temperature for first moves / early iterations
TEMPERATURE_FINAL = 0.0            # Deterministic after threshold
TEMPERATURE_MOVE_THRESHOLD = 3     # Number of opening moves to keep high temperature
TEMPERATURE_ANNEAL_ITER = 100      # After this iteration, always use final temperature

# Data augmentation toggle (dihedral symmetries of 3x3 board)
SYMMETRY_AUGMENT = True


class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        
    def is_expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def q_from_parent_view(self):
        """Return Q value from the parent's perspective.
        self.value() is from this node's perspective (player to move when node was created).
        The parent needs the opposite sign when deciding (because after parent moves,
        perspective flips)."""
        return -self.value()

    def ucb_score(self, c_puct=1.0):
        """PUCT score combining exploitation (Q) and exploration (U) without infinities.
        Uses parent's visit count for scaling and correct sign for Q from parent's view."""
        parent_N = 1 + (self.parent.visit_count if self.parent else 0)
        U = c_puct * self.prior * (np.sqrt(parent_N) / (1 + self.visit_count))
        Q = self.q_from_parent_view()
        return Q + U
    
    def select_child(self, c_puct=1.0):
        """Select child with maximum PUCT score. Ties broken implicitly by prior via U."""
        return max(self.children.values(), key=lambda ch: ch.ucb_score(c_puct))
    
    def expand(self, game, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                new_state = game.copy()
                new_state.make_move(action)
                self.children[action] = MCTSNode(
                    state=new_state.state(), 
                    parent=self, 
                    action=action, 
                    prior=prob
                )
    
    def backup(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)  # Negate for opponent


class AlphaZeroMCTS:
    def __init__(self, game, model, c_puct=C_PUCT, num_simulations=N_SIMULATIONS):
        self.game = game
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
    
    def search(self, root_state):
        root = MCTSNode(root_state)

        # Expand root with Dirichlet noise (no backup here)
        game_copy_root = self.game.copy()
        if not game_copy_root.over():
            v_root, policy_root = self.model.predict(game_copy_root)
            legal = game_copy_root.valid_moves()
            probs = [(a, policy_root[a]) for a in legal]
            s = sum(p for _, p in probs)
            if s > 0:
                probs = [(a, p / s) for a, p in probs]
            if probs:
                alpha, eps = 0.3, 0.25
                noise = np.random.dirichlet([alpha] * len(probs))
                probs = [(a, (1 - eps) * p + eps * n) for (a, p), n in zip(probs, noise)]
            root.expand(game_copy_root, probs)

        for _ in range(self.num_simulations):
            node = root
            game_copy = self.game.copy()

            while node.is_expanded() and not game_copy.over():
                node = node.select_child(self.c_puct)
                game_copy.make_move(node.action)

            if not game_copy.over():
                v, policy = self.model.predict(game_copy)  # v is from player-to-move view
                legal = game_copy.valid_moves()
                probs = [(a, policy[a]) for a in legal]
                s = sum(p for _, p in probs)
                if s > 0:
                    probs = [(a, p / s) for a, p in probs]
                node.expand(game_copy, probs)
                node.backup(v)  # signed during recursion
            else:
                # Score is winner in {+1,-1}, convert to player-to-move view
                terminal_v = game_copy.score() * game_copy.current_player
                node.backup(terminal_v)

        return root
    
    def get_action_probabilities(self, root_state, temperature=1.0):
        root = self.search(root_state)
        
        actions = []
        visit_counts = []
        
        for action, child in root.children.items():
            actions.append(action)
            visit_counts.append(child.visit_count)
        
        if temperature == 0:
            # Greedy selection
            best_action_idx = np.argmax(visit_counts)
            probs = np.zeros(len(actions))
            probs[best_action_idx] = 1.0
        else:
            # Softmax with temperature
            visit_counts = np.array(visit_counts)
            visit_counts = visit_counts ** (1.0 / temperature)
            probs = visit_counts / np.sum(visit_counts)
        
        return actions, probs


class ResidualBlock(nn.Module):
    """Residual block for ResNet architecture"""
    
    def __init__(self, hidden_size: int, dropout: float = DROPOUT):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out += residual
        out = F.relu(out)
        
        return out


class AlphaZeroNet(nn.Module):
    """ResNet-based neural network for AlphaZero with value and policy heads"""
    
    def __init__(self, action_size: int = 9, hidden_size: int = HIDDEN_SIZE, num_res_blocks: int = NUM_RES_BLOCKS):
        super(AlphaZeroNet, self).__init__()
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Initial projection layer
        self.input_fc = nn.Linear(18, hidden_size)  # 3x3x2 = 18 input features
        self.input_bn = nn.LayerNorm(hidden_size)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout=DROPOUT) 
            for _ in range(num_res_blocks)
        ])
        
        # Value head
        self.value_fc1 = nn.Linear(hidden_size, 128)
        self.value_bn1 = nn.LayerNorm(128)
        self.value_fc2 = nn.Linear(128, 64)
        self.value_fc3 = nn.Linear(64, 1)
        self.value_dropout = nn.Dropout(DROPOUT)
        
        # Policy head  
        self.policy_fc1 = nn.Linear(hidden_size, 128)
        self.policy_bn1 = nn.LayerNorm(128)
        self.policy_fc2 = nn.Linear(128, 64)
        self.policy_fc3 = nn.Linear(64, action_size)
        self.policy_dropout = nn.Dropout(DROPOUT)
        
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)  # (batch_size, 18)
        
        # Initial projection
        x = self.input_fc(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Value head
        value = self.value_fc1(x)
        value = self.value_bn1(value)
        value = F.relu(value)
        value = self.value_dropout(value)
        value = F.relu(self.value_fc2(value))
        value = torch.tanh(self.value_fc3(value))  # [-1, 1]
        
        # Policy head
        policy = self.policy_fc1(x)
        policy = self.policy_bn1(policy)
        policy = F.relu(policy)
        policy = self.policy_dropout(policy)
        policy = F.relu(self.policy_fc2(policy))
        policy_logits = self.policy_fc3(policy)
        
        return value.squeeze(-1), policy_logits


class AlphaZeroModel:
    """PyTorch-based AlphaZero model with training capabilities"""
    
    def __init__(self, board_size: int = 3, action_size: int = 9, learning_rate: float = LEARNING_RATE):
        self.board_size = board_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Initialize network
        self.net = AlphaZeroNet(action_size=action_size)
        
        # Better optimizer with weight decay
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
        
        # Training replay buffer
        self.training_data = []
        self.replay_capacity = REPLAY_CAPACITY
    
    def encode_state(self, game):
        """Convert game state to neural network input format.
        Accepts either a game object or a (state_tuple, player) pair captured during self-play.
        Produces two planes: current player's stones and opponent stones, from the perspective of the stored player.
        """
        # Case 1: full game object
        if hasattr(game, 'board') and hasattr(game, 'current_player'):
            board = np.array(game.board).reshape(self.board_size, self.board_size)
            cur = game.current_player
        else:
            # Case 2: stored (state, player) pair
            if isinstance(game, (tuple, list)) and len(game) == 2 and isinstance(game[1], int) and isinstance(game[0], (tuple, list)) and len(game[0]) == self.board_size * self.board_size:
                board = np.array(game[0]).reshape(self.board_size, self.board_size)
                cur = game[1]
            else:
                # Fallback: assume raw board without player info, default current player = 1
                raw = game[0] if isinstance(game, (tuple, list)) and len(game) > 0 else [0] * (self.board_size * self.board_size)
                board = np.array(raw).reshape(self.board_size, self.board_size)
                cur = 1
        current_player_plane = (board == cur).astype(np.float32)
        opponent_plane = (board == -cur).astype(np.float32)
        encoded = np.stack([current_player_plane, opponent_plane], axis=-1)
        return encoded
    
    def predict(self, game):
        """Get value and policy predictions from the network"""
        state = self.encode_state(game)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        
        self.net.eval()
        with torch.no_grad():
            value, policy_logits = self.net(state_tensor)
        
        # Convert to numpy and extract single predictions
        value = float(value[0])
        policy_logits = policy_logits[0].numpy()
        
        # Apply softmax to get probabilities
        policy = self._softmax(policy_logits)
        
        return value, policy
    
    def _softmax(self, x):
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def compute_loss(self, batch_states, batch_values, batch_policies):
        """Compute loss for a batch of training data with legality masking on policy head."""
        values, policy_logits = self.net(batch_states)

        # Derive legality mask from planes: empty if neither plane occupies the cell
        with torch.no_grad():
            # batch_states shape: [B, 3, 3, 2] for 3x3; generalizes by flattening
            occ = batch_states[..., 0] + batch_states[..., 1]  # [B, 3, 3]
            legal_mask = (occ.view(occ.size(0), -1) == 0)      # [B, 9] True=legal

        # Mask logits for illegal moves
        policy_logits = policy_logits.masked_fill(~legal_mask, -1e9)
        log_probs = F.log_softmax(policy_logits, dim=1)

        # Loss components
        value_loss = F.mse_loss(values, batch_values)
        policy_loss = -torch.mean(torch.sum(batch_policies * log_probs, dim=1))
        total_loss = value_loss + policy_loss
        return total_loss, value_loss, policy_loss
    
    def train_step(self, batch_states, batch_values, batch_policies):
        """Single training step"""
        self.net.train()
        self.optimizer.zero_grad()
        total_loss, value_loss, policy_loss = self.compute_loss(batch_states, batch_values, batch_policies)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        return total_loss.item(), value_loss.item(), policy_loss.item()
    
    def add_training_data(self, state, action_probs, value):
        """Add training example to replay buffer and enforce capacity.
        State should be either a game object or (board_tuple, player) pair to preserve perspective."""
        self.training_data.append((state, action_probs, value))
        if len(self.training_data) > self.replay_capacity:
            self.training_data = self.training_data[-self.replay_capacity:]
    
    def train(self, batch_size: int = BATCH_SIZE, epochs: int = EPOCHS):
        """Train using replay buffer sampling; keep buffer contents across iterations."""
        if len(self.training_data) < batch_size:
            return

        # Sample a working subset (cap at 4096 for efficiency)
        subset_size = min(len(self.training_data), 4096)
        idx = torch.randperm(len(self.training_data))[:subset_size]
        batch_samples = [self.training_data[i] for i in idx.tolist()]

        states = []
        values = []
        policies = []
        for state, action_probs, value in batch_samples:
            states.append(self.encode_state(state))
            values.append(value)
            policies.append([
                action_probs.get(i, 0.0) if isinstance(action_probs, dict) else action_probs[i]
                for i in range(self.action_size)
            ])

        states = torch.FloatTensor(np.array(states))
        values = torch.FloatTensor(np.array(values))
        policies = torch.FloatTensor(np.array(policies))

        dataset_size = len(states)
        best_loss = float('inf')

        for epoch in range(epochs):
            indices = torch.randperm(dataset_size)
            states_shuffled = states[indices]
            values_shuffled = values[indices]
            policies_shuffled = policies[indices]

            total_loss = 0
            total_value_loss = 0
            total_policy_loss = 0
            num_batches = 0

            for i in range(0, dataset_size, batch_size):
                batch_end = min(i + batch_size, dataset_size)
                batch_states = states_shuffled[i:batch_end]
                batch_values = values_shuffled[i:batch_end]
                batch_policies = policies_shuffled[i:batch_end]

                loss, value_loss, policy_loss = self.train_step(batch_states, batch_values, batch_policies)
                total_loss += loss
                total_value_loss += value_loss
                total_policy_loss += policy_loss
                num_batches += 1

            if num_batches == 0:
                print(f"Warning: No valid batches in epoch {epoch}")
                continue

            avg_loss = total_loss / num_batches
            avg_value_loss = total_value_loss / num_batches
            avg_policy_loss = total_policy_loss / num_batches

            if epoch % 2 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Value={avg_value_loss:.4f}, Policy={avg_policy_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss

            self.scheduler.step()

        print(f"Training complete. Best loss (subset): {best_loss:.4f}")
    
    def save_model(self, filepath: str):
        """Save the trained model to a file"""
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'board_size': self.board_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'optimizer_type': 'AdamW',
            'weight_decay': WEIGHT_DECAY
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, device: str = 'cpu'):
        """Load a trained model from a file"""
        checkpoint = torch.load(filepath, map_location=device)
        self.board_size = checkpoint['board_size']
        self.action_size = checkpoint['action_size']
        self.learning_rate = checkpoint['learning_rate']
        weight_decay = checkpoint.get('weight_decay', WEIGHT_DECAY)
        opt_type = checkpoint.get('optimizer_type', 'AdamW')
        
        # Recreate network with loaded parameters
        self.net = AlphaZeroNet(action_size=self.action_size)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.to(device)
        
        # Recreate optimizer consistently
        if opt_type == 'AdamW':
            self.optimizer = optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {filepath} (optimizer={opt_type}, weight_decay={weight_decay})")
    
    @classmethod
    def load_from_file(cls, filepath: str, device: str = 'cpu'):
        """Create a new AlphaZeroModel instance from a saved file"""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            board_size=checkpoint['board_size'],
            action_size=checkpoint['action_size'],
            learning_rate=checkpoint['learning_rate']
        )
        model.net.load_state_dict(checkpoint['model_state_dict'])
        model.net.to(device)
        weight_decay = checkpoint.get('weight_decay', WEIGHT_DECAY)
        opt_type = checkpoint.get('optimizer_type', 'AdamW')
        if opt_type == 'AdamW':
            model.optimizer = optim.AdamW(model.net.parameters(), lr=model.learning_rate, weight_decay=weight_decay)
        else:
            model.optimizer = optim.Adam(model.net.parameters(), lr=model.learning_rate, weight_decay=weight_decay)
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath} (optimizer={opt_type}, weight_decay={weight_decay})")
        return model


def _play_game_worker(args):
    """Worker function for parallel self-play (must be at module level for pickling).
    Args:
        game_class: Game class to instantiate
        model_state: State dict of the current model
        mcts_simulations: Number of simulations for MCTS
        iteration: Current training iteration (for temperature annealing)
    """
    game_class, model_state, mcts_simulations, iteration = args
    
    # Recreate model from state dict
    model = AlphaZeroModel()
    model.net.load_state_dict(model_state)
    model.net.eval()  # Set to evaluation mode for inference
    
    # Play one game
    game = game_class()
    mcts = AlphaZeroMCTS(game, model, num_simulations=mcts_simulations)
    
    training_examples = []
    
    move_index = 0
    while not game.over():
        # Store the current player BEFORE making the move
        current_player = game.current_player
        
        # Get action probabilities from MCTS
        # Annealed temperature: high for opening moves & early iterations
        if iteration >= TEMPERATURE_ANNEAL_ITER:
            temp = TEMPERATURE_FINAL
        else:
            temp = TEMPERATURE_INITIAL if move_index < TEMPERATURE_MOVE_THRESHOLD else TEMPERATURE_FINAL
        actions, action_probs = mcts.get_action_probabilities(game.state(), temperature=temp)
        
        # Store training example (state, policy, value_placeholder, player)
        training_examples.append([(game.state(), current_player), dict(zip(actions, action_probs)), None, current_player])
        
        # Sample action from MCTS policy
        action = np.random.choice(actions, p=action_probs)
        game.make_move(action)
        move_index += 1
    
    # Fill in actual game result for all training examples
    game_result = game.score()  # +1 if player 1 won, -1 if player -1 won, 0 for draw
    for example in training_examples:
        state, action_probs, _, player = example
        # Value from the perspective of the player who made the move
        example[2] = game_result * player
    
    # Remove the player marker before returning (only need state, probs, value)
    return [[ex[0], ex[1], ex[2]] for ex in training_examples]


class AlphaZeroTrainer:
    def __init__(self, game_class, model, mcts_simulations=N_SIMULATIONS):
        self.game_class = game_class
        self.model = model
        self.mcts_simulations = mcts_simulations
    
    def _compute_temperature(self, iteration: int, move_index: int) -> float:
        """Compute annealed temperature for a move in a game."""
        if iteration >= TEMPERATURE_ANNEAL_ITER:
            return TEMPERATURE_FINAL
        if move_index < TEMPERATURE_MOVE_THRESHOLD:
            return TEMPERATURE_INITIAL
        return TEMPERATURE_FINAL

    def _symmetry_permutations(self):
        """Return list of index permutations representing dihedral symmetries of 3x3 board."""
        perms = []
        perms.append([0,1,2,3,4,5,6,7,8])  # identity
        def idx(r,c): return r*3 + c
        # Rot90
        rot90 = [None]*9
        for r in range(3):
            for c in range(3):
                rot90[idx(r,c)] = idx(c,2-r)
        perms.append(rot90)
        # Rot180
        rot180 = [rot90[i] for i in rot90]
        perms.append(rot180)
        # Rot270
        rot270 = [rot180[i] for i in rot90]
        perms.append(rot270)
        # Horizontal flip (r,c)->(2-r,c)
        refl_h = [None]*9
        for r in range(3):
            for c in range(3):
                refl_h[idx(r,c)] = idx(2-r,c)
        perms.append(refl_h)
        # Vertical flip (r,c)->(r,2-c)
        refl_v = [None]*9
        for r in range(3):
            for c in range(3):
                refl_v[idx(r,c)] = idx(r,2-c)
        perms.append(refl_v)
        # Main diagonal (r,c)->(c,r)
        refl_d1 = [None]*9
        for r in range(3):
            for c in range(3):
                refl_d1[idx(r,c)] = idx(c,r)
        perms.append(refl_d1)
        # Anti diagonal (r,c)->(2-c,2-r)
        refl_d2 = [None]*9
        for r in range(3):
            for c in range(3):
                refl_d2[idx(r,c)] = idx(2-c,2-r)
        perms.append(refl_d2)
        return perms

    def _augment_examples(self, examples):
        if not SYMMETRY_AUGMENT:
            return examples
        perms = self._symmetry_permutations()
        augmented = []
        for state, action_probs, value in examples:
            board_tuple, player = state if (isinstance(state, (tuple,list)) and len(state)==2) else (state, 1)
            board_list = list(board_tuple)
            # Convert policy to list
            if isinstance(action_probs, dict):
                base_policy = [action_probs.get(i,0.0) for i in range(9)]
            else:
                base_policy = list(action_probs)
            for perm in perms:
                new_board = [None]*9
                for i,val in enumerate(board_list):
                    new_board[perm[i]] = val
                # Permute policy: probability for original action i moves to perm[i]
                new_policy = {perm[i]: base_policy[i] for i in range(9) if base_policy[i] > 0}
                augmented.append(((tuple(new_board), player), new_policy, value))
        return augmented

    def self_play_game(self, iteration: int):
        game = self.game_class()
        mcts = AlphaZeroMCTS(game, self.model, num_simulations=self.mcts_simulations)
        training_examples = []
        move_index = 0
        while not game.over():
            current_player = game.current_player
            temp = self._compute_temperature(iteration, move_index)
            actions, action_probs = mcts.get_action_probabilities(game.state(), temperature=temp)
            training_examples.append([(game.state(), current_player), dict(zip(actions, action_probs)), None, current_player])
            action = np.random.choice(actions, p=action_probs)
            game.make_move(action)
            move_index += 1
        game_result = game.score()
        for example in training_examples:
            _, _, _, player = example
            example[2] = game_result * player
        return [[ex[0], ex[1], ex[2]] for ex in training_examples]
    
    def train_iteration(self, iteration: int, num_games=N_GAMES, num_workers=None):
        """
        Run one training iteration with self-play games.
        
        Args:
            num_games: Number of self-play games to generate
            num_workers: Number of parallel workers (None = auto-detect, 1 = sequential)
        """
        print(f"Starting training iteration {iteration} with {num_games} self-play games...")

        # Auto-detect number of workers if not specified
        if num_workers is None:
            num_workers = max(1, os.cpu_count() - 1)  # Leave one core free

        all_training_examples = []

        if num_workers > 1:
            # Parallel self-play using multiprocessing
            print(f"Using {num_workers} parallel workers...")

            # Prepare arguments for each game
            model_state = self.model.net.state_dict()
            worker_args = [
                (self.game_class, model_state, self.mcts_simulations, iteration)
                for _ in range(num_games)
            ]

            # Run games in parallel
            with Pool(processes=num_workers) as pool:
                results = pool.map(_play_game_worker, worker_args)

            # Flatten results
            for examples in results:
                all_training_examples.extend(examples)

            print(f"Parallel self-play complete. Generated {len(all_training_examples)} raw training examples.")
        else:
            # Sequential self-play
            print("Using sequential self-play...")
            for game_num in range(num_games):
                if game_num % 100 == 0:
                    print(f"Self-play game {game_num}/{num_games}")
                examples = self.self_play_game(iteration)
                all_training_examples.extend(examples)

        # Symmetry augmentation
        augmented_examples = self._augment_examples(all_training_examples)
        for state, action_probs, value in augmented_examples:
            self.model.add_training_data(state, action_probs, value)

        # Train the model
        print("Training neural network (post-augmentation)...")
        self.model.train()
        print(f"Training iteration complete. Raw examples: {len(all_training_examples)}, Augmented: {len(augmented_examples)}")

    
    def evaluate_model(self, opponent_model, num_games=EVAL_GAMES):
        """Compare current model against opponent model"""
        wins = 0
        losses = 0
        draws = 0
        
        for game_num in range(num_games):
            game = self.game_class()
            
            # Alternate who goes first
            current_first = game_num % 2 == 0
            
            if current_first:
                current_mcts = AlphaZeroMCTS(game, self.model, num_simulations=EVAL_SIMULATIONS)
                opponent_mcts = AlphaZeroMCTS(game, opponent_model, num_simulations=EVAL_SIMULATIONS)
                players = [current_mcts, opponent_mcts]
            else:
                opponent_mcts = AlphaZeroMCTS(game, opponent_model, num_simulations=EVAL_SIMULATIONS)
                current_mcts = AlphaZeroMCTS(game, self.model, num_simulations=EVAL_SIMULATIONS)
                players = [opponent_mcts, current_mcts]
            
            player_idx = 0
            while not game.over():
                actions, probs = players[player_idx].get_action_probabilities(game.state(), temperature=0.1)
                action = actions[np.argmax(probs)]
                game.make_move(action)
                player_idx = 1 - player_idx
            
            # Check result from current model's perspective
            result = game.score()
            
            if result == 0:
                draws += 1
            elif current_first:
                # Current model played first (player 1)
                if result > 0:
                    wins += 1
                else:
                    losses += 1
            else:
                # Current model played second (player -1)
                if result < 0:
                    wins += 1
                else:
                    losses += 1
        
        win_rate = wins / num_games
        print(f"Win rate against opponent: {win_rate:.2f} (W:{wins} L:{losses} D:{draws})")
        return wins, draws, losses


# Example usage and game interface
class TicTacToeGame:
    """Example game implementation for tic-tac-toe"""
    def __init__(self):
        self.board = [0] * 9  # 0=empty, 1=X, -1=O
        self.current_player = 1
    
    def copy(self):
        new_game = TicTacToeGame()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game
    
    def state(self):
        return tuple(self.board)
    
    def valid_moves(self):
        return [i for i in range(9) if self.board[i] == 0]
    
    def make_move(self, action):
        if self.board[action] == 0:
            self.board[action] = self.current_player
            self.current_player = -self.current_player
            return True
        return False
    
    def over(self):
        return self.score() != 0 or len(self.valid_moves()) == 0
    
    def score(self):
        # Check rows, columns, diagonals
        lines = [
            [0,1,2], [3,4,5], [6,7,8],  # rows
            [0,3,6], [1,4,7], [2,5,8],  # columns  
            [0,4,8], [2,4,6]             # diagonals
        ]
        
        for line in lines:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]] != 0:
                return self.board[line[0]]
        
        return 0  # Draw or game not over


# Example training script
def train_alphazero():
    """Train AlphaZero with improved configuration"""
    print("=== Training AlphaZero ===")
    
    # Better hyperparameters
    model = AlphaZeroModel(
        board_size=3, 
        action_size=9, 
        learning_rate=LEARNING_RATE  # Lower learning rate for stability
    )
    
    trainer = AlphaZeroTrainer(
        TicTacToeGame, 
        model, 
        mcts_simulations=N_SIMULATIONS  # More MCTS simulations
    )
    
    # Save initial baseline model as iteration 0
    print("Saving initial baseline model as iteration 0...")
    baseline_model_path = "models/alphazero_iter_0.pth"
    os.makedirs("models", exist_ok=True)
    model.save_model(baseline_model_path)
    
    # Initialize CSV file for evaluation results
    csv_path = "models/evaluation_results.csv"
    with open(csv_path, 'w') as f:
        f.write("current_iter,opponent_iter,wins,draws,losses,win_rate\n")
    
    # Longer training with better monitoring
    for iteration in range(1, N_ITERATIONS + 1):  # Start from 1 since 0 is baseline
        print(f"\n=== Training Iteration {iteration}/{N_ITERATIONS} ===")

        # More games per iteration for better data
        trainer.train_iteration(iteration, num_games=N_GAMES)

        # Save checkpoint every SAVE_INTERVAL iterations
        if iteration % SAVE_INTERVAL == 0:
            checkpoint_path = f"models/alphazero_iter_{iteration}.pth"
            model.save_model(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # Evaluate against all previous checkpoints
            print(f"\n*** Evaluating iteration {iteration} against all previous checkpoints ***")
            saved_iterations = list(range(0, iteration, SAVE_INTERVAL))

            for prev_iter in saved_iterations:
                print(f"\nEvaluating against iteration {prev_iter}...")
                prev_model_path = f"models/alphazero_iter_{prev_iter}.pth"
                prev_model = AlphaZeroModel.load_from_file(prev_model_path)

                wins, draws, losses = trainer.evaluate_model(prev_model, num_games=EVAL_GAMES)
                win_rate = wins / EVAL_GAMES

                # Save to CSV
                with open(csv_path, 'a') as f:
                    f.write(f"{iteration},{prev_iter},{wins},{draws},{losses},{win_rate:.4f}\n")

                print(f"Results vs iter {prev_iter}: W:{wins} D:{draws} L:{losses} (WR: {win_rate:.2%})")
    
    # Save final model
    final_path = "models/alphazero_final.pth"
    model.save_model(final_path)
    print(f"\nTraining complete! Final model saved as {final_path}")
    print(f"Evaluation results saved to {csv_path}")
    
    return model, trainer


if __name__ == "__main__":
    # Run training
    trained_model, trainer = train_alphazero()
    print("Training complete!")