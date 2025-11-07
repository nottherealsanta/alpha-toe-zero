#!/usr/bin/env python3
# alphazero_4x4x4.py  — AlphaZero for 4×4×4 (Qubic) with 3D conv net + symmetry augmentation + Elo every 10 iters
import os
import sys
import math
import csv
import itertools
from datetime import datetime
from multiprocessing import Pool
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# ======================
# Global hyperparameters
# ======================

# Board/game
BOARD_SHAPE = (4, 4, 4)         # (D, H, W)
ACTION_SIZE = BOARD_SHAPE[0] * BOARD_SHAPE[1] * BOARD_SHAPE[2]  # 64

# Training loop
N_ITERATIONS = 200
N_GAMES = 64                    # self-play games per iteration
N_SIMULATIONS = 100              # MCTS simulations per move (train)
EVAL_SIMULATIONS = 100           # MCTS simulations per move (eval/tournament)
BATCH_SIZE = 128
EPOCHS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 50
LR_GAMMA = 0.9
REPLAY_CAPACITY = 200_000

# Network
CHANNELS = 32                    # base channels for 3D conv trunk
NUM_RES_BLOCKS = 4
DROPOUT = 0.15

# MCTS
C_PUCT = 1

# Temperature annealing
TEMPERATURE_INITIAL = 1.0
TEMPERATURE_FINAL = 0.0
TEMPERATURE_MOVE_THRESHOLD = 6
TEMPERATURE_ANNEAL_ITER = 100

# Persistence / evaluation
SAVE_DIR = "models_4x4x4"
SAVE_INTERVAL = 10               # save checkpoint every N iters
EVAL_ELO_INTERVAL = 10           # Elo tournament every N iters
EVAL_GAMES_PER_PAIR = 6          # per pair in tournament
K_FACTOR = 32.0                  # Elo K

# Augmentation
SYMMETRY_AUGMENT = True          # enable symmetry augmentation
SYMM_INCLUDE_REFLECTIONS = False # False -> 24 rotations; True -> 48 rotations+reflections

# Progress control (set PROGRESS=1 in env to enable tqdm bars)
SHOW_PROGRESS = 1

def progress(iterable, **kwargs):
    """Return tqdm progress iterator if SHOW_PROGRESS else the raw iterable."""
    return tqdm(iterable, **kwargs) if SHOW_PROGRESS else iterable


# ===============
# Utility helpers
# ===============

def to_coords(i: int) -> Tuple[int, int, int]:
    """Map flat index -> (x, y, z) with x fastest."""
    x = i % 4
    y = (i // 4) % 4
    z = i // 16
    return x, y, z


def to_index(x: int, y: int, z: int) -> int:
    """Map (x, y, z) -> flat index with x fastest."""
    return z * 16 + y * 4 + x


# ---------- cube symmetry mappings (cache) ----------

_CUBE_SYMM_CACHE: Dict[bool, List[List[int]]] = {}

def _perm_parity(perm: Tuple[int, int, int]) -> int:
    """+1 even, -1 odd."""
    inv = 0
    p = list(perm)
    for i in range(3):
        for j in range(i + 1, 3):
            if p[i] > p[j]:
                inv += 1
    return 1 if (inv % 2 == 0) else -1

def get_cube_symmetry_mappings(include_reflections: bool = False) -> List[List[int]]:
    """
    Return list of index permutations for 4×4×4 cube symmetries.
    include_reflections=False -> 24 rotations (SO(3) on cube).
    include_reflections=True  -> 48 rotations+reflections (full octahedral group).
    Each mapping maps old_index -> new_index.
    """
    global _CUBE_SYMM_CACHE
    if include_reflections in _CUBE_SYMM_CACHE:
        return _CUBE_SYMM_CACHE[include_reflections]

    maps: List[List[int]] = []
    for perm in itertools.permutations((0, 1, 2), 3):  # axis permutation
        perm_sign = _perm_parity(perm)                 # +1 even, -1 odd
        for sgn in itertools.product((1, -1), repeat=3):  # axis flips
            det = (sgn[0] * sgn[1] * sgn[2]) * perm_sign
            if include_reflections:
                keep = True
            else:
                keep = (det == 1)  # rotations only
            if not keep:
                continue

            mapping = [0] * ACTION_SIZE
            for idx in range(ACTION_SIZE):
                x, y, z = to_coords(idx)
                v = [x, y, z]
                xn = v[perm[0]]
                yn = v[perm[1]]
                zn = v[perm[2]]
                if sgn[0] == -1:
                    xn = 3 - xn
                if sgn[1] == -1:
                    yn = 3 - yn
                if sgn[2] == -1:
                    zn = 3 - zn
                mapping[idx] = to_index(xn, yn, zn)
            maps.append(mapping)

    # Deduplicate just in case
    uniq = []
    seen = set()
    for m in maps:
        t = tuple(m)
        if t not in seen:
            seen.add(t)
            uniq.append(m)
    _CUBE_SYMM_CACHE[include_reflections] = uniq
    return uniq


# ============
# Game: Qubic
# ============

class Qubic4x4x4:
    """4×4×4 Tic-Tac-Toe (Qubic). 1 = X, -1 = O, 0 = empty."""
    def __init__(self):
        self.board = [0] * ACTION_SIZE
        self.current_player = 1
        if not hasattr(Qubic4x4x4, "_WIN_LINES"):
            Qubic4x4x4._WIN_LINES = Qubic4x4x4._compute_win_lines()

    @staticmethod
    def _compute_win_lines() -> List[List[int]]:
        lines = []
        # Axes: x, y, z
        for z in range(4):
            for y in range(4):
                lines.append([to_index(x, y, z) for x in range(4)])  # x-lines
        for z in range(4):
            for x in range(4):
                lines.append([to_index(x, y, z) for y in range(4)])  # y-lines
        for y in range(4):
            for x in range(4):
                lines.append([to_index(x, y, z) for z in range(4)])  # z-lines

        # Plane diagonals: XY planes
        for z in range(4):
            lines.append([to_index(i, i, z) for i in range(4)])
            lines.append([to_index(i, 3 - i, z) for i in range(4)])
        # XZ planes
        for y in range(4):
            lines.append([to_index(i, y, i) for i in range(4)])
            lines.append([to_index(i, y, 3 - i) for i in range(4)])
        # YZ planes
        for x in range(4):
            lines.append([to_index(x, i, i) for i in range(4)])
            lines.append([to_index(x, i, 3 - i) for i in range(4)])

        # Space diagonals (4)
        lines.append([to_index(i, i, i) for i in range(4)])               # + + +
        lines.append([to_index(i, i, 3 - i) for i in range(4)])           # + + -
        lines.append([to_index(i, 3 - i, i) for i in range(4)])           # + - +
        lines.append([to_index(3 - i, i, i) for i in range(4)])           # - + +
        return lines

    def copy(self):
        g = Qubic4x4x4()
        g.board = self.board.copy()
        g.current_player = self.current_player
        return g

    def state(self):
        return tuple(self.board)

    def valid_moves(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v == 0]

    def make_move(self, action: int) -> bool:
        if self.board[action] != 0:
            return False
        self.board[action] = self.current_player
        self.current_player = -self.current_player
        return True

    def _winner(self) -> int:
        b = self.board
        for line in self._WIN_LINES:
            s = b[line[0]] + b[line[1]] + b[line[2]] + b[line[3]]
            if s == 4:
                return 1
            if s == -4:
                return -1
        return 0

    def over(self) -> bool:
        if self._winner() != 0:
            return True
        return all(v != 0 for v in self.board)

    def score(self) -> int:
        w = self._winner()
        return w if w != 0 else 0


# =============
# MCTS classes
# =============

class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[int, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = float(prior)

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def q_from_parent_view(self) -> float:
        return -self.value()

    def ucb_score(self, c_puct: float) -> float:
        parent_N = 1 + (self.parent.visit_count if self.parent else 0)
        U = c_puct * self.prior * (math.sqrt(parent_N) / (1 + self.visit_count))
        Q = self.q_from_parent_view()
        return Q + U

    def select_child(self, c_puct: float) -> "MCTSNode":
        return max(self.children.values(), key=lambda ch: ch.ucb_score(c_puct))

    def expand(self, game, action_probs: List[Tuple[int, float]]):
        for action, prob in action_probs:
            if action not in self.children:
                g2 = game.copy()
                g2.make_move(action)
                self.children[action] = MCTSNode(
                    state=g2.state(), parent=self, action=action, prior=prob
                )

    def backup(self, value: float):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)


class AlphaZeroMCTS:
    def __init__(self, game, model, c_puct=C_PUCT, num_simulations=N_SIMULATIONS, add_dirichlet=True,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25):
        self.game = game
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.add_dirichlet = add_dirichlet
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def search(self, root_state):
        root = MCTSNode(root_state)
        game_copy_root = self.game.copy()
        if not game_copy_root.over():
            v_root, policy_root = self.model.predict(game_copy_root)
            legal = game_copy_root.valid_moves()
            probs = [(a, policy_root[a]) for a in legal]
            s = sum(p for _, p in probs)
            if s > 0:
                probs = [(a, p / s) for a, p in probs]
            if self.add_dirichlet and probs:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(probs))
                probs = [(a, (1 - self.dirichlet_eps) * p + self.dirichlet_eps * n)
                         for (a, p), n in zip(probs, noise)]
            root.expand(game_copy_root, probs)

        # Optional MCTS simulation progress (can be quite verbose; controlled by SHOW_PROGRESS)
        sim_iter = range(self.num_simulations)
        for _ in sim_iter:
            node = root
            game_copy = self.game.copy()
            while node.is_expanded() and not game_copy.over():
                node = node.select_child(self.c_puct)
                game_copy.make_move(node.action)

            if not game_copy.over():
                v, policy = self.model.predict(game_copy)
                legal = game_copy.valid_moves()
                probs = [(a, policy[a]) for a in legal]
                s = sum(p for _, p in probs)
                if s > 0:
                    probs = [(a, p / s) for a, p in probs]
                node.expand(game_copy, probs)
                node.backup(v)
            else:
                terminal_v = game_copy.score() * game_copy.current_player
                node.backup(terminal_v)

        return root

    def get_action_probabilities(self, root_state, temperature=1.0):
        root = self.search(root_state)
        actions, visits = [], []
        for a, ch in root.children.items():
            actions.append(a)
            visits.append(ch.visit_count)

        if not actions:
            legal = self.game.valid_moves()
            if not legal:
                return [], np.array([])
            actions = legal
            visits = [1] * len(actions)

        if temperature == 0:
            best_idx = int(np.argmax(visits))
            probs = np.zeros(len(actions), dtype=np.float32)
            probs[best_idx] = 1.0
        else:
            vc = np.array(visits, dtype=np.float64) ** (1.0 / temperature)
            probs = (vc / vc.sum()).astype(np.float32)
        return actions, probs


# ============================
# 3D Conv ResNet architecture
# ============================

class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = F.relu(out)
        return out


class AlphaZeroConvNet(nn.Module):
    """3D-Conv ResNet with value and policy heads."""
    def __init__(self, board_shape=BOARD_SHAPE, action_size=ACTION_SIZE,
                 channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS):
        super().__init__()
        self.board_shape = tuple(board_shape)
        self.action_size = int(action_size)
        self.channels = int(channels)
        self.num_res_blocks = int(num_res_blocks)

        # Trunk
        self.in_conv = nn.Conv3d(2, channels, kernel_size=3, padding=1, bias=False)
        self.in_bn = nn.BatchNorm3d(channels)
        self.res_blocks = nn.ModuleList([ResidualBlock3D(channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv3d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm3d(2)
        self.policy_fc = nn.Linear(2 * np.prod(self.board_shape), self.action_size)

        # Value head
        self.value_conv = nn.Conv3d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm3d(1)
        self.value_fc1 = nn.Linear(1 * np.prod(self.board_shape), 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: [B, 2, D, H, W]
        out = self.in_conv(x)
        out = self.in_bn(out)
        out = F.relu(out)
        for rb in self.res_blocks:
            out = rb(out)

        # Policy
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = torch.flatten(p, 1)
        p_logits = self.policy_fc(p)  # [B, ACTION_SIZE]

        # Value
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v)
        v = torch.flatten(v, 1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)  # [B]

        return v, p_logits


# ====================
# AlphaZero model API
# ====================

class AlphaZeroModel:
    def __init__(self, board_shape=BOARD_SHAPE, action_size=ACTION_SIZE,
                 channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS,
                 learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                 device: str = "cpu"):
        self.board_shape = tuple(board_shape)
        self.action_size = int(action_size)
        self.channels = int(channels)
        self.num_res_blocks = int(num_res_blocks)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.device = torch.device(device)

        self.net = AlphaZeroConvNet(
            board_shape=self.board_shape,
            action_size=self.action_size,
            channels=self.channels,
            num_res_blocks=self.num_res_blocks
        ).to(self.device)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

        self.training_data: List[Tuple[Tuple, Dict[int, float], float]] = []
        self.replay_capacity = REPLAY_CAPACITY

    # ---- encoding / inference ----
    def encode_state(self, game_or_pair):
        """Return [2, D, H, W] channels-first, current-player perspective."""
        if hasattr(game_or_pair, 'board') and hasattr(game_or_pair, 'current_player'):
            board = np.array(game_or_pair.board, dtype=np.int8)
            cur = int(game_or_pair.current_player)
        else:
            st, cur = game_or_pair
            board = np.array(st, dtype=np.int8)

        d, h, w = self.board_shape
        board3 = board.reshape(d, h, w)
        current_plane = (board3 == cur).astype(np.float32)
        opponent_plane = (board3 == -cur).astype(np.float32)
        enc = np.stack([current_plane, opponent_plane], axis=0)  # [2, D, H, W]
        return enc

    @torch.no_grad()
    def predict(self, game_or_pair):
        self.net.eval()
        state = self.encode_state(game_or_pair)
        x = torch.from_numpy(state).unsqueeze(0).to(self.device)  # [1, 2, D, H, W]
        v, logits = self.net(x)
        v = float(v[0].cpu())
        policy = F.softmax(logits[0], dim=0).cpu().numpy()
        return v, policy

    # ---- training ----
    def _compute_loss(self, batch_states, batch_values, batch_policies):
        values_pred, policy_logits = self.net(batch_states)
        with torch.no_grad():
            occ = batch_states[:, 0] + batch_states[:, 1]              # [B, D, H, W]
            legal_mask = (occ.view(occ.size(0), -1) == 0)              # [B, A]
        policy_logits = policy_logits.masked_fill(~legal_mask, -1e9)
        log_probs = F.log_softmax(policy_logits, dim=1)
        value_loss = F.mse_loss(values_pred, batch_values)
        policy_loss = -(batch_policies * log_probs).sum(dim=1).mean()
        total = value_loss + policy_loss
        return total, value_loss, policy_loss

    def train_step(self, batch_states, batch_values, batch_policies):
        self.net.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, v_loss, p_loss = self._compute_loss(batch_states, batch_values, batch_policies)
        total.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        return total.item(), v_loss.item(), p_loss.item()

    def add_training_data(self, state, action_probs, value):
        self.training_data.append((state, action_probs, value))
        if len(self.training_data) > self.replay_capacity:
            self.training_data = self.training_data[-self.replay_capacity:]

    def train(self, batch_size=BATCH_SIZE, epochs=EPOCHS):
        if len(self.training_data) < batch_size:
            return
        subset_size = min(len(self.training_data), 8192)
        idx = np.random.choice(len(self.training_data), subset_size, replace=False)
        batch = [self.training_data[i] for i in idx]

        states, values, policies = [], [], []
        for s, a_probs, v in batch:
            states.append(self.encode_state(s))
            values.append(v)
            if isinstance(a_probs, dict):
                pol = np.zeros(self.action_size, dtype=np.float32)
                for k, p in a_probs.items():
                    pol[int(k)] = float(p)
            else:
                pol = np.array(a_probs, dtype=np.float32)
            policies.append(pol)

        states_t = torch.from_numpy(np.asarray(states)).to(self.device)       # [B, 2, D, H, W]
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        policies_t = torch.from_numpy(np.asarray(policies)).to(self.device)

        N = states_t.size(0)
        best_loss = float('inf')
        for _ in range(epochs):
            perm = torch.randperm(N)
            for i in range(0, N, batch_size):
                sl = perm[i:i + batch_size]
                loss, v_loss, p_loss = self.train_step(states_t[sl], values_t[sl], policies_t[sl])
                best_loss = min(best_loss, loss)
            self.scheduler.step()
        print(f"Train: best loss {best_loss:.4f} on subset {subset_size}")

    # ---- save/load ----
    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'board_shape': self.board_shape,
            'action_size': self.action_size,
            'channels': self.channels,
            'num_res_blocks': self.num_res_blocks,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'optimizer_type': 'AdamW'
        }, filepath)
        print(f"Saved: {filepath}")

    def load_model(self, filepath: str, device: str = 'cpu'):
        ckpt = torch.load(filepath, map_location=device)
        self.board_shape = tuple(ckpt['board_shape'])
        self.action_size = int(ckpt['action_size'])
        self.channels = int(ckpt['channels'])
        self.num_res_blocks = int(ckpt['num_res_blocks'])
        self.learning_rate = float(ckpt['learning_rate'])
        self.weight_decay = float(ckpt.get('weight_decay', WEIGHT_DECAY))
        self.device = torch.device(device)
        self.net = AlphaZeroConvNet(self.board_shape, self.action_size, self.channels, self.num_res_blocks).to(self.device)
        self.net.load_state_dict(ckpt['model_state_dict'])
        opt_type = ckpt.get('optimizer_type', 'AdamW')
        if opt_type == 'AdamW':
            self.optimizer = optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
        print(f"Loaded: {filepath} (opt={opt_type})")

    @classmethod
    def load_from_file(cls, filepath: str, device: str = 'cpu') -> "AlphaZeroModel":
        ckpt = torch.load(filepath, map_location=device)
        model = cls(
            board_shape=tuple(ckpt['board_shape']),
            action_size=int(ckpt['action_size']),
            channels=int(ckpt['channels']),
            num_res_blocks=int(ckpt['num_res_blocks']),
            learning_rate=float(ckpt['learning_rate']),
            weight_decay=float(ckpt.get('weight_decay', WEIGHT_DECAY)),
            device=device
        )
        model.net.load_state_dict(ckpt['model_state_dict'])
        opt_type = ckpt.get('optimizer_type', 'AdamW')
        if opt_type == 'AdamW':
            model.optimizer = optim.AdamW(model.net.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)
        else:
            model.optimizer = optim.Adam(model.net.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)
        model.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model.scheduler = optim.lr_scheduler.StepLR(model.optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
        return model


# ======================
# Self-play worker (MP)
# ======================

def _play_game_worker(args):
    """Module-level for pickling."""
    game_class, model_state, mcts_simulations, iteration, board_shape, action_size, channels, num_res_blocks = args

    model = AlphaZeroModel(board_shape=board_shape, action_size=action_size, channels=channels, num_res_blocks=num_res_blocks)
    model.net.load_state_dict(model_state)
    model.net.eval()

    game = game_class()
    mcts = AlphaZeroMCTS(game, model, num_simulations=mcts_simulations, add_dirichlet=True)

    examples = []
    move_idx = 0
    while not game.over():
        cur_player = game.current_player
        temp = (TEMPERATURE_FINAL if iteration >= TEMPERATURE_ANNEAL_ITER
                else (TEMPERATURE_INITIAL if move_idx < TEMPERATURE_MOVE_THRESHOLD else TEMPERATURE_FINAL))
        actions, probs = mcts.get_action_probabilities(game.state(), temperature=temp)
        examples.append([(game.state(), cur_player), dict(zip(actions, probs)), None, cur_player])
        action = int(np.random.choice(actions, p=probs))
        game.make_move(action)
        move_idx += 1

    result = game.score()
    for ex in examples:
        ex[2] = result * ex[3]
    return [[ex[0], ex[1], ex[2]] for ex in examples]


# =================
# Trainer wrapper
# =================

class AlphaZeroTrainer:
    def __init__(self, game_class, model: AlphaZeroModel, mcts_simulations=N_SIMULATIONS):
        self.game_class = game_class
        self.model = model
        self.mcts_simulations = mcts_simulations

    def _augment_examples(self, examples):
        if not SYMMETRY_AUGMENT:
            return examples
        mappings = get_cube_symmetry_mappings(include_reflections=SYMM_INCLUDE_REFLECTIONS)
        augmented = []
        for state, action_probs, value in examples:
            # state is (board_tuple, player)
            if isinstance(state, (tuple, list)) and len(state) == 2:
                board_tuple, player = state
            else:
                board_tuple, player = state, 1
            base_board = list(board_tuple)

            if isinstance(action_probs, dict):
                # sparse input policy
                base_policy = np.zeros(ACTION_SIZE, dtype=np.float32)
                for k, p in action_probs.items():
                    base_policy[int(k)] = float(p)
            else:
                base_policy = np.array(action_probs, dtype=np.float32)

            for m in mappings:
                new_board = [0] * ACTION_SIZE
                for i, val in enumerate(base_board):
                    new_board[m[i]] = val
                # map policy indices
                new_policy_dict = {m[i]: float(base_policy[i]) for i in range(ACTION_SIZE) if base_policy[i] > 0.0}
                augmented.append(((tuple(new_board), player), new_policy_dict, value))
        return augmented

    def self_play_game(self, iteration: int):
        g = self.game_class()
        mcts = AlphaZeroMCTS(g, self.model, num_simulations=self.mcts_simulations, add_dirichlet=True)
        examples = []
        move_idx = 0
        while not g.over():
            cur_player = g.current_player
            temp = (TEMPERATURE_FINAL if iteration >= TEMPERATURE_ANNEAL_ITER
                    else (TEMPERATURE_INITIAL if move_idx < TEMPERATURE_MOVE_THRESHOLD else TEMPERATURE_FINAL))
            actions, probs = mcts.get_action_probabilities(g.state(), temperature=temp)
            examples.append([(g.state(), cur_player), dict(zip(actions, probs)), None, cur_player])
            action = int(np.random.choice(actions, p=probs))
            g.make_move(action)
            move_idx += 1
        result = g.score()
        for ex in examples:
            ex[2] = result * ex[3]
        return [[ex[0], ex[1], ex[2]] for ex in examples]

    def train_iteration(self, iteration: int, num_games=N_GAMES, num_workers=None):
        print(f"=== Iteration {iteration} | self-play {num_games} games ===")
        if num_workers is None:
            try:
                num_workers = max(1, os.cpu_count() - 1)
            except Exception:
                num_workers = 1

        all_examples = []
        model_state = self.model.net.state_dict()
        if num_workers > 1:
            args = [
                (self.game_class, model_state, self.mcts_simulations, iteration,
                 self.model.board_shape, self.model.action_size, self.model.channels, self.model.num_res_blocks)
                for _ in range(num_games)
            ]
            with Pool(processes=num_workers) as pool:
                iterator = pool.imap_unordered(_play_game_worker, args)
                if SHOW_PROGRESS:
                    iterator = tqdm(iterator, total=num_games, desc="Self-play (MP)")
                for res in iterator:
                    all_examples.extend(res)
        else:
            for _ in progress(range(num_games), desc="Self-play"):
                all_examples.extend(self.self_play_game(iteration))

        augmented = self._augment_examples(all_examples)
        for s, a_probs, v in augmented:
            self.model.add_training_data(s, a_probs, v)

        print("Training...")
        self.model.train()
        print(f"Examples: raw={len(all_examples)} augmented={len(augmented)}")

    def evaluate_match(self, opponent_model: AlphaZeroModel, num_games=16) -> Tuple[int, int, int]:
        wins = draws = losses = 0
        for gnum in range(num_games):
            game = self.game_class()
            first_is_current = (gnum % 2 == 0)
            mcts_A = AlphaZeroMCTS(game, self.model, num_simulations=EVAL_SIMULATIONS, add_dirichlet=False)
            mcts_B = AlphaZeroMCTS(game, opponent_model, num_simulations=EVAL_SIMULATIONS, add_dirichlet=False)
            players = [mcts_A, mcts_B] if first_is_current else [mcts_B, mcts_A]
            pid = 0
            while not game.over():
                actions, probs = players[pid].get_action_probabilities(game.state(), temperature=0.0)
                action = actions[int(np.argmax(probs))]
                game.make_move(action)
                pid = 1 - pid
            r = game.score()
            if r == 0:
                draws += 1
            elif first_is_current:
                wins += 1 if r > 0 else 0
                losses += 1 if r < 0 else 0
            else:
                wins += 1 if r < 0 else 0
                losses += 1 if r > 0 else 0
        return wins, draws, losses


# =======================
# Elo tournament utilities
# =======================

def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def update_elo(r_a: float, r_b: float, score_a: float, k: float) -> Tuple[float, float]:
    ea = expected_score(r_a, r_b)
    eb = expected_score(r_b, r_a)
    r_a_new = r_a + k * (score_a - ea)
    r_b_new = r_b + k * ((1.0 - score_a) - eb)
    return r_a_new, r_b_new

def play_single_deterministic(game_class, model_x: AlphaZeroModel, model_o: AlphaZeroModel,
                              sims: int, store_moves: bool = False) -> Tuple[int, List[int]]:
    game = game_class()
    moves = []
    mcts_X = AlphaZeroMCTS(game, model_x, num_simulations=sims, add_dirichlet=False)
    mcts_O = AlphaZeroMCTS(game, model_o, num_simulations=sims, add_dirichlet=False)
    while not game.over():
        cp = game.current_player
        mcts = mcts_X if cp == 1 else mcts_O
        actions, probs = mcts.get_action_probabilities(game.state(), temperature=0.0)
        action = actions[int(np.argmax(probs))]
        game.make_move(action)
        if store_moves:
            moves.append(action)
    return game.score(), moves

def run_elo_tournament(model_paths: List[str],
                       board_shape=BOARD_SHAPE, action_size=ACTION_SIZE,
                       channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS,
                       games_per_pair=EVAL_GAMES_PER_PAIR,
                       mcts_simulations=EVAL_SIMULATIONS,
                       k_factor=K_FACTOR,
                       history_csv=os.path.join(SAVE_DIR, "tournament_history.csv"),
                       ratings_csv=os.path.join(SAVE_DIR, "tournament_ratings.csv"),
                       seed: int = 0,
                       store_moves: bool = False,
                       game_class=Qubic4x4x4):
    np.random.seed(seed)

    models: Dict[str, AlphaZeroModel] = {}
    for p in model_paths:
        try:
            models[p] = AlphaZeroModel.load_from_file(p)
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")

    keys = list(models.keys())
    if len(keys) < 2:
        print("Tournament skipped: need >= 2 models.")
        return

    elos = {p: 1000.0 for p in keys}

    os.makedirs(os.path.dirname(history_csv), exist_ok=True)
    if not os.path.exists(history_csv):
        with open(history_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp','game_id','model_x','model_o','winner','result_score_x','result_score_o',
                'elo_x_before','elo_o_before','elo_x_after','elo_o_after','k_factor','moves'
            ])

    game_id = 1
    for a, b in progress(itertools.combinations(keys, 2), desc="Pairs"):
        print(f"[Elo] {os.path.basename(a)} vs {os.path.basename(b)}")
        game_range = progress(range(games_per_pair), desc="Games", leave=False)
        for g in game_range:
            a_is_x = (g % 2 == 0)
            mx = models[a] if a_is_x else models[b]
            mo = models[b] if a_is_x else models[a]
            px = a if a_is_x else b
            po = b if a_is_x else a

            elo_x_before = elos[px]
            elo_o_before = elos[po]

            score, moves = play_single_deterministic(game_class, mx, mo, mcts_simulations, store_moves=store_moves)
            s_x = 1.0 if score == 1 else (0.5 if score == 0 else 0.0)
            elo_x_after, elo_o_after = update_elo(elo_x_before, elo_o_before, s_x, k_factor)
            elos[px], elos[po] = elo_x_after, elo_o_after

            winner = 'X' if score == 1 else ('O' if score == -1 else 'D')
            row = [
                datetime.utcnow().isoformat(), str(game_id), px, po, winner, str(score), str(-score),
                f"{elo_x_before:.2f}", f"{elo_o_before:.2f}", f"{elo_x_after:.2f}", f"{elo_o_after:.2f}",
                f"{k_factor}", ';'.join(map(str, moves)) if store_moves else ''
            ]
            with open(history_csv, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            game_id += 1

    with open(ratings_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'model_path', 'elo'])
        for p, r in sorted(elos.items(), key=lambda x: -x[1]):
            writer.writerow([os.path.basename(p), p, f"{r:.2f}"])
    print(f"[Elo] Tournament complete. Ratings written to {ratings_csv}")


# ================
# Training script
# ================

def train_alphazero_4x4x4():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("=== AlphaZero 4×4×4 (Qubic) ===")

    model = AlphaZeroModel(board_shape=BOARD_SHAPE, action_size=ACTION_SIZE,
                           channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS, learning_rate=LEARNING_RATE)
    trainer = AlphaZeroTrainer(Qubic4x4x4, model, mcts_simulations=N_SIMULATIONS)

    base_path = os.path.join(SAVE_DIR, "alphazero_iter_0.pth")
    model.save_model(base_path)

    for it in progress(range(1, N_ITERATIONS + 1), desc="Iterations"):
        trainer.train_iteration(it, num_games=N_GAMES)

        if it % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"alphazero_iter_{it}.pth")
            model.save_model(ckpt_path)

        if it % EVAL_ELO_INTERVAL == 0:
            files = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".pth")]
            files = sorted(files, key=lambda p: os.path.getmtime(p))
            print(f"[Elo] Running tournament on {len(files)} models...")
            run_elo_tournament(
                model_paths=files,
                board_shape=BOARD_SHAPE,
                action_size=ACTION_SIZE,
                channels=CHANNELS,
                num_res_blocks=NUM_RES_BLOCKS,
                games_per_pair=EVAL_GAMES_PER_PAIR,
                mcts_simulations=EVAL_SIMULATIONS,
                k_factor=K_FACTOR,
                history_csv=os.path.join(SAVE_DIR, "tournament_history.csv"),
                ratings_csv=os.path.join(SAVE_DIR, "tournament_ratings.csv"),
                seed=0,
                store_moves=False,
                game_class=Qubic4x4x4
            )

    final_path = os.path.join(SAVE_DIR, "alphazero_final.pth")
    model.save_model(final_path)
    print(f"Done. Final model: {final_path}")


if __name__ == "__main__":
    # Optional CLI overrides via env
    N_ITERATIONS = int(os.getenv("N_ITERATIONS", N_ITERATIONS))
    N_GAMES = int(os.getenv("N_GAMES", N_GAMES))
    N_SIMULATIONS = int(os.getenv("N_SIMULATIONS", N_SIMULATIONS))
    EVAL_SIMULATIONS = int(os.getenv("EVAL_SIMULATIONS", EVAL_SIMULATIONS))
    SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", SAVE_INTERVAL))
    EVAL_ELO_INTERVAL = int(os.getenv("EVAL_ELO_INTERVAL", EVAL_ELO_INTERVAL))
    EVAL_GAMES_PER_PAIR = int(os.getenv("EVAL_GAMES_PER_PAIR", EVAL_GAMES_PER_PAIR))
    SYMMETRY_AUGMENT = bool(int(os.getenv("SYMMETRY_AUGMENT", "1")))
    SYMM_INCLUDE_REFLECTIONS = bool(int(os.getenv("SYMM_INCLUDE_REFLECTIONS", "0")))
    train_alphazero_4x4x4()