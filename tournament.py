#!/usr/bin/env python3
"""Tournament evaluator for multiple AlphaZero Tic-Tac-Toe models.

Features:
- Round-robin tournament among provided model checkpoints
- Elo rating updates after every game (configurable K-factor, initial Elo)
- Alternating starting player (X / O) for fairness
- Deterministic move selection via MCTS with temperature=0 (argmax policy)
- Game history appended to CSV (resume supported)
- Separate ratings CSV regenerated after run
- Optional storage of move sequence for each game

Example:
    python tournament.py \
        --models models/alphazero_iter_0.pth models/alphazero_iter_160.pth \
        --games-per-pair 20 --mcts-simulations 200 --history-csv models/tournament_history.csv

Resume:
    python tournament.py --resume --history-csv models/tournament_history.csv \
        --models models/alphazero_iter_0.pth models/alphazero_iter_160.pth --games-per-pair 10

CSV Schemas:
History CSV columns:
    timestamp,game_id,model_x,model_o,winner,result_score_x,result_score_o,\
    elo_x_before,elo_o_before,elo_x_after,elo_o_after,starting_player,moves
    - winner: X, O, or D
    - result_score_x: +1 / -1 / 0 (game.score())
    - moves: semicolon-separated move indices (optional if --store-moves)

Ratings CSV columns:
    model_name,model_path,elo

"""
from __future__ import annotations
import argparse
import csv
import itertools
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

try:
    from main import TicTacToeGame, AlphaZeroModel, AlphaZeroMCTS
except Exception:  # Fallback / clearer error
    print("Failed to import from main.py. Ensure you're running from repo root.")
    raise

# ---------------- Elo Utilities ---------------- #

def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def update_elo(r_a: float, r_b: float, score_a: float, k: float) -> Tuple[float, float]:
    ea = expected_score(r_a, r_b)
    eb = expected_score(r_b, r_a)
    r_a_new = r_a + k * (score_a - ea)
    r_b_new = r_b + k * ((1.0 - score_a) - eb)
    return r_a_new, r_b_new

# ---------------- Tournament Logic ---------------- #

def load_models(model_paths: List[str]) -> Dict[str, AlphaZeroModel]:
    models: Dict[str, AlphaZeroModel] = {}
    for path in model_paths:
        if not os.path.exists(path):
            print(f"[WARN] Model path does not exist: {path}")
            continue
        try:
            models[path] = AlphaZeroModel.load_from_file(path)
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
    return models

def init_elos(model_paths: List[str], initial_elo: float) -> Dict[str, float]:
    return {p: initial_elo for p in model_paths}

def read_history(history_csv: str) -> List[Dict[str, str]]:
    if not os.path.exists(history_csv):
        return []
    rows = []
    with open(history_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def recompute_elos_from_history(rows: List[Dict[str, str]], initial_elo: float) -> Dict[str, float]:
    elos: Dict[str, float] = {}
    for row in rows:
        mx = row['model_x']
        mo = row['model_o']
        if mx not in elos:
            elos[mx] = initial_elo
        if mo not in elos:
            elos[mo] = initial_elo
        # Elo before (string) -> float
        # Could recompute using result to avoid trusting stored elo
        score_x = float(row['result_score_x'])  # +1 / -1 / 0
        # Map to 1/0.5/0 for model_x
        if score_x == 1:
            s_x = 1.0
        elif score_x == 0:
            s_x = 0.5
        else:  # -1
            s_x = 0.0
        # We need to reverse update that produced stored values; simpler: just apply new update
        k = float(row.get('k_factor', '0')) or 32.0  # fallback if not stored
        elos[mx], elos[mo] = update_elo(elos[mx], elos[mo], s_x, k)
    return elos

def write_history_header_if_needed(history_csv: str):
    if not os.path.exists(history_csv):
        with open(history_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp','game_id','model_x','model_o','winner','result_score_x','result_score_o',
                'elo_x_before','elo_o_before','elo_x_after','elo_o_after','k_factor','starting_player','moves'
            ])

def write_ratings_csv(ratings_csv: str, elos: Dict[str, float]):
    with open(ratings_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_name','model_path','elo'])
        for path, elo in sorted(elos.items(), key=lambda x: -x[1]):
            writer.writerow([os.path.basename(path), path, f"{elo:.2f}"])

def play_single_game(model_x: AlphaZeroModel, model_o: AlphaZeroModel, mcts_simulations: int, store_moves: bool=False) -> Tuple[int, List[int]]:
    """Play one deterministic game: return game.score() and move list."""
    game = TicTacToeGame()
    moves: List[int] = []
    while not game.over():
        current_player = game.current_player  # 1 (X) or -1 (O)
        model = model_x if current_player == 1 else model_o
        mcts = AlphaZeroMCTS(game, model, num_simulations=mcts_simulations)
        actions, probs = mcts.get_action_probabilities(game.state(), temperature=0.0)
        # Choose argmax
        best_idx = int(np.argmax(probs))
        action = actions[best_idx]
        game.make_move(action)
        if store_moves:
            moves.append(action)
    return game.score(), moves

def run_tournament(model_paths: List[str], games_per_pair: int, mcts_simulations: int, k_factor: float,
                   initial_elo: float, history_csv: str, ratings_csv: str, seed: int|None, resume: bool,
                   store_moves: bool) -> None:
    if seed is not None:
        np.random.seed(seed)

    model_paths = list(dict.fromkeys(model_paths))  # dedupe preserve order
    models = load_models(model_paths)
    if len(models) < 2:
        print("Need at least two valid models to run a tournament.")
        return

    # History handling
    existing_rows = read_history(history_csv) if resume else []
    write_history_header_if_needed(history_csv)

    if resume and existing_rows:
        elos = recompute_elos_from_history(existing_rows, initial_elo)
        game_id_start = max(int(r['game_id']) for r in existing_rows) + 1
        print(f"Resuming: loaded {len(existing_rows)} past games. Starting at game_id {game_id_start}.")
    else:
        elos = init_elos(model_paths, initial_elo)
        game_id_start = 1

    # Add any new models not in existing Elo dict
    for p in model_paths:
        elos.setdefault(p, initial_elo)

    total_pairs = 0
    total_games = 0

    # Pair schedule (round robin)
    pairs = list(itertools.combinations(model_paths, 2))
    for a_path, b_path in pairs:
        total_pairs += 1
        print(f"\n=== Pair {total_pairs}/{len(pairs)}: {os.path.basename(a_path)} vs {os.path.basename(b_path)} ===")

        model_a = models[a_path]
        model_b = models[b_path]

        for g in range(games_per_pair):
            game_id = game_id_start + total_games
            # Alternate starting player: even -> A is X, odd -> B is X
            a_is_x = (g % 2 == 0)
            model_x = model_a if a_is_x else model_b
            model_o = model_b if a_is_x else model_a
            path_x = a_path if a_is_x else b_path
            path_o = b_path if a_is_x else a_path

            elo_x_before = elos[path_x]
            elo_o_before = elos[path_o]

            score, moves = play_single_game(model_x, model_o, mcts_simulations, store_moves=store_moves)
            # score: +1 if X wins, -1 if O wins, 0 draw
            if score == 1:
                winner = 'X'
            elif score == -1:
                winner = 'O'
            else:
                winner = 'D'

            # Map score for model_x to Elo result (1 win / 0.5 draw / 0 loss)
            if score == 1:
                s_x = 1.0
            elif score == 0:
                s_x = 0.5
            else:  # -1
                s_x = 0.0

            elo_x_after, elo_o_after = update_elo(elo_x_before, elo_o_before, s_x, k_factor)
            elos[path_x] = elo_x_after
            elos[path_o] = elo_o_after

            row = {
                'timestamp': datetime.utcnow().isoformat(),
                'game_id': str(game_id),
                'model_x': path_x,
                'model_o': path_o,
                'winner': winner,
                'result_score_x': str(score),
                'result_score_o': str(-score),
                'elo_x_before': f"{elo_x_before:.2f}",
                'elo_o_before': f"{elo_o_before:.2f}",
                'elo_x_after': f"{elo_x_after:.2f}",
                'elo_o_after': f"{elo_o_after:.2f}",
                'k_factor': f"{k_factor}",
                'starting_player': 'X' if a_is_x else ('X' if not a_is_x else 'X'),  # always X starts
                'moves': ';'.join(map(str, moves)) if store_moves else ''
            }
            # Append to history CSV
            with open(history_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)

            total_games += 1

            print(f"Game {g+1}/{games_per_pair} | Winner: {winner} | Score: {score} | Elo X: {row['elo_x_after']} | Elo O: {row['elo_o_after']}")

    # Write ratings summary
    write_ratings_csv(ratings_csv, elos)
    print(f"\nTournament complete. {total_games} games played. Ratings written to {ratings_csv}")

# ---------------- CLI ---------------- #

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaZero Tic-Tac-Toe tournament evaluator (Elo ratings)")
    p.add_argument('--models', nargs='+', help='Explicit list of model checkpoint paths (.pth)')
    p.add_argument('--models-dir', nargs='+', help='One or more directories to scan for model checkpoints (.pth)')
    p.add_argument('--recursive', action='store_true', help='Recursively scan provided --models-dir directories')
    p.add_argument('--sort', choices=['name','mtime','iteration'], default='iteration', help='Sorting strategy for discovered models')
    p.add_argument('--limit', type=int, default=None, help='Optional limit on number of discovered models after sorting')
    p.add_argument('--games-per-pair', type=int, default=10, help='Number of games per model pair')
    p.add_argument('--mcts-simulations', type=int, default=100, help='MCTS simulations per move')
    p.add_argument('--k-factor', type=float, default=32.0, help='Elo K-factor')
    p.add_argument('--initial-elo', type=float, default=1000.0, help='Initial Elo for new models')
    p.add_argument('--history-csv', type=str, default='models/tournament_history.csv', help='CSV file to append game history')
    p.add_argument('--ratings-csv', type=str, default='models/tournament_ratings.csv', help='CSV file to write final Elo ratings')
    p.add_argument('--seed', type=int, default=None, help='Random seed (for reproducibility if randomness added later)')
    p.add_argument('--resume', action='store_true', help='Resume from existing history CSV (recompute Elo)')
    p.add_argument('--store-moves', action='store_true', help='Store move sequence in history CSV')
    return p.parse_args(argv)


def main(argv: List[str]):
    args = parse_args(argv)
    os.makedirs(os.path.dirname(args.history_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.ratings_csv), exist_ok=True)

    # Auto-discovery if models not explicitly supplied
    model_paths: List[str] = []
    if args.models:
        model_paths.extend(args.models)
    if args.models_dir:
        for d in args.models_dir:
            if not os.path.isdir(d):
                print(f"[WARN] Not a directory: {d}")
                continue
            if args.recursive:
                for root, _, files in os.walk(d):
                    for f in files:
                        if f.endswith('.pth'):
                            model_paths.append(os.path.join(root, f))
            else:
                for f in os.listdir(d):
                    full = os.path.join(d, f)
                    if os.path.isfile(full) and f.endswith('.pth'):
                        model_paths.append(full)
    if not model_paths:
        print("No models provided or discovered. Use --models or --models-dir.")
        return
    # Deduplicate
    model_paths = list(dict.fromkeys(model_paths))

    # Sorting strategies
    def extract_iteration(path: str) -> int:
        base = os.path.basename(path)
        # Expect pattern alphazero_iter_<N>.pth
        for token in base.replace('.', '_').split('_'):
            if token.isdigit():
                return int(token)
        return -1
    if args.sort == 'name':
        model_paths.sort(key=lambda p: os.path.basename(p))
    elif args.sort == 'mtime':
        model_paths.sort(key=lambda p: os.path.getmtime(p))
    elif args.sort == 'iteration':
        model_paths.sort(key=extract_iteration)
    if args.limit is not None:
        model_paths = model_paths[:args.limit]
    print(f"Discovered {len(model_paths)} models for tournament.")
    for p in model_paths:
        print(f" - {p}")

    run_tournament(
        model_paths=model_paths,
        games_per_pair=args.games_per_pair,
        mcts_simulations=args.mcts_simulations,
        k_factor=args.k_factor,
        initial_elo=args.initial_elo,
        history_csv=args.history_csv,
        ratings_csv=args.ratings_csv,
        seed=args.seed,
        resume=args.resume,
        store_moves=args.store_moves,
    )


if __name__ == '__main__':
    main(sys.argv[1:])
