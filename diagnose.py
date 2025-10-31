#!/usr/bin/env python3
"""
Diagnostic script to analyze AlphaZero model performance
"""
from main import TicTacToeGame, AlphaZeroModel, AlphaZeroMCTS
import numpy as np

def test_basic_positions():
    """Test model on basic tic-tac-toe positions"""
    model = AlphaZeroModel.load_from_file("models/alphazero_final.pth")
    
    print("=== Testing Basic Positions ===\n")
    
    # Test 1: Win in one move
    print("Test 1: Win in one move")
    game = TicTacToeGame()
    game.board = [1, 1, 0, 0, 0, 0, 0, 0, 0]  # X can win at position 2
    game.current_player = 1
    
    print_board(game)
    value, policy = model.predict(game)
    print(f"Position value: {value:.3f}")
    print(f"Policy (should prefer position 2): {[f'{i}:{p:.3f}' for i, p in enumerate(policy)]}")
    
    # Test with MCTS
    mcts = AlphaZeroMCTS(game, model, num_simulations=100)
    actions, probs = mcts.get_action_probabilities(game.state(), temperature=0)
    best_action = actions[np.argmax(probs)]
    print(f"MCTS best move: {best_action} (should be 2)")
    print(f"MCTS action probs: {dict(zip(actions, probs))}")
    print()
    
    # Test 2: Block opponent win
    print("Test 2: Block opponent win")
    game = TicTacToeGame()
    game.board = [-1, -1, 0, 0, 0, 0, 0, 0, 0]  # O about to win, X must block at position 2
    game.current_player = 1
    
    print_board(game)
    value, policy = model.predict(game)
    print(f"Position value: {value:.3f}")
    print(f"Policy (should prefer position 2): {[f'{i}:{p:.3f}' for i, p in enumerate(policy)]}")
    
    mcts = AlphaZeroMCTS(game, model, num_simulations=100)
    actions, probs = mcts.get_action_probabilities(game.state(), temperature=0)
    best_action = actions[np.argmax(probs)]
    print(f"MCTS best move: {best_action} (should be 2)")
    print()
    
    # Test 3: Center play from empty board
    print("Test 3: Empty board opening")
    game = TicTacToeGame()
    
    print_board(game)
    value, policy = model.predict(game)
    print(f"Position value: {value:.3f}")
    print(f"Policy: {[f'{i}:{p:.3f}' for i, p in enumerate(policy)]}")
    
    mcts = AlphaZeroMCTS(game, model, num_simulations=100)
    actions, probs = mcts.get_action_probabilities(game.state(), temperature=0)
    best_action = actions[np.argmax(probs)]
    print(f"MCTS best move: {best_action} (center=4 is often good)")

def print_board(game):
    """Print the current board state"""
    board = np.array(game.board).reshape(3, 3)
    symbols = {1: 'X', -1: 'O', 0: '.'}
    
    print("Board:")
    for i in range(3):
        row = ' '.join(symbols[board[i,j]] for j in range(3))
        print(f"  {row}")
    print()

def test_random_games():
    """Test model against random play"""
    model = AlphaZeroModel.load_from_file("models/alphazero_final.pth")
    
    wins = 0
    draws = 0
    losses = 0
    
    print("=== Testing vs Random Player (100 games) ===")
    
    for game_num in range(100):
        game = TicTacToeGame()
        ai_is_x = game_num % 2 == 0  # Alternate who goes first
        
        while not game.over():
            if (game.current_player == 1 and ai_is_x) or (game.current_player == -1 and not ai_is_x):
                # AI turn
                mcts = AlphaZeroMCTS(game, model, num_simulations=50)
                actions, probs = mcts.get_action_probabilities(game.state(), temperature=0)
                action = actions[np.argmax(probs)]
                game.make_move(action)
            else:
                # Random turn
                valid_moves = game.valid_moves()
                action = np.random.choice(valid_moves)
                game.make_move(action)
        
        result = game.score()
        if result == 0:
            draws += 1
        elif (result == 1 and ai_is_x) or (result == -1 and not ai_is_x):
            wins += 1
        else:
            losses += 1
        
        if game_num % 20 == 19:
            print(f"After {game_num + 1} games: W={wins}, D={draws}, L={losses}")
    
    print(f"\nFinal results: Wins={wins}, Draws={draws}, Losses={losses}")
    print(f"Win rate: {wins/100:.1%}, Draw rate: {draws/100:.1%}, Loss rate: {losses/100:.1%}")

def test_perfect_play():
    """Test if model can find perfect moves in key positions"""
    model = AlphaZeroModel.load_from_file("models/alphazero_final.pth")
    
    print("=== Perfect Play Tests ===\n")
    
    # Test positions where there's a clearly best move
    test_positions = [
        {
            'name': 'Immediate win',
            'board': [1, 1, 0, -1, -1, 0, 0, 0, 0],  # X wins at 2, O wins at 5
            'player': 1,
            'best_moves': [2],
            'description': 'X should win immediately at position 2'
        },
        {
            'name': 'Block immediate loss',
            'board': [-1, -1, 0, 1, 0, 0, 0, 0, 0],  # O about to win at 2
            'player': 1,
            'best_moves': [2],
            'description': 'X must block O win at position 2'
        },
        {
            'name': 'Create fork',
            'board': [1, 0, 0, 0, 1, 0, 0, 0, -1],  # X can create fork
            'player': 1,
            'best_moves': [2, 6],  # Either creates a fork
            'description': 'X should create a winning fork'
        }
    ]
    
    for test in test_positions:
        print(f"Test: {test['name']}")
        print(f"Description: {test['description']}")
        
        game = TicTacToeGame()
        game.board = test['board'].copy()
        game.current_player = test['player']
        
        print_board(game)
        
        # Test with different MCTS simulation counts
        for sims in [50, 100, 200]:
            mcts = AlphaZeroMCTS(game, model, num_simulations=sims)
            actions, probs = mcts.get_action_probabilities(game.state(), temperature=0)
            best_action = actions[np.argmax(probs)]
            
            correct = best_action in test['best_moves']
            status = "✓" if correct else "✗"
            print(f"  {sims} sims: {best_action} {status}")
        
        print()

if __name__ == "__main__":
    test_basic_positions()
    print("\n" + "="*50 + "\n")
    test_perfect_play()
    print("\n" + "="*50 + "\n")
    test_random_games()