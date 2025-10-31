#!/usr/bin/env python3
"""
Interactive Tic-Tac-Toe game against AlphaZero AI
"""
import numpy as np
from main import TicTacToeGame, AlphaZeroModel, AlphaZeroMCTS
# from fix_training import ImprovedAlphaZeroModel
import os

def print_board(game):
    """Print the current board state"""
    board = np.array(game.board).reshape(3, 3)
    symbols = {1: 'X', -1: 'O', 0: ' '}
    
    print("\n  0   1   2")
    for i in range(3):
        print(f"{i} {symbols[board[i,0]]} | {symbols[board[i,1]]} | {symbols[board[i,2]]}")
        if i < 2:
            print("  --|---|--")
    print()

def get_human_move(game):
    """Get a move from the human player"""
    valid_moves = game.valid_moves()
    
    while True:
        try:
            print(f"Valid moves: {valid_moves}")
            move = input("Enter your move (row,col) or position (0-8): ").strip()
            
            if ',' in move:
                # Row, column format
                row, col = map(int, move.split(','))
                position = row * 3 + col
            else:
                # Single position format
                position = int(move)
            
            if position in valid_moves:
                return position
            else:
                print(f"Invalid move! Position {position} is not available.")
                
        except (ValueError, IndexError):
            print("Invalid input! Please enter a number 0-8 or row,col format (e.g., '1,2')")

def get_ai_move(game, model, mcts_simulations=100):
    """Get a move from the AI using MCTS"""
    mcts = AlphaZeroMCTS(game, model, num_simulations=mcts_simulations)
    actions, probs = mcts.get_action_probabilities(game.state(), temperature=0.0)
    
    # Choose the action with highest probability
    best_action_idx = np.argmax(probs)
    action = actions[best_action_idx]
    
    print(f"AI chooses position {action}")
    return action

def play_game(model_path="models/alphazero_iter_10.pth", human_first=True, mcts_simulations=100):
    """Play an interactive game against the AI"""
    
    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found! Please train the model first by running main.py")
        return
    
    model = AlphaZeroModel.load_from_file(model_path)
    # model = ImprovedAlphaZeroModel.load_from_file(model_path)
    
    # Initialize game
    game = TicTacToeGame()
    
    # Determine player assignments
    if human_first:
        human_player = 1  # X
        ai_player = -1    # O
        print("You are X, AI is O. You go first!")
    else:
        human_player = -1  # O
        ai_player = 1     # X
        print("AI is X, you are O. AI goes first!")
    
    print_board(game)
    
    # Game loop
    while not game.over():
        if game.current_player == human_player:
            # Human turn
            print("Your turn!")
            move = get_human_move(game)
            game.make_move(move)
        else:
            # AI turn
            print("AI is thinking...")
            move = get_ai_move(game, model, mcts_simulations)
            game.make_move(move)
        
        print_board(game)
    
    # Game over - show result
    result = game.score()
    if result == 0:
        print("It's a draw!")
    elif result == human_player:
        print("Congratulations! You won!")
    else:
        print("AI wins! Better luck next time.")

def main():
    """Main interactive menu"""
    print("=== AlphaZero Tic-Tac-Toe ===")
    print("Welcome to Tic-Tac-Toe against AlphaZero AI!")
    
    while True:
        print("\nOptions:")
        print("1. Play as X (you go first)")
        print("2. Play as O (AI goes first)")
        print("3. Quit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            play_game(human_first=True)
        elif choice == '2':
            play_game(human_first=False)
        elif choice == '3':
            print("Thanks for playing!")
            break
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")
        
        # Ask if they want to play again
        if choice in ['1', '2']:
            play_again = input("\nWould you like to play again? (y/n): ").strip().lower()
            if play_again != 'y':
                print("Thanks for playing!")
                break

if __name__ == "__main__":
    main()