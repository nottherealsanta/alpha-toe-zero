import numpy as np
from main import MCTSNode, TicTacToeGame, AlphaZeroMCTS

class DummyModel:
    def __init__(self, winning_action=2, action_size=9):
        self.winning_action = winning_action
        self.action_size = action_size
    def encode_state(self, game):
        # Minimal compatibility; not used for dummy forward
        if hasattr(game, 'board'):
            board = np.array(game.board).reshape(3,3)
            cur = game.current_player
            current_plane = (board == cur).astype(np.float32)
            opp_plane = (board == -cur).astype(np.float32)
            return np.stack([current_plane, opp_plane], axis=-1)
        return np.zeros((3,3,2), dtype=np.float32)
    def predict(self, game):
        # Value neutral, policy concentrated on winning_action
        value = 0.0
        policy = np.zeros(self.action_size, dtype=np.float32)
        policy[self.winning_action] = 1.0
        return value, policy


def test_backup_sign():
    leaf_v = +1.0
    root = MCTSNode(state=None)
    child = MCTSNode(state=None, parent=root, action=0, prior=1.0)
    child.backup(leaf_v)
    assert np.isclose(child.value(), +1.0), f"Child value expected +1.0, got {child.value()}"
    assert np.isclose(root.value(), -1.0), f"Root value expected -1.0, got {root.value()}"


def test_selection_sign():
    root = MCTSNode(state=None)
    root.visit_count = 10
    a = MCTSNode(state=None, parent=root, action=0, prior=0.5)
    a.visit_count = 10
    a.value_sum = +5   # child view +0.5 => parent Q=-0.5
    b = MCTSNode(state=None, parent=root, action=1, prior=0.5)
    b.visit_count = 10
    b.value_sum = -5   # child view -0.5 => parent Q=+0.5
    root.children = {0: a, 1: b}
    picked = root.select_child(c_puct=0.0)
    assert picked is b, "Selection should prefer higher parent-view Q (node b)"


def test_terminal_orientation():
    g = TicTacToeGame()
    # Board set so action 2 by X wins the top row
    g.board = [1,1,0, 0,-1,0, 0,0,-1]
    g.current_player = 1  # X to move
    dummy_model = DummyModel(winning_action=2)
    mcts = AlphaZeroMCTS(g, dummy_model, num_simulations=1)
    root = mcts.search(g.state())
    # After one simulation, the winning child should have been explored and backed up
    child = root.children.get(2)
    assert child is not None, "Winning action child (2) should exist"
    # Compute expected terminal value: score()*current_player AFTER move -> current_player flips to -1
    # After move, score()=+1, current_player=-1 => terminal_v=-1 propagated at leaf
    expected_terminal_v = -1.0
    assert np.isclose(child.value(), expected_terminal_v), f"Leaf value {child.value()} != expected {expected_terminal_v}"
    assert np.isclose(root.value(), -expected_terminal_v), f"Root value {root.value()} != expected {-expected_terminal_v}"


def run_all():
    test_backup_sign()
    test_selection_sign()
    test_terminal_orientation()
    print("All sanity tests passed.")

if __name__ == "__main__":
    run_all()
