import numpy as np
from qubic import (
    ACTION_SIZE,
    AlphaZeroMCTS,
    MCTSNode,
    Qubic4x4x4,
    get_cube_symmetry_mappings,
    to_coords,
    to_index,
)


class DummyModel:
    """Policy stub that concentrates probability mass on a chosen action."""

    def __init__(self, winning_action: int, action_size: int = ACTION_SIZE):
        self.winning_action = winning_action
        self.action_size = action_size

    def encode_state(self, game):
        if hasattr(game, "board") and hasattr(game, "current_player"):
            board = np.asarray(game.board, dtype=np.int8).reshape(4, 4, 4)
            cur = int(game.current_player)
        else:
            state, cur = game
            board = np.asarray(state, dtype=np.int8).reshape(4, 4, 4)
            cur = int(cur)
        current_plane = (board == cur).astype(np.float32)
        opponent_plane = (board == -cur).astype(np.float32)
        return np.stack([current_plane, opponent_plane], axis=0)

    def predict(self, game):
        value = 0.0
        policy = np.zeros(self.action_size, dtype=np.float32)
        policy[self.winning_action] = 1.0
        return value, policy


def test_to_coords_round_trip():
    for idx in range(ACTION_SIZE):
        coords = to_coords(idx)
        assert to_index(*coords) == idx
        assert all(0 <= c < 4 for c in coords)


def test_win_lines_count_and_length():
    game = Qubic4x4x4()
    lines = game._WIN_LINES
    assert len(lines) == 76
    for line in lines:
        assert len(line) == 4
        assert len({to_coords(i) for i in line}) == 4


def test_winner_detects_axis_line():
    game = Qubic4x4x4()
    line = [to_index(x, 0, 0) for x in range(4)]
    for idx in line:
        game.board[idx] = 1
    assert game._winner() == 1
    assert game.score() == 1
    assert game.over() is True


def test_symmetry_mapping_counts():
    rotations = get_cube_symmetry_mappings(include_reflections=False)
    reflections = get_cube_symmetry_mappings(include_reflections=True)
    assert len(rotations) == 24
    assert len(reflections) == 48
    for mapping in rotations + reflections:
        assert sorted(mapping) == list(range(ACTION_SIZE))


def test_mcts_terminal_orientation_qubic():
    game = Qubic4x4x4()
    winning_line = [to_index(x, 0, 0) for x in range(4)]
    for idx in winning_line[:3]:
        game.board[idx] = 1
    game.current_player = 1
    winning_action = winning_line[3]

    dummy_model = DummyModel(winning_action)
    mcts = AlphaZeroMCTS(game, dummy_model, num_simulations=1, add_dirichlet=False)
    root = mcts.search(game.state())

    child = root.children.get(winning_action)
    assert child is not None
    assert np.isclose(child.value(), -1.0)
    assert np.isclose(root.value(), 1.0)


def test_backup_sign_qubic():
    leaf_value = 1.0
    root = MCTSNode(state=None)
    child = MCTSNode(state=None, parent=root, action=0, prior=1.0)
    child.backup(leaf_value)
    assert np.isclose(child.value(), 1.0)
    assert np.isclose(root.value(), -1.0)


def run_all():
    tests = [
        test_to_coords_round_trip,
        test_win_lines_count_and_length,
        test_winner_detects_axis_line,
        test_symmetry_mapping_counts,
        test_mcts_terminal_orientation_qubic,
        test_backup_sign_qubic,
    ]
    for fn in tests:
        fn()
    print("All Qubic sanity checks passed.")


if __name__ == "__main__":
    run_all()
