Read:

MCTS
https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/
https://mcts.ai/about/index.html

AlphaZero
https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/
https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0
https://suragnair.github.io/posts/alphazero.html
https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ

## Progress Bars (tqdm)

Set the environment variable `PROGRESS=1` to enable tqdm progress bars during training (`qubic.py`) and Elo tournaments. Example:

```bash
PROGRESS=1 uv run qubic.py
```

Bars shown:
* Iterations loop
* Self-play games (single-process) or multiprocessing aggregate
* MCTS simulations (only when `num_simulations >= 50` to avoid excessive output)
* Elo tournament pair and game loops

Unset (or `PROGRESS=0`) for clean, minimal console output.

## Tournament Elo Evaluation

Use `tournament.py` to run a round-robin Elo tournament between multiple saved checkpoints.

### Basic Usage

```bash
python tournament.py \
	--models models/alphazero_iter_0.pth models/alphazero_iter_160.pth models/alphazero_final.pth \
	--games-per-pair 20 \
	--mcts-simulations 200 \
	--history-csv models/tournament_history.csv \
	--ratings-csv models/tournament_ratings.csv \
	--store-moves
```

### Resume From Previous History

If you already have a `tournament_history.csv`, you can resume (adding more games and recalculating Elo from scratch) with:

```bash
python tournament.py --resume \
	--models models/alphazero_iter_0.pth models/alphazero_iter_160.pth models/alphazero_final.pth \
	--games-per-pair 10 \
	--history-csv models/tournament_history.csv \
	--ratings-csv models/tournament_ratings.csv
```

### CSV Outputs

`tournament_history.csv` columns:
```
timestamp,game_id,model_x,model_o,winner,result_score_x,result_score_o,elo_x_before,elo_o_before,elo_x_after,elo_o_after,k_factor,starting_player,moves
```
`winner` is `X`, `O`, or `D`, and `result_score_x` is +1 / -1 / 0 from TicTacToe scoring perspective. `moves` is a semicolon-separated list of positions (0-8) if `--store-moves` was used.

`tournament_ratings.csv` columns:
```
model_name,model_path,elo
```

### Notes
* Elo is updated after every game using the standard logistic expectancy and the chosen `--k-factor` (default 32).
* Starting player alternates each game within a pair to reduce first-move bias.
* Deterministic play uses MCTS with temperature=0 (argmax over visit counts). Increase `--mcts-simulations` for stronger play.
* On resume, Elo is recomputed from the full history so changing K-factor midstream will affect new updates only.
