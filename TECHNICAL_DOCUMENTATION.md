# AlphaZero Tic-Tac-Toe: Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Training Pipeline](#training-pipeline)
5. [Detailed Flowcharts](#detailed-flowcharts)
6. [Hyperparameters](#hyperparameters)
7. [API Reference](#api-reference)

---

## Overview

This project implements a simplified version of DeepMind's AlphaZero algorithm for playing Tic-Tac-Toe. The system combines:
- **Monte Carlo Tree Search (MCTS)** for strategic planning
- **Deep Neural Networks** (ResNet architecture) for position evaluation and move prediction
- **Self-play reinforcement learning** for training without human data

## Key Features
- PyTorch-based deep neural network with residual blocks
- Multi-process parallel self-play for efficient training
- Comprehensive evaluation system with historical checkpoints
- Configurable hyperparameters for experimentation
- Temperature annealing for improved exploration/exploitation balance
- Data augmentation via dihedral symmetries (rotations and reflections)
- Replay buffer with capacity management
- Dirichlet noise at root node for exploration
- Legal move masking in policy head during training

---

## System Architecture

```mermaid
graph TB
    subgraph "AlphaZero System"
        A[Training Loop] --> B[Self-Play Games]
        B --> C[MCTS Search]
        C --> D[Neural Network]
        D --> C
        C --> E[Training Data]
        E --> F[Neural Network Training]
        F --> D
        A --> G[Evaluation]
        G --> H[Checkpoint Management]
    end
    
    subgraph "Neural Network"
        D --> I[Input Layer]
        I --> J[Residual Blocks]
        J --> K1[Value Head]
        J --> K2[Policy Head]
    end
    
    subgraph "MCTS Components"
        C --> L[Selection]
        L --> M[Expansion]
        M --> N[Evaluation]
        N --> O[Backup]
    end
```

---

## Core Components

### 1. MCTSNode
Represents a node in the Monte Carlo Tree Search.

**Attributes:**
- `state`: Current game state
- `parent`: Parent node reference
- `action`: Action that led to this state
- `children`: Dictionary of child nodes
- `visit_count`: Number of times visited
- `value_sum`: Cumulative value
- `prior`: Prior probability from neural network

**Key Methods:**
- `ucb_score()`: Calculates Upper Confidence Bound for tree policy
- `select_child()`: Selects best child using UCB formula
- `expand()`: Creates child nodes for all legal actions
- `backup()`: Propagates value up the tree

### 2. AlphaZeroMCTS
Manages the Monte Carlo Tree Search process.

**Parameters:**
- `game`: Game environment instance
- `model`: Neural network model
- `c_puct`: Exploration constant (default: 1.0)
- `num_simulations`: Number of MCTS simulations (default: 100)

**Key Methods:**
- `search()`: Performs MCTS simulations from root state with Dirichlet noise (alpha=0.3, epsilon=0.25)
- `get_action_probabilities()`: Returns visit-count-based action distribution with temperature control

### 3. AlphaZeroNet
ResNet-based neural network with dual heads.

**Architecture:**
```
Input (3x3x2) → Projection Layer → Residual Blocks × 4 → 
    ├─ Value Head → tanh(-1 to 1)
    └─ Policy Head → softmax(9 actions)
```

**Components:**
- **Input Layer**: Linear projection from 18 features to hidden dimension
- **Residual Blocks**: 4 blocks with batch normalization and dropout
- **Value Head**: 3-layer MLP outputting game outcome prediction
- **Policy Head**: 3-layer MLP outputting action probabilities

### 4. AlphaZeroModel
High-level model wrapper for training and inference.

**Features:**
- State encoding (board → tensor) with dual-plane representation
- Training data management with replay buffer (capacity: 50,000)
- Loss computation with legal move masking:
  - Value loss: Mean Squared Error (MSE)
  - Policy loss: Cross-entropy with illegal moves masked to -1e9 before softmax
- Model persistence (save/load) with optimizer state
- Learning rate scheduling with StepLR (step_size=100, gamma=0.9)
- AdamW optimizer with weight decay (1e-4) and gradient clipping (max_norm=1.0)

### 5. AlphaZeroTrainer
Orchestrates the training process.

**Capabilities:**
- Self-play game generation (sequential or parallel)
- Training iteration management
- Model evaluation against checkpoints
- Results tracking and CSV export
- Temperature annealing for exploration control
- Data augmentation via 8 dihedral symmetries (rotations and reflections)
- Replay buffer integration

---

## Training Pipeline

### Complete Training Flow

```mermaid
flowchart TD
    Start([Start Training]) --> Init[Initialize Model & Trainer]
    Init --> SaveBaseline[Save Baseline Model<br/>iteration_0.pth]
    SaveBaseline --> IterLoop{Iteration < N_ITERATIONS?}
    
    IterLoop -->|Yes| SelfPlay[Self-Play Phase]
    IterLoop -->|No| SaveFinal[Save Final Model]
    SaveFinal --> End([Training Complete])
    
    SelfPlay --> ParallelCheck{Parallel Mode?}
    
    ParallelCheck -->|Yes| MultiProc[Multiprocessing Pool]
    MultiProc --> Workers[N Workers Play Games]
    Workers --> Collect[Collect All Examples]
    
    ParallelCheck -->|No| Sequential[Sequential Games]
    Sequential --> Collect
    
    Collect --> Augment[Data Augmentation<br/>8 Symmetries]
    Augment --> AddData[Add to Replay Buffer<br/>capacity: 50,000]
    AddData --> TrainNN[Train Neural Network<br/>Sample from Buffer]
    
    TrainNN --> CheckpointCheck{iteration % SAVE_INTERVAL == 0?}
    
    CheckpointCheck -->|Yes| SaveCheckpoint[Save Checkpoint]
    SaveCheckpoint --> EvalLoop[Evaluate Against<br/>Previous Checkpoints]
    EvalLoop --> SaveCSV[Save Results to CSV]
    SaveCSV --> IterLoop
    
    CheckpointCheck -->|No| IterLoop
```

### Self-Play Game Flow

```mermaid
flowchart TD
    StartGame([Start Self-Play Game]) --> InitGame[Initialize Game State]
    InitGame --> GameLoop{Game Over?}
    
    GameLoop -->|No| SavePlayer[Save Current Player]
    SavePlayer --> CalcTemp[Calculate Temperature<br/>Based on Iteration & Move]
    CalcTemp --> RunMCTS[Run MCTS Search<br/>N_SIMULATIONS times]
    
    RunMCTS --> GetProbs[Get Action Probabilities<br/>from Visit Counts + Temp]
    GetProbs --> StoreExample[Store Training Example<br/>state, probs, None, player]
    
    StoreExample --> SampleAction[Sample Action<br/>from Distribution]
    SampleAction --> MakeMove[Make Move]
    MakeMove --> GameLoop
    
    GameLoop -->|Yes| GetResult[Get Game Result<br/>+1, -1, or 0]
    GetResult --> FillValues[Fill Values in Examples<br/>result × player]
    FillValues --> ReturnExamples([Return Training Examples])
```

### MCTS Search Process

```mermaid
flowchart TD
    StartMCTS([Start MCTS Search]) --> CreateRoot[Create Root Node]
    CreateRoot --> ExpandRoot[Expand Root with<br/>Dirichlet Noise]
    ExpandRoot --> GetRootNN[Neural Network Prediction]
    GetRootNN --> AddNoise[Add Dirichlet Noise<br/>alpha=0.3, eps=0.25]
    AddNoise --> SimLoop{Simulation < N_SIMULATIONS?}
    
    SimLoop -->|Yes| Selection[Selection Phase]
    SimLoop -->|No| ReturnRoot([Return Root Node])
    
    Selection --> CopyGame[Copy Game State]
    CopyGame --> TraverseLoop{Node Expanded &<br/>Game Not Over?}
    
    TraverseLoop -->|Yes| SelectChild[Select Child with<br/>Best UCB Score]
    SelectChild --> ApplyMove[Apply Move to Game]
    ApplyMove --> AddToPath[Add to Path]
    AddToPath --> TraverseLoop
    
    TraverseLoop -->|No| TerminalCheck{Game Over?}
    
    TerminalCheck -->|Yes| GetScore[Get Terminal Score<br/>× Current Player]
    GetScore --> Backup[Backup Value<br/>Negate at Each Level]
    Backup --> SimLoop
    
    TerminalCheck -->|No| Expansion[Expansion Phase]
    Expansion --> NNPredict[Neural Network Prediction]
    
    NNPredict --> GetValue[Get Value Estimate]
    NNPredict --> GetPolicy[Get Policy Logits]
    
    GetPolicy --> FilterLegal[Filter Legal Actions]
    FilterLegal --> Normalize[Normalize Probabilities]
    Normalize --> CreateChildren[Create Child Nodes<br/>with Priors]
    CreateChildren --> GetValue
    GetValue --> Backup
```

### Neural Network Training

```mermaid
flowchart TD
    StartTrain([Start NN Training]) --> CheckData{Enough Data?<br/>≥ BATCH_SIZE}
    
    CheckData -->|No| Skip([Skip Training])
    CheckData -->|Yes| SampleData[Sample from Replay Buffer<br/>Max 4096 samples]
    
    SampleData --> EncodeStates[Encode States<br/>to Tensors]
    EncodeStates --> ConvertPolicy[Convert Policies<br/>to Vectors]
    ConvertPolicy --> ToTensor[Convert to<br/>PyTorch Tensors]
    
    ToTensor --> EpochLoop{Epoch < EPOCHS?}
    
    EpochLoop -->|Yes| Shuffle[Shuffle Data]
    Shuffle --> BatchLoop{More Batches?}
    
    BatchLoop -->|Yes| GetBatch[Get Next Batch]
    GetBatch --> Forward[Forward Pass]
    
    Forward --> MaskIllegal[Mask Illegal Moves<br/>in Policy Logits]
    MaskIllegal --> ValueLoss[Compute Value Loss<br/>MSE]
    MaskIllegal --> PolicyLoss[Compute Policy Loss<br/>Cross-Entropy]
    
    ValueLoss --> TotalLoss[Total Loss =<br/>Value + Policy]
    PolicyLoss --> TotalLoss
    
    TotalLoss --> Backward[Backward Pass]
    Backward --> ClipGrad[Clip Gradients<br/>max_norm=1.0]
    ClipGrad --> UpdateWeights[Update Weights<br/>AdamW]
    UpdateWeights --> BatchLoop
    
    BatchLoop -->|No| LogEpoch[Log Epoch Results]
    LogEpoch --> UpdateLR[Update Learning Rate<br/>StepLR Scheduler]
    UpdateLR --> EpochLoop
    
    EpochLoop -->|No| Done([Training Complete<br/>Keep Buffer])
```

### UCB Score Calculation

The Upper Confidence Bound (UCB) formula balances exploitation and exploration:

```
UCB(node) = Q(node) + c_puct × P(node) × √(N(parent)) / (1 + N(node))
```

Where:
- **Q(node)**: Average value of the node (exploitation)
- **P(node)**: Prior probability from neural network
- **N(parent)**: Visit count of parent node
- **N(node)**: Visit count of current node
- **c_puct**: Exploration constant (typically 1.0)

```mermaid
flowchart LR
    subgraph "UCB Components"
        A[Visit Count = 0] -->|Yes| B[Return Infinity]
        A -->|No| C[Calculate Q<br/>value_sum / visit_count]
        C --> D[Calculate Exploration<br/>c_puct × prior × √parent_visits / 1+visits]
        D --> E[UCB = Q + Exploration]
    end
```

---

## Hyperparameters

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_ITERATIONS` | 500 | Total training iterations |
| `N_GAMES` | 200 | Self-play games per iteration |
| `N_SIMULATIONS` | 100 | MCTS simulations per move |
| `C_PUCT` | 1.0 | UCB exploration constant |
| `TEMPERATURE_INITIAL` | 1.0 | Exploration temperature for opening moves |
| `TEMPERATURE_FINAL` | 0.0 | Deterministic selection after threshold |
| `TEMPERATURE_MOVE_THRESHOLD` | 3 | Number of opening moves with high temperature |
| `TEMPERATURE_ANNEAL_ITER` | 100 | Iteration after which temperature is always final |
| `SYMMETRY_AUGMENT` | True | Enable data augmentation via symmetries |
| `REPLAY_CAPACITY` | 50000 | Maximum replay buffer size |

### Neural Network

| Parameter | Value | Description |
|-----------|-------|-------------|
| `HIDDEN_SIZE` | 256 | Hidden layer dimension |
| `NUM_RES_BLOCKS` | 4 | Number of residual blocks |
| `DROPOUT` | 0.2 | Dropout probability |
| `LEARNING_RATE` | 0.001 | Initial learning rate |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 10 | Epochs per iteration |

### Optimization

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LR_STEP_SIZE` | 100 | LR scheduler step size |
| `LR_GAMMA` | 0.9 | LR decay factor |
| `SAVE_INTERVAL` | 10 | Checkpoint save interval |
| `EVAL_INTERVAL` | 10 | Evaluation interval |
| `EVAL_GAMES` | 50 | Games for evaluation |
| `EVAL_SIMULATIONS` | 100 | MCTS simulations during evaluation |

---

## API Reference

### TicTacToeGame Interface

```python
class TicTacToeGame:
    def __init__(self):
        """Initialize empty 3x3 board"""
        
    def copy(self) -> TicTacToeGame:
        """Create deep copy of game state"""
        
    def state(self) -> tuple:
        """Return immutable state representation"""
        
    def valid_moves(self) -> list[int]:
        """Return list of legal move indices (0-8)"""
        
    def make_move(self, action: int) -> bool:
        """Apply move and switch players"""
        
    def over(self) -> bool:
        """Check if game is terminal"""
        
    def score(self) -> int:
        """Return +1 (player 1 win), -1 (player -1 win), 0 (draw)"""
```

### Model Usage

```python
# Initialize model
model = AlphaZeroModel(
    board_size=3,
    action_size=9,
    learning_rate=0.001
)

# Get predictions
game = TicTacToeGame()
value, policy = model.predict(game)  # Returns value in [-1,1], policy as numpy array

# Add training data (stores in replay buffer)
model.add_training_data(
    state=(game.state(), game.current_player),  # State with player perspective
    action_probs={0: 0.5, 1: 0.3, 2: 0.2},     # Action probabilities
    value=1.0                                    # Game outcome from player's view
)

# Train on collected data (samples from replay buffer)
model.train(batch_size=64, epochs=10)

# Save/load with optimizer state
model.save_model("checkpoint.pth")
loaded_model = AlphaZeroModel.load_from_file("checkpoint.pth")
```

### Training Example

```python
# Create trainer
trainer = AlphaZeroTrainer(
    game_class=TicTacToeGame,
    model=model,
    mcts_simulations=100
)

# Run single iteration with parallel self-play
trainer.train_iteration(
    iteration=1,
    num_games=200,
    num_workers=4  # None = auto-detect, 1 = sequential
)

# Evaluate against baseline
wins, draws, losses = trainer.evaluate_model(
    opponent_model=baseline_model,
    num_games=50
)
```

---

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Input"
        A[Game State<br/>3×3 Board]
    end
    
    subgraph "Encoding"
        B[Current Player Plane<br/>3×3]
        C[Opponent Plane<br/>3×3]
        D[Stacked Input<br/>3×3×2 = 18]
    end
    
    subgraph "Neural Network"
        E[Input Layer<br/>18→256]
        F[ResBlock ×4<br/>256→256]
        G1[Value Head<br/>256→128→64→1]
        G2[Policy Head<br/>256→128→64→9]
    end
    
    subgraph "Output"
        H1[Value<br/>-1 to +1]
        H2[Policy<br/>9 probabilities]
    end
    
    A --> B
    A --> C
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G1
    F --> G2
    G1 --> H1
    G2 --> H2
```

---

## State Encoding

The game state is encoded into two binary planes:

1. **Current Player Plane**: `1` where current player has pieces, `0` elsewhere
2. **Opponent Plane**: `1` where opponent has pieces, `0` elsewhere

Example:
```
Board:        Current (X):  Opponent (O):
X | O | -     1 | 0 | 0     0 | 1 | 0
---------     ---------     ---------
- | X | -  →  0 | 1 | 0  +  0 | 0 | 0
---------     ---------     ---------
O | - | -     0 | 0 | 0     1 | 0 | 0

Flattened: [1,0,0,0,1,0,0,0,0, 0,1,0,0,0,0,1,0,0] (18 features)
```

---

## File Structure

```
alpha-toe/
├── main.py                      # Core implementation
├── play.py                      # Interactive play script
├── diagnose.py                  # Debugging utilities
├── export_onnx.py              # ONNX model export
├── tournament.py               # Tournament evaluation
├── test_sanity.py              # Sanity tests
├── TECHNICAL_DOCUMENTATION.md   # This file
├── README.md                    # Project overview
├── report.md                    # Training report
├── pyproject.toml              # Dependencies
├── uv.lock                     # Lock file
└── models/
    ├── alphazero_iter_0.pth    # Baseline checkpoint
    ├── alphazero_iter_10.pth   # Checkpoint at iteration 10
    ├── alphazero_iter_20.pth   # etc.
    ├── evaluation_results.csv  # Evaluation metrics
    ├── tournament_history.csv  # Tournament results
    ├── tournament_ratings.csv  # ELO ratings
    └── run1/, run2/, run3/     # Training run archives
```

---

## Advanced Features

### Temperature Annealing

The system implements a sophisticated temperature annealing scheme to balance exploration and exploitation during training:

**Configuration:**
- `TEMPERATURE_INITIAL = 1.0`: High exploration for opening moves
- `TEMPERATURE_FINAL = 0.0`: Deterministic (greedy) selection after threshold
- `TEMPERATURE_MOVE_THRESHOLD = 3`: First 3 moves use high temperature
- `TEMPERATURE_ANNEAL_ITER = 100`: After iteration 100, always use final temperature

**Logic:**
```python
if iteration >= TEMPERATURE_ANNEAL_ITER:
    temperature = TEMPERATURE_FINAL  # Always deterministic
elif move_index < TEMPERATURE_MOVE_THRESHOLD:
    temperature = TEMPERATURE_INITIAL  # Explore opening
else:
    temperature = TEMPERATURE_FINAL  # Exploit mid/endgame
```

**Effect on Action Selection:**
- Temperature = 1.0: Samples from visit count distribution (exploration)
- Temperature = 0.0: Selects action with maximum visit count (exploitation)
- Temperature between 0 and 1: Interpolates between the two

### Data Augmentation via Symmetries

The trainer augments each training example using the 8 dihedral symmetries of the 3×3 board:

**Symmetry Operations:**
1. **Identity**: Original position
2. **Rotation 90°**: Clockwise rotation
3. **Rotation 180°**: Half turn
4. **Rotation 270°**: Counter-clockwise rotation
5. **Horizontal Reflection**: Flip across horizontal axis
6. **Vertical Reflection**: Flip across vertical axis
7. **Main Diagonal Reflection**: Flip across diagonal (0,4,8)
8. **Anti-Diagonal Reflection**: Flip across diagonal (2,4,6)

**Implementation:**
- Each symmetry generates a permutation of board indices
- Both board state and policy are transformed consistently
- Augmentation increases training data 8-fold
- Can be disabled by setting `SYMMETRY_AUGMENT = False`

**Example:**
```
Original:        Rot90:          HFlip:
X | O | -        - | X | X       O | - | -
---------        ---------       ---------
X | - | -   →    - | - | O   or  X | - | -
---------        ---------       ---------
- | - | -        - | - | -       X | O | -
```

### Replay Buffer

The model maintains a replay buffer to stabilize training:

**Features:**
- **Capacity**: 50,000 training examples
- **Persistence**: Buffer persists across training iterations
- **Sampling**: Random subset (up to 4,096) sampled each iteration
- **FIFO Policy**: Oldest examples removed when capacity exceeded

**Benefits:**
- Reduces correlation between consecutive training batches
- Enables learning from diverse game positions
- Improves sample efficiency

### Dirichlet Noise for Exploration

During MCTS, Dirichlet noise is added to the root node's prior probabilities:

**Parameters:**
- `alpha = 0.3`: Concentration parameter (lower = more concentrated)
- `epsilon = 0.25`: Mixing weight (25% noise, 75% policy)

**Formula:**
```
P_noisy = (1 - ε) × P_network + ε × Dirichlet(α)
```

**Purpose:**
- Ensures diverse exploration during self-play
- Prevents premature convergence to suboptimal strategies
- Only applied at root node (not during tree traversal)

---

## Performance Considerations

### Parallelization
- Self-play games run in parallel using `multiprocessing.Pool`
- Auto-detects CPU cores (uses `cpu_count - 1`)
- Worker processes receive serialized model state
- Can be disabled by setting `num_workers=1`

### Memory Management
- Replay buffer capped at 50,000 examples
- Training samples up to 4,096 examples per iteration
- Gradient clipping (max_norm=1.0) prevents gradient explosion
- Terminal output truncated to prevent memory overflow

### Computational Complexity
- **MCTS per move**: O(N_SIMULATIONS × tree_depth)
- **Neural network forward pass**: O(hidden_size²)
- **Training**: O(EPOCHS × dataset_size / BATCH_SIZE)

---

## Common Workflows

### Training from Scratch
```bash
python main.py
```

### Resume Training
```python
model = AlphaZeroModel.load_from_file("models/alphazero_iter_50.pth")
trainer = AlphaZeroTrainer(TicTacToeGame, model)
# Continue training...
```

### Play Against Model
```python
from play import play_against_ai
play_against_ai("models/alphazero_iter_50.pth")
```

### Analyze Evaluation Results
```python
import pandas as pd
df = pd.read_csv("models/evaluation_results.csv")
print(df.groupby('current_iter')['win_rate'].mean())
```

---

## Future Improvements

1. **Advanced Features**
   - ✅ Dirichlet noise for root exploration (implemented)
   - ✅ Temperature annealing schedule (implemented)
   - ✅ Prioritized experience replay via capacity management (implemented)
   - ✅ Data augmentation via symmetries (implemented)
   - Model ensemble voting
   - Virtual loss for parallel MCTS

2. **Optimizations**
   - GPU acceleration for batch predictions
   - Cached neural network evaluations (transposition table)
   - Asynchronous self-play with separate actors
   - Distributed training across multiple machines

3. **Extensions**
   - Larger board games (Connect Four, Gomoku)
   - Imperfect information games
   - Transfer learning experiments
   - Opening book generation

---

## References

- Silver, D., et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." arXiv:1712.01815
- Silver, D., et al. (2016). "Mastering the game of Go with deep neural networks and tree search." Nature 529, 484-489
- Browne, C., et al. (2012). "A Survey of Monte Carlo Tree Search Methods." IEEE Transactions on Computational Intelligence and AI in Games

---

**Last Updated**: November 1, 2025  
**Version**: 2.0  
**Author**: AlphaZero Tic-Tac-Toe Project
