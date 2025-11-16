# AlphaToe - 4×4×4 Qubic MCTS Explainer

An interactive web application that demonstrates AlphaZero-style reinforcement learning for 4×4×4 Qubic (3D Tic-Tac-Toe). Built with vanilla JavaScript and ONNX Runtime, this application allows users to play against a trained neural network and visualize the Monte Carlo Tree Search (MCTS) decision-making process in real-time.

## Features

- 🎮 **Interactive Gameplay**: Play 4×4×4 Qubic against an AI trained using AlphaZero techniques
- 🧠 **MCTS Visualization**: See real-time analysis of the AI's thought process
- 📊 **Move Analysis**: View detailed statistics including visit counts, Q-values, and policy probabilities
- 🎯 **Heat Maps**: Visual representation of move probabilities overlaid on the 4 game boards
- ⚡ **WebGPU/WASM Support**: Fast neural network inference using ONNX Runtime
- 📱 **Responsive Design**: Clean, modern UI that works across devices
- 🎨 **Dynamic Background**: Animated X/O tile background for visual appeal
- 🎲 **3D Animations**: Interactive 3D board animations and component effects

## Project Structure

```
teach/
├── index.html                    # Main HTML file with minimal markup
├── package.json                  # NPM configuration
├── AGENTS.md                     # Agent guidelines for coding assistants
├── src/                          # JavaScript source modules
│   ├── show_3d_ttt_game.js      # Main application logic and initialization
│   ├── game.js                  # 4×4×4 Qubic game rules and logic
│   ├── mcts.js                  # Monte Carlo Tree Search implementation
│   ├── model.js                 # ONNX neural network wrapper
│   ├── ui.js                    # UI rendering and visualization
│   ├── utils.js                 # Utility functions
│   ├── hero-animation.js        # Background hero animation with X/O tiles
│   ├── qubic-animation.js       # 3D qubic board animation
│   ├── components-animation.js  # Component animations
│   └── styles.css               # Application styles
├── public/                       # Static assets served by Vite
│   └── assets/
│       ├── model_4x4x4.onnx     # Trained neural network model
│       ├── x.svg                # X symbol for animations
│       └── o.svg                # O symbol for animations
├── docs/                         # Documentation
├── notebooks/                    # Jupyter notebooks
└── assets/                       # Legacy assets (may be removed)
```

## Architecture

### Module Overview

#### `utils.js`
Pure utility functions for mathematical operations and coordinate transformations:
- `softmax()` - Softmax normalization for policy distributions
- `normalize()` - Array normalization
- `toIdx()` - Convert 3D coordinates (x, y, z) to flat index
- `toXYZ()` - Convert flat index to 3D coordinates

#### `game.js`
4×4×4 Qubic game logic:
- **Game Class**: Manages game state, board representation, and rules
- **Win Detection**: Checks 76 possible winning lines (axis lines, plane diagonals, space diagonals)
- **Move Validation**: Ensures legal moves and state transitions

#### `model.js`
ONNX neural network integration:
- **ONNXAlphaZero Class**: Wrapper for ONNX Runtime inference
- **Board Encoding**: Converts game state to neural network input format
- **Dual Output**: Returns both value estimate and policy logits
- **Backend Support**: Automatic fallback from WebGPU to WASM

#### `mcts.js`
Monte Carlo Tree Search algorithm:
- **Node Class**: Tree node representation with visit counts and values
- **MCTS Class**: Complete MCTS implementation with UCB selection
- **Dirichlet Noise**: Exploration noise at root node
- **Parallel Inference**: Efficient batched neural network queries

#### `ui.js`
User interface and rendering:
- **Board Rendering**: Canvas-based 2D projection of 4×4×4 game state
- **Heat Maps**: Visual overlay of move probabilities
- **Animations**: AI thinking indicators and game over overlays
- **Statistics Display**: Real-time MCTS analysis table

#### `show_3d_ttt_game.js`
Application orchestration:
- **Event Handlers**: Mouse clicks, button interactions
- **Game Flow**: Turn management, AI vs human coordination
- **State Management**: Game state and UI synchronization
- **Initialization**: Model loading and setup

#### `hero-animation.js`
Background animation system:
- **Tile Generation**: Creates grid of X/O tiles filling viewport
- **Animation Loop**: Randomly adds/removes symbols for dynamic background
- **SVG Integration**: Uses vector graphics for crisp rendering

#### `qubic-animation.js`
3D board visualization:
- **3D Rendering**: Animates the 4×4×4 game board in 3D space
- **Interactive Elements**: Visual effects for game state changes

## Technical Details

### Game Rules: 4×4×4 Qubic

Qubic is a 3D extension of Tic-Tac-Toe played on a 4×4×4 cube:
- **Objective**: Get 4 pieces in a row
- **Winning Lines**: 76 possible lines total
  - 48 axis-aligned lines (3 axes × 16 lines each)
  - 24 plane diagonals (12 per orientation)
  - 4 space diagonals (corner to corner through center)

### AlphaZero Algorithm

The AI uses a simplified AlphaZero approach:

1. **Neural Network**: Dual-head architecture
   - Policy head: Outputs move probabilities
   - Value head: Outputs position evaluation (-1 to +1)

2. **MCTS Process**:
   - **Selection**: Traverse tree using UCB formula: `Q + c_puct × P × √N_parent / (1 + N)`
   - **Expansion**: Add new node and evaluate with neural network
   - **Backup**: Propagate value up the tree, alternating signs

3. **Dirichlet Noise**: Added to root node for exploration
   - Formula: `π = (1 - ε) × P + ε × Dir(α)`
   - Default: ε=0.25, α=0.3

### ONNX Runtime

The model runs entirely in the browser using ONNX Runtime Web:
- **WebGPU**: Hardware acceleration when available
- **WASM**: CPU fallback for compatibility
- **Input Shape**: `[1, 2, 4, 4, 4]` (batch, channels, z, y, x)
- **Outputs**:
  - `policy_logits`: [64] unnormalized action preferences
  - `value`: [1] position evaluation

### Animations

The application includes several animation systems:
- **Hero Background**: Grid of X/O tiles that randomly appear/disappear
- **Qubic Board**: 3D rotation and movement effects for the game board
- **UI Components**: Smooth transitions for buttons, overlays, and statistics
- **AI Thinking**: Pulsing indicators during MCTS computation

## Setup & Installation

### Prerequisites
- Modern web browser (Chrome/Edge recommended for WebGPU support)
- Node.js and npm (optional, for development server)

### Quick Start

1. **Clone the repository** (if not already done):
   ```bash
   cd teach/
   ```

2. **Install dependencies** (optional):
   ```bash
   npm install
   ```

3. **Serve the application**:
   
   **Option A - Using Python:**
   ```bash
   python3 -m http.server 8000
   ```
   
   **Option B - Using Node.js (http-server):**
   ```bash
   npx http-server -p 8000
   ```
   
    **Option C - Using Vite (already in package.json):**
    ```bash
    npm run dev
    ```
    (Runs on http://localhost:5173 by default)

4. **Open in browser**:
   ```
   http://localhost:8000
   ```

### Important Notes

- ⚠️ **File Protocol**: Due to CORS restrictions with ES modules and ONNX model loading, you **must** serve the files over HTTP. Simply opening `index.html` directly won't work.
- 💡 **WebGPU**: For best performance, enable WebGPU in Chrome: `chrome://flags/#enable-unsafe-webgpu`
- 🔄 **Fallback**: The app automatically falls back to WASM if WebGPU is unavailable
- 🎨 **Assets**: Static assets are served from `public/assets/` via Vite dev server

## Usage

### Playing the Game

1. **Start Screen**: Choose who goes first (You or AI)
2. **Make Moves**: Click any empty cell on any of the 4 boards (z-layers 0-3)
3. **AI Response**: The AI will think and make its move automatically
4. **Analysis**: Click "More Info" to see detailed MCTS statistics
5. **Hint**: Click "Ask AI to help you" to see what the AI would do in your position

### Understanding the Analysis

When "More Info" is enabled, you'll see:

- **Top 16 Actions**: Best moves ranked by visit count
  - **Cell (x,y,z)**: Coordinate of the move
  - **N**: Visit count (higher = more explored)
  - **Q(parent)**: Quality estimate from parent's perspective
  - **P**: Prior probability from neural network
  - **U**: Exploration bonus (UCB term)
  - **π**: Final policy (visit-based probability)

- **Heat Maps**: Blue overlay on boards showing move probabilities
  - Darker blue = higher probability
  - Updates after running MCTS

- **Predicted Value**: Neural network's evaluation of current position
  - +1: AI winning
  - 0: Draw
  - -1: Human winning

### MCTS Parameters

The application uses these default parameters:

- **Simulations**: 1000 MCTS iterations per move
- **c_puct**: 1.0 (exploration constant)
- **Temperature τ**: 0.0 (deterministic play, picks best move)
- **Dirichlet ε**: 0.0 (no exploration noise during play)
- **Dirichlet α**: 0.3 (unused when ε=0)

## Browser Compatibility

| Browser | WebGPU | WASM | Status |
|---------|--------|------|--------|
| Chrome 113+ | ✅ | ✅ | Full support |
| Edge 113+ | ✅ | ✅ | Full support |
| Firefox | ❌ | ✅ | WASM only |
| Safari | ❌ | ✅ | WASM only |

## Performance

- **MCTS Speed**: ~100-200 simulations/second (WebGPU) or ~50-100 (WASM)
- **Model Size**: ~500KB ONNX model
- **Inference Time**: 5-20ms per neural network call
- **Total Thinking Time**: 5-15 seconds for 1000 simulations

## Development

### Local Development Server

The project uses Vite for development. Run `npm run dev` to start the dev server on http://localhost:5173. This server is typically kept running during development for hot reloading and fast iteration.

### File Organization

The codebase uses ES6 modules for clean separation of concerns:
- Each module has a single responsibility
- Dependencies are explicitly imported/exported
- No global namespace pollution
- Easy to test and maintain

### Adding Features

To extend the application:

1. **New UI Elements**: Edit `index.html` and `src/styles.css`
2. **Game Logic**: Modify `src/game.js`
3. **AI Behavior**: Adjust parameters in `src/app.js` or `src/mcts.js`
4. **Visualization**: Update rendering in `src/ui.js`

### Debugging

Open browser DevTools console to see:
- Model loading status (WebGPU vs WASM)
- Inference timing information
- Any runtime errors

## Credits

- **Algorithm**: Based on DeepMind's AlphaZero paper
- **Game**: Qubic (4×4×4 Tic-Tac-Toe)
- **Framework**: ONNX Runtime Web
- **UI**: Vanilla JavaScript, Canvas API

## License

Part of the alpha-toe repository. See parent directory for license information.

## Troubleshooting

### Model Won't Load
- Ensure you're serving over HTTP (not file://)
- Check browser console for CORS errors
- Verify `assets/model_4x4x4.onnx` exists

### Slow Performance
- Try enabling WebGPU in browser flags
- Reduce simulation count (edit `src/app.js`)
- Close other browser tabs

### UI Issues
- Clear browser cache
- Verify `src/styles.css` is loading
- Check console for JavaScript errors

## Future Improvements

Potential enhancements:
- [ ] Adjustable MCTS parameters via UI
- [ ] Move history and undo functionality
- [ ] Save/load game states
- [ ] Multiple AI difficulty levels
- [ ] 3D visualization option
- [ ] Mobile touch optimization
- [ ] Game analysis replay
