/* ===== Main Application Logic ===== */

import { toIdx } from './utils.js';
import { Game } from './game.js';
import { ONNXAlphaZero } from './model.js';
import { MCTS } from './mcts.js';
import {
  drawAll,
  updateStats,
  showGameOver,
  startThinkingAnimation,
  stopThinkingAnimation,
  showStartChoice,
  showCellTooltip,
  hideCellTooltip
} from './ui.js';

// Game state
let game = new Game();
let model = new ONNXAlphaZero();
let lastRoot = null;
let startPlayer = null;
let isUserRequestedHelp = false;

// Canvas setup
const canvases = [0, 1, 2, 3].map(i => document.getElementById(`board${i}`));
const ctxs = canvases.map(c => c.getContext("2d"));

/**
 * Run MCTS once
 */
async function runMCTSOnce(userRequested = false) {
  const sims = 1000;
  const cpuct = 1.0;
  const eps = 0.0;
  const alpha = 0.3;
  const tau = 0.0;

  const mcts = new MCTS(game.clone(), model, cpuct, alpha, eps);
  lastRoot = await mcts.search(sims);
  isUserRequestedHelp = userRequested;
  updateStats(lastRoot, tau, game, ctxs, drawAll);
}

/**
 * AI makes a move
 */
async function aiMove() {
  startThinkingAnimation();
  await runMCTSOnce();
  stopThinkingAnimation();
  
  const pi = new Array(64).fill(0);
  let maxN = -1;
  for (const [a, child] of lastRoot.children.entries()) {
    if (child.N > maxN) {
      maxN = child.N;
      pi.fill(0);
      pi[a] = 1;
    }
  }
  
  const legal = game.validMoves();
  if (legal.length === 0) return;
  
  let best = legal[0], bestP = -1;
  for (const a of legal) {
    if (pi[a] > bestP) {
      bestP = pi[a];
      best = a;
    }
  }
  
  game.makeMove(best);
  lastRoot = null;
  drawAll(ctxs, game.board, null);
  
  if (game.over()) {
    showGameOver(game, ctxs, drawAll);
  }
}

/**
 * Setup event handlers
 */
function setupEventHandlers() {
  // Board clicks (human move)
  canvases.forEach((cv, z) => {
    cv.addEventListener("click", async (e) => {
      const aiStatusText = document.getElementById('aiStatusText');
      if (aiStatusText && aiStatusText.innerHTML.includes('Thinking')) return;
      if (game.player !== 1) return;
      
      const rect = cv.getBoundingClientRect();
      const x = Math.floor((e.clientX - rect.left) / (rect.width / 4));
      const y = Math.floor((e.clientY - rect.top) / (rect.height / 4));
      if (x < 0 || x > 3 || y < 0 || y > 3) return;
      
      const idx = toIdx(x, y, z);
      if (game.board[idx] !== 0 || game.over()) return;
      
      game.makeMove(idx);
      drawAll(ctxs, game.board, null);
      lastRoot = null;
      isUserRequestedHelp = false;
      
      if (game.over()) {
        showGameOver(game, ctxs, drawAll);
      } else if (game.player === -1) {
        await aiMove();
      }
    });

    // Hover for tooltips
    cv.addEventListener("mousemove", (e) => {
      const rect = cv.getBoundingClientRect();
      const x = Math.floor((e.clientX - rect.left) / (rect.width / 4));
      const y = Math.floor((e.clientY - rect.top) / (rect.height / 4));
      if (x < 0 || x > 3 || y < 0 || y > 3) {
        hideCellTooltip();
        return;
      }
      showCellTooltip(x, y, z, e, game, isUserRequestedHelp);
    });

    cv.addEventListener("mouseout", () => {
      hideCellTooltip();
    });
  });

  // Reset button
  const resetBtn = document.getElementById("reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", async () => {
      game = new Game();
      lastRoot = null;
      drawAll(ctxs, game.board, null);
      startPlayer = null;
      
      const gameOverOverlay = document.getElementById("gameOverOverlay");
      if (gameOverOverlay) gameOverOverlay.style.display = 'none';
      
      await showStartChoice();
      game.player = startPlayer;
      if (startPlayer === -1) {
        await aiMove();
      }
    });
  }

  // Restart game button
  const restartBtn = document.getElementById("restartGame");
  if (restartBtn) {
    restartBtn.addEventListener("click", async () => {
      game = new Game();
      lastRoot = null;
      drawAll(ctxs, game.board, null);
      startPlayer = null;
      
      const gameOverOverlay = document.getElementById("gameOverOverlay");
      if (gameOverOverlay) gameOverOverlay.style.display = 'none';
      
      await showStartChoice();
      game.player = startPlayer;
      if (startPlayer === -1) {
        await aiMove();
      }
    });
  }

  // Run MCTS button (Ask AI to help)
  const runBtn = document.getElementById("run");
  if (runBtn) {
    runBtn.addEventListener("click", async () => {
      const aiStatusText = document.getElementById('aiStatusText');
      if (aiStatusText && aiStatusText.innerHTML.includes('Thinking')) return;
      startThinkingAnimation();
      await runMCTSOnce(true);
      stopThinkingAnimation();
    });
  }

  // Start choice buttons
  const humanStartBtn = document.getElementById("humanStart");
  if (humanStartBtn) {
    humanStartBtn.addEventListener("click", () => {
      if (startPlayer === null) {
        startPlayer = 1;
        const overlay = document.getElementById("startOverlay");
        if (overlay) overlay.style.display = 'none';
        if (window.resolveStart) window.resolveStart();
      }
    });
  }

  const aiStartBtn = document.getElementById("aiStart");
  if (aiStartBtn) {
    aiStartBtn.addEventListener("click", () => {
      if (startPlayer === null) {
        startPlayer = -1;
        const overlay = document.getElementById("startOverlay");
        if (overlay) overlay.style.display = 'none';
        if (window.resolveStart) window.resolveStart();
      }
    });
  }
}

/**
 * Start hero background animation
 */
function startHeroAnimation() {
  const tiles = document.querySelectorAll('.tile');
  const targetFillPercentage = 0.35; // 35% of tiles should be filled
  const targetCount = Math.floor(tiles.length * targetFillPercentage);

  const animate = () => {
    const filledTiles = Array.from(tiles).filter(tile => tile.classList.contains('x') || tile.classList.contains('o'));
    const emptyTiles = Array.from(tiles).filter(tile => !tile.classList.contains('x') && !tile.classList.contains('o'));
    
    const currentCount = filledTiles.length;

    if (currentCount < targetCount && emptyTiles.length > 0) {
      // Add a new X or O
      const randomTile = emptyTiles[Math.floor(Math.random() * emptyTiles.length)];
      const isX = Math.random() < 0.5;
      randomTile.classList.add(isX ? 'x' : 'o');
      randomTile.setAttribute('data-symbol', isX ? 'X' : 'O');
    } else if (currentCount > 0) {
      // Randomly remove an X or O
      const randomFilledTile = filledTiles[Math.floor(Math.random() * filledTiles.length)];
      randomFilledTile.className = 'tile';
      randomFilledTile.removeAttribute('data-symbol');
    }

    setTimeout(animate, 400); // Animate every 400ms
  };

  animate();
}

/**
 * Initialize app
 */
export async function init() {
  // Draw initial board
  drawAll(ctxs, game.board, null);
  
  // Update parameter displays
  const simsVal = document.getElementById("simsVal");
  const cpuctVal = document.getElementById("cpuctVal");
  const tempVal = document.getElementById("tempVal");
  const epsVal = document.getElementById("epsVal");
  const alphaVal = document.getElementById("alphaVal");
  
  if (simsVal) simsVal.textContent = "1000";
  if (cpuctVal) cpuctVal.textContent = "1.00";
  if (tempVal) tempVal.textContent = "0.00";
  if (epsVal) epsVal.textContent = "0.00";
  if (alphaVal) alphaVal.textContent = "0.00";
  
  // Set initial UI state
  const aiStatusText = document.getElementById('aiStatusText');
  if (aiStatusText) aiStatusText.innerHTML = 'Waiting';
  
  const aiLabel = document.getElementById('aiLabel');
  if (aiLabel) aiLabel.innerHTML = 'AI';
  
  const analysis = document.querySelector("main > section:nth-child(2)");
  if (analysis) analysis.style.display = 'none';
  
   const gameOverOverlay = document.getElementById("gameOverOverlay");
   if (gameOverOverlay) gameOverOverlay.style.display = 'none';

   // Setup hero background tiles
   const heroBg = document.getElementById('hero-bg');
   if (heroBg) {
     const fragment = document.createDocumentFragment();
     // Calculate how many tiles we need to fill the viewport (using 40px tiles)
     const tilesX = Math.ceil(window.innerWidth / 40);
     const tilesY = Math.ceil(window.innerHeight / 40);
     const totalTiles = tilesX * tilesY;
     
     for (let i = 0; i < totalTiles; i++) {
       const tile = document.createElement('div');
       tile.className = 'tile';
       fragment.appendChild(tile);
     }
     heroBg.appendChild(fragment);
     startHeroAnimation();
   }

   // Load model
  const ok = await model.load("./assets/model_4x4x4.onnx");
  if (!ok) {
    alert("Cannot load/run the 4×4×4 ONNX in this browser. Try Chrome/Edge with WebGPU enabled (chrome://flags → WebGPU), or export a 2D-only model.");
    return;
  }
  
  // Setup event handlers
  setupEventHandlers();
  
  // Show start choice
  const startPromise = showStartChoice();
  await startPromise;
  game.player = startPlayer;
  
  // AI moves first if selected
  if (startPlayer === -1) {
    await aiMove();
  }
}
