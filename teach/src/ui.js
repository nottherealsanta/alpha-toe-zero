/* ===== UI and Rendering ===== */

import { toIdx, toXYZ, normalize } from './utils.js';

const SIZE = 800;
const CELL = SIZE / 4;

/**
 * Draw all 4 boards
 */
export function drawAll(ctxs, board, visitsPi = null, winning = null) {
  for (let z = 0; z < 4; z++) {
    drawBoard(ctxs[z], board, z, visitsPi, winning);
  }
}

/**
 * Draw a single board (one z-layer)
 */
export function drawBoard(ctx, board, z, visitsPi, winning = null) {
  // Background
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, SIZE, SIZE);
  
  // Grid lines
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--grid').trim() || '#d8dee4';
  ctx.lineWidth = 2;
  for (let i = 1; i < 4; i++) {
    ctx.beginPath();
    ctx.moveTo(i * CELL, 0);
    ctx.lineTo(i * CELL, SIZE);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * CELL);
    ctx.lineTo(SIZE, i * CELL);
    ctx.stroke();
  }
  
  // Heat map (visit probabilities)
  if (visitsPi) {
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        const idx = toIdx(x, y, z);
        const p = visitsPi[idx] || 0;
        if (p > 0) {
          ctx.fillStyle = `rgba(9,105,218,${0.10 + 0.35 * p})`;
          ctx.fillRect(x * CELL + 3, y * CELL + 3, CELL - 6, CELL - 6);
        }
      }
    }
  }
  
  // Pieces (X and O)
  ctx.lineWidth = 10;
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      const idx = toIdx(x, y, z);
      const val = board[idx];
      const px = x * CELL;
      const py = y * CELL;
      const isWinning = winning && winning.includes(idx);
      ctx.strokeStyle = isWinning ? "red" : "#1f2328";
      
      if (val === 1) {
        // Draw X
        ctx.beginPath();
        ctx.moveTo(px + 28, py + 28);
        ctx.lineTo(px + CELL - 28, py + CELL - 28);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(px + CELL - 28, py + 28);
        ctx.lineTo(px + 28, py + CELL - 28);
        ctx.stroke();
      } else if (val === -1) {
        // Draw O
        ctx.beginPath();
        ctx.arc(px + CELL / 2, py + CELL / 2, CELL / 2 - 36, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }
}

/**
 * Convert root visit counts to policy distribution
 */
export function toPiFromVisits(root, tau) {
  const counts = new Array(64).fill(0);
  let sumN = 0;
  for (const [a, child] of root.children.entries()) {
    counts[a] = child.N;
    sumN += child.N;
  }
  if (sumN === 0) return normalize(counts);
  if (tau === 0) {
    const idx = counts.indexOf(Math.max(...counts));
    const pi = new Array(64).fill(0);
    pi[idx] = 1;
    return pi;
  } else {
    const pow = counts.map(n => Math.pow(n, 1.0 / Math.max(tau, 1e-6)));
    const s = pow.reduce((a, b) => a + b, 0);
    return s > 0 ? pow.map(v => v / s) : normalize(counts);
  }
}

/**
 * Update stats table
 */
export function updateStats(root, tau, game, ctxs, drawAllFn) {
  const pi = toPiFromVisits(root, tau);
  drawAllFn(ctxs, game.board, pi);

  const parentN = 1 + root.N;
  const rows = [];
  for (const [a, child] of root.children.entries()) {
    const U = child.prior * (Math.sqrt(parentN) / (1 + child.N)) * 1.0;
    const [x, y, z] = toXYZ(a);
    const row = `
      <tr>
        <td>(${x},${y},${z})</td>
        <td>${child.N}</td>
        <td>${child.QFromParent().toFixed(3)}</td>
        <td>${child.prior.toFixed(3)}</td>
        <td>${U.toFixed(3)}</td>
        <td>${(pi[a] * 100).toFixed(1)}%</td>
      </tr>`;
    rows.push([child.N, row]);
  }
  rows.sort((x, y) => y[0] - x[0]);
  
  const statsBody = document.getElementById('statsBody');
  if (statsBody) {
    statsBody.innerHTML = rows.slice(0, 16).map(x => x[1]).join("");
  }
  
  const predV = document.getElementById('predV');
  if (predV) {
    predV.textContent = root.children.size ? avgRootQ(root).toFixed(3) : "—";
  }
}

/**
 * Calculate average Q value from root
 */
function avgRootQ(root) {
  let num = 0, den = 0;
  for (const child of root.children.values()) {
    num += child.N * child.QFromParent();
    den += child.N;
  }
  return den > 0 ? num / den : 0;
}

/**
 * Show game over overlay
 */
export function showGameOver(game, ctxs, drawAllFn) {
  const score = game.score();
  let text = "It's a draw!";
  if (score === 1) text = "You win!";
  else if (score === -1) text = "AI wins!";
  
  const gameOverText = document.getElementById("gameOverText");
  if (gameOverText) gameOverText.textContent = text;
  
  const gameOverOverlay = document.getElementById("gameOverOverlay");
  if (gameOverOverlay) gameOverOverlay.style.display = 'flex';
  
  const winning = game.getWinningLine();
  drawAllFn(ctxs, game.board, null, winning);
}

/**
 * Animation for AI thinking
 */
let dotsInterval;

export function startThinkingAnimation() {
  let dots = '';
  dotsInterval = setInterval(() => {
    dots = dots.length < 3 ? dots + '.' : '';
    const aiStatusText = document.getElementById('aiStatusText');
    if (aiStatusText) aiStatusText.innerHTML = 'Thinking' + dots;
  }, 200);
}

export function stopThinkingAnimation() {
  clearInterval(dotsInterval);
  const aiStatusText = document.getElementById('aiStatusText');
  if (aiStatusText) aiStatusText.innerHTML = 'Waiting';
}

/**
 * Show start choice overlay
 */
export function showStartChoice() {
  return new Promise((resolve) => {
    window.resolveStart = resolve;
    const overlay = document.getElementById("startOverlay");
    if (overlay) overlay.style.display = 'flex';
  });
}
