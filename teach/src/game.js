/* ===== Game Logic for 4×4×4 Qubic ===== */

import { toIdx } from './utils.js';

export class Game {
  constructor(board = null, player = 1) {
    this.board = board ? Int8Array.from(board) : new Int8Array(64).fill(0);
    this.player = player; // 1 or -1
    if (!Game.WIN_LINES) Game.WIN_LINES = Game.computeWinLines();
  }

  static computeWinLines() {
    const lines = [];
    
    // Axis lines (48 lines)
    for (let z = 0; z < 4; z++) {
      for (let y = 0; y < 4; y++) {
        lines.push([0, 1, 2, 3].map(x => toIdx(x, y, z)));
      }
    }
    for (let z = 0; z < 4; z++) {
      for (let x = 0; x < 4; x++) {
        lines.push([0, 1, 2, 3].map(y => toIdx(x, y, z)));
      }
    }
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        lines.push([0, 1, 2, 3].map(z => toIdx(x, y, z)));
      }
    }
    
    // Plane diagonals (24 lines)
    for (let z = 0; z < 4; z++) {
      lines.push([0, 1, 2, 3].map(i => toIdx(i, i, z)));
      lines.push([0, 1, 2, 3].map(i => toIdx(i, 3 - i, z)));
    }
    for (let y = 0; y < 4; y++) {
      lines.push([0, 1, 2, 3].map(i => toIdx(i, y, i)));
      lines.push([0, 1, 2, 3].map(i => toIdx(i, y, 3 - i)));
    }
    for (let x = 0; x < 4; x++) {
      lines.push([0, 1, 2, 3].map(i => toIdx(x, i, i)));
      lines.push([0, 1, 2, 3].map(i => toIdx(x, i, 3 - i)));
    }
    
    // Space diagonals (4 lines)
    lines.push([0, 1, 2, 3].map(i => toIdx(i, i, i)));
    lines.push([0, 1, 2, 3].map(i => toIdx(i, i, 3 - i)));
    lines.push([0, 1, 2, 3].map(i => toIdx(i, 3 - i, i)));
    lines.push([0, 1, 2, 3].map(i => toIdx(3 - i, i, i)));
    
    return lines;
  }

  clone() {
    return new Game(this.board, this.player);
  }

  validMoves() {
    const moves = [];
    for (let i = 0; i < 64; i++) {
      if (this.board[i] === 0) moves.push(i);
    }
    return moves;
  }

  makeMove(action) {
    if (this.board[action] !== 0) return false;
    this.board[action] = this.player;
    this.player = -this.player;
    return true;
  }

  score() {
    const b = this.board;
    for (const L of Game.WIN_LINES) {
      const s = b[L[0]] + b[L[1]] + b[L[2]] + b[L[3]];
      if (s === 4) return 1;
      if (s === -4) return -1;
    }
    return 0;
  }

  getWinningLine() {
    const b = this.board;
    for (const L of Game.WIN_LINES) {
      const s = b[L[0]] + b[L[1]] + b[L[2]] + b[L[3]];
      if (s === 4 || s === -4) return L;
    }
    return null;
  }

  over() {
    return this.score() !== 0 || this.validMoves().length === 0;
  }
}
