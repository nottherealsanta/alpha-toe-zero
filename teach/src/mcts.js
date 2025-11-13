/* ===== Monte Carlo Tree Search (MCTS) ===== */

import { softmax } from './utils.js';
import { Game } from './game.js';

export class Node {
  constructor(state, parent = null, action = null, prior = 0) {
    this.state = state; // {board:Int8Array, player:int}
    this.parent = parent;
    this.action = action;
    this.prior = prior;
    this.children = new Map(); // action -> Node
    this.N = 0;
    this.W = 0; // sum of values from this node's perspective
  }

  get Q() {
    return this.N ? this.W / this.N : 0;
  }

  QFromParent() {
    return -this.Q;
  }
}

export class MCTS {
  constructor(game, model, c_puct = 1.25, alpha = 0.3, eps = 0.25) {
    this.game = game;
    this.model = model;
    this.c_puct = c_puct;
    this.alpha = alpha;
    this.eps = eps;
  }

  async search(simulations = 400) {
    const root = new Node({ board: this.game.board.slice(), player: this.game.player });
    await this.expandWithNoise(root);

    for (let s = 0; s < simulations; s++) {
      let node = root;
      const g = new Game(root.state.board, root.state.player);

      while (node.children.size > 0 && !g.over()) {
        node = this.select(node);
        g.makeMove(node.action);
      }

      if (!g.over()) {
        const v = await this.expand(node, g);
        this.backup(node, v);
      } else {
        const terminalV = g.score() * g.player;
        this.backup(node, terminalV);
      }
    }
    return root;
  }

  select(node) {
    const parentVisits = 1 + node.N;
    let best = null, bestScore = -Infinity;
    for (const child of node.children.values()) {
      const U = this.c_puct * child.prior * (Math.sqrt(parentVisits) / (1 + child.N));
      const score = child.QFromParent() + U;
      if (score > bestScore) {
        bestScore = score;
        best = child;
      }
    }
    return best;
  }

  async expand(node, game) {
    const { v, logits } = await this.model.predict(game);
    const legal = game.validMoves();
    const masked = legal.map(a => logits[a]);
    const probsL = softmax(masked);
    for (let i = 0; i < legal.length; i++) {
      const a = legal[i];
      if (!node.children.has(a)) {
        const childState = { board: game.board.slice(), player: game.player };
        const child = new Node(childState, node, a, probsL[i]);
        node.children.set(a, child);
      }
    }
    return v;
  }

  async expandWithNoise(root) {
    const g = new Game(root.state.board, root.state.player);
    if (g.over()) return;
    const { logits } = await this.model.predict(g);
    const legal = g.validMoves();
    if (legal.length === 0) return;
    const priors = softmax(legal.map(a => logits[a]));
    const noise = this.dirichlet(this.alpha, legal.length);
    const mixed = priors.map((p, i) => (1 - this.eps) * p + this.eps * noise[i]);
    for (let i = 0; i < legal.length; i++) {
      const a = legal[i];
      const childState = { board: g.board.slice(), player: g.player };
      const child = new Node(childState, root, a, mixed[i]);
      root.children.set(a, child);
    }
  }

  backup(node, vLeaf) {
    let cur = node;
    let v = vLeaf;
    while (cur) {
      cur.N += 1;
      cur.W += v;
      v = -v;
      cur = cur.parent;
    }
  }

  dirichlet(alpha, k) {
    const samples = new Array(k).fill(0).map(() => this.gamma(alpha, 1));
    const sum = samples.reduce((a, b) => a + b, 0);
    return samples.map(x => x / sum);
  }

  gamma(shape, scale) {
    if (shape < 1) {
      const u = Math.random();
      return this.gamma(1 + shape, scale) * Math.pow(u, 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      let x, v;
      do {
        x = this.normal();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = Math.random();
      if (u < 1 - 0.0331 * (x * x) * (x * x)) return scale * d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return scale * d * v;
    }
  }

  normal() {
    const u = 1 - Math.random();
    const v = 1 - Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
}
