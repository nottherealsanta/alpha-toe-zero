/* ===== ONNX Model Wrapper ===== */

import { toIdx } from './utils.js';

export class ONNXAlphaZero {
  constructor() {
    this.session = null;
    this.inferenceQueue = Promise.resolve();
  }

  async load(url = "./assets/model_4x4x4.onnx") {
    // Try WebGPU first, fall back to WASM on error
    try {
      // First attempt: WebGPU
      if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
        try {
          this.session = await ort.InferenceSession.create(url, {
            executionProviders: ['webgpu']
          });
          console.log('Using WebGPU backend');
          return true;
        } catch (webgpuError) {
          console.warn('WebGPU failed, falling back to WASM:', webgpuError);
        }
      }

      // Fallback: WASM only
      this.session = await ort.InferenceSession.create(url, {
        executionProviders: ['wasm']
      });
      console.log('Using WASM backend');
      return true;
    } catch (e) {
      console.error("ONNX load failed:", e);
      this.session = null;
      return false;
    }
  }

  encode(game) {
    // planes C=2, Z=4, Y=4, X=4 (channel-first)
    const cur = game.player;
    const arr = new Float32Array(1 * 2 * 4 * 4 * 4);
    for (let z = 0; z < 4; z++) {
      for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 4; x++) {
          const idx = toIdx(x, y, z);
          const v = game.board[idx];
          const off0 = 0 * 64 + z * 16 + y * 4 + x;
          const off1 = 1 * 64 + z * 16 + y * 4 + x;
          arr[off0] = (v === cur) ? 1.0 : 0.0;
          arr[off1] = (v === -cur) ? 1.0 : 0.0;
        }
      }
    }
    return arr;
  }

  async predict(game) {
    if (!this.session) throw new Error("ONNX session not initialized");

    // Queue inference to prevent concurrent session access
    return new Promise((resolve, reject) => {
      this.inferenceQueue = this.inferenceQueue.then(async () => {
        try {
          const input = this.encode(game);
          const feeds = { "input": new ort.Tensor('float32', input, [1, 2, 4, 4, 4]) };
          const out = await this.session.run(feeds);
          const v = out["value"].data[0];
          const logits = Array.from(out["policy_logits"].data);
          resolve({ v, logits });
        } catch (error) {
          reject(error);
        }
      }).catch(reject);
    });
  }
}
