/* ===== Utility Functions ===== */

/**
 * Compute softmax of an array
 */
export const softmax = (arr) => {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / s);
};

/**
 * Normalize array to sum to 1
 */
export const normalize = (xs) => {
  const s = xs.reduce((a, b) => a + b, 0);
  return s > 0 ? xs.map(v => v / s) : xs.map(_ => 0);
};

/**
 * Convert 3D coordinates to flat index
 */
export const toIdx = (x, y, z) => z * 16 + y * 4 + x;

/**
 * Convert flat index to 3D coordinates
 */
export const toXYZ = (idx) => [idx % 4, Math.floor(idx / 4) % 4, Math.floor(idx / 16) % 4];
