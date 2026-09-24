const fs = require('fs');
const path = require('path');

let cachedUint8Memory = null;
let cachedDataView = null;

function getUint8Memory(wasm) {
  if (cachedUint8Memory === null || cachedUint8Memory.buffer !== wasm.memory.buffer) {
    cachedUint8Memory = new Uint8Array(wasm.memory.buffer);
  }
  return cachedUint8Memory;
}

function getDataView(wasm) {
  if (cachedDataView === null || cachedDataView.buffer !== wasm.memory.buffer) {
    cachedDataView = new DataView(wasm.memory.buffer);
  }
  return cachedDataView;
}

function passStringToWasm(wasm, str) {
  const enc = new TextEncoder();
  const buf = enc.encode(str);
  const ptr = wasm.__wbindgen_export_0(buf.length, 1) >>> 0;
  getUint8Memory(wasm).set(buf, ptr);
  return { ptr, len: buf.length };
}

function solvePow(wasm, algorithm, challenge, salt, difficulty, expireAt) {
  if (algorithm !== "DeepSeekHashV1") {
    throw new Error(`Unsupported algorithm: ${algorithm}`);
  }

  const prefix = `${salt}_${expireAt}_`;

  const stackPtr = wasm.__wbindgen_add_to_stack_pointer(-16);

  try {
    const challengePtr = passStringToWasm(wasm, challenge);
    const prefixPtr = passStringToWasm(wasm, prefix);

    wasm.wasm_solve(
      stackPtr,
      challengePtr.ptr, challengePtr.len,
      prefixPtr.ptr, prefixPtr.len,
      difficulty
    );

    const status = getDataView(wasm).getInt32(stackPtr + 0, true);
    const answer = getDataView(wasm).getFloat64(stackPtr + 8, true);

    if (status === 0) {
      throw new Error("No solution found (Wasm returned 0)");
    }

    return answer;
  } finally {
    wasm.__wbindgen_add_to_stack_pointer(16);
  }
}

class PowSolver {
  constructor(wasmPath) {
    this.wasmPath = wasmPath || path.join(__dirname, 'sha3_wasm_bg.7b9ca65ddd.wasm');
    this.wasm = null;
  }

  async init() {
    const wasmBuffer = fs.readFileSync(this.wasmPath);
    const { instance } = await WebAssembly.instantiate(wasmBuffer, { wbg: {} });
    this.wasm = instance.exports;
    return this;
  }

  solve(challengeData) {
    if (!this.wasm) {
      throw new Error("WASM not initialized. Call init() first.");
    }

    const startTime = Date.now();

    const answer = solvePow(
      this.wasm,
      challengeData.algorithm,
      challengeData.challenge,
      challengeData.salt,
      challengeData.difficulty,
      challengeData.expire_at
    );

    const endTime = Date.now();

    return {
      algorithm: challengeData.algorithm,
      challenge: challengeData.challenge,
      salt: challengeData.salt,
      answer: answer,
      signature: challengeData.signature
    };
  }
}

module.exports = PowSolver;