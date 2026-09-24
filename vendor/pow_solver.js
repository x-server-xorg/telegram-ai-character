// CLI-обёртка над официальным WASM-солвером DeepSeek (DeepSeekHashV1).
// Использование:  node pow_solver.js <challenge.json
// Печатает в stdout готовый заголовок x-ds-pow-response (base64).
const fs = require("fs");
const path = require("path");
const PowSolver = require("./pow-solver-lib.js");

function readStdin() {
  try {
    return fs.readFileSync(0, "utf8");
  } catch {
    return "";
  }
}

(async () => {
  const raw = readStdin() || process.argv[2] || "";
  if (!raw) {
    process.stderr.write("usage: node pow_solver.js '{\"challenge\":...}'\n");
    process.exit(2);
  }
  let challenge;
  try {
    const parsed = JSON.parse(raw);
    challenge = parsed.challenge && typeof parsed.challenge === "object"
      ? parsed.challenge
      : parsed;
  } catch (e) {
    process.stderr.write("invalid json: " + e.message + "\n");
    process.exit(2);
  }

  const solver = new PowSolver(path.join(__dirname, "deepseek_pow.wasm"));
  await solver.init();
  const payload = solver.solve(challenge);
  if (challenge.target_path) payload.target_path = challenge.target_path;

  const blob = JSON.stringify(payload);
  process.stdout.write(Buffer.from(blob, "utf8").toString("base64"));
})().catch((e) => {
  process.stderr.write("pow error: " + (e && e.message ? e.message : e) + "\n");
  process.exit(1);
});
