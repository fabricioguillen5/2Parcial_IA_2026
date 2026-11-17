const fs = require('node:fs');
const path = require('node:path');
const ort = require('onnxruntime-web/wasm');

function filesIn(folder) {
  return fs.readdirSync(folder, { withFileTypes: true }).flatMap(item => {
    const file = path.join(folder, item.name);
    return item.isDirectory() ? filesIn(file) : item.name.toLowerCase().endsWith('.csv') ? [file] : [];
  });
}

async function main() {
  const folder = path.resolve(process.argv[2]);
  const expectedClass = Number(process.argv[3]);
  if (![0, 1].includes(expectedClass)) throw new Error('Expected class must be 0=rest or 1=activity.');
  ort.env.wasm.numThreads = 1;
  const model = process.argv[4] ? path.resolve(process.argv[4]) : path.resolve(__dirname, '../public/stage1_resnet_domain.onnx');
  const session = await ort.InferenceSession.create(fs.readFileSync(model), { executionProviders: ['wasm'] });
  let pending = [];
  let rest = 0;
  let active = 0;
  let invalid = 0;
  const perFile = [];
  async function flush() {
    if (!pending.length) return;
    const data = new Float32Array(pending.length * 4000);
    pending.forEach((window, index) => data.set(window, index * 4000));
    const output = await session.run({ ecg_input: new ort.Tensor('float32', data, [pending.length, 1, 4000]) });
    const logits = output.logits.data;
    for (let i = 0; i < pending.length; i++) {
      if (logits[i * 2] >= logits[i * 2 + 1]) rest++;
      else active++;
    }
    pending = [];
  }
  try {
    for (const file of filesIn(folder).sort()) {
      const before = { rest, active };
      const samples = fs.readFileSync(file, 'utf8').split(/\r?\n/).flatMap(line => {
        const parts = line.split(';');
        const value = parts.length >= 2 && parts[1].trim() !== '' ? Number(parts[1]) : NaN;
        return Number.isFinite(value) ? [value] : [];
      });
      for (let start = 0; start + 4000 <= samples.length; start += 4000) {
        const window = new Float32Array(samples.slice(start, start + 4000));
        const mean = window.reduce((sum, v) => sum + v, 0) / 4000;
        const sd = Math.sqrt(window.reduce((sum, v) => sum + (v - mean) ** 2, 0) / 4000);
        if (sd <= 1e-4) { invalid++; continue; }
        pending.push(window.map(v => (v - mean) / (sd + 1e-8)));
        if (pending.length === 16) await flush();
      }
      await flush();
      perFile.push({ file: path.basename(file), rest: rest - before.rest, active: active - before.active });
      console.log(JSON.stringify(perFile[perFile.length - 1]));
    }
  } finally {
    await session.release();
  }
  const total = rest + active;
  console.log(JSON.stringify({ expectedClass, windows: total, predictedRest: rest, predictedActive: active, invalid,
    correctFraction: total ? (expectedClass === 0 ? rest : active) / total : null,
    note: 'Descriptive evaluation on supplied data; original training membership is unknown.' }, null, 2));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
