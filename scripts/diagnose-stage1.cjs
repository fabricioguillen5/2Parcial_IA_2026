const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const ts = require('typescript');
const ort = require('onnxruntime-web/wasm');

async function main() {
  const project = path.resolve(__dirname, '..');
  const source = fs.readFileSync(path.join(project, 'lib/ecg-inference.ts'), 'utf8');
  const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
  const inference = {};
  new Function('exports', compiled)(inference);
  ort.env.wasm.numThreads = 1;
  const bytes = fs.readFileSync(process.argv[2] ? path.resolve(process.argv[2]) : path.join(project, 'public', inference.STAGE1_MODEL_URL.slice(1)));
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  const signals = JSON.parse(fs.readFileSync(path.join(project, 'public/signals.json'), 'utf8'));
  const cases = [];
  try {
    for (const [name, signal] of Object.entries(signals)) {
      let observedLogits;
      const instrumentedStage1 = {
        async run(feeds) {
          const result = await session.run(feeds);
          observedLogits = Array.from(result.logits.data);
          return result;
        },
      };
      const stage2 = { async run() { return { logits: new ort.Tensor('float32', [0, 0, 0, 0], [1, 4]) }; } };
      const prediction = await inference.predictWindow(signal, instrumentedStage1, stage2, ort);
      const [rest, active] = observedLogits;
      const pActive = 1 / (1 + Math.exp(rest - active));
      cases.push({ example: name, samples: signal.length, logits: observedLogits, pActive, stage1Prediction: prediction.rest, cascadeDecision: prediction.stage2Status });
    }
  } finally {
    await session.release();
  }
  const report = {
    modelSha256: crypto.createHash('sha256').update(bytes).digest('hex'),
    classOrder: ['Resting', 'Active'],
    preprocessing: 'float32; per-window mean removal and population standard deviation + 1e-8; 4000 samples',
    cases,
    note: 'Stage 2 output is stubbed here: this report diagnoses Stage 1 only. The example names are not evidence of physical activity labels. Validation data and the matching checkpoint metadata are needed before changing weights, polarity, labels or threshold.',
  };
  console.log(JSON.stringify(report, null, 2));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
