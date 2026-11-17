const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const ort = require('onnxruntime-web/wasm');

const project = path.resolve(__dirname, '..');
const code = ts.transpileModule(fs.readFileSync(path.join(project, 'lib/ecg-inference.ts'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const inference = {};
new Function('exports', code)(inference);
const { predictWindow } = inference;

function session(logits, classes, label, calls) {
  return { async run({ ecg_input }) {
    calls.push(label);
    assert.deepEqual(ecg_input.dims, [1, 1, 4000]);
    assert(ecg_input.data.every(Number.isFinite));
    return { logits: new ort.Tensor('float32', Float32Array.from(logits), [1, classes]) };
  } };
}

async function main() {
  ort.env.wasm.numThreads = 1;
  const signals = JSON.parse(fs.readFileSync(path.join(project, 'public/signals.json'), 'utf8'));
  // Branches depend on stage 1's output even when the example name disagrees.
  let calls = [];
  let result = await predictWindow(signals.Normal,
    session([0, 5], 2, 'stage1', calls),
    { run() { throw new Error('Stage 2 must not run after activity.'); } }, ort);
  assert.deepEqual(calls, ['stage1']);
  assert.equal(result.stage2Status, 'skipped');
  assert.match(result.rest, /^Active/);

  for (let i = 0; i < 4; i++) {
    calls = [];
    const logits = Array.from({ length: 4 }, (_, j) => i === j ? 5 : 0);
    result = await predictWindow(signals.Active,
      session([5, 0], 2, 'stage1', calls), session(logits, 4, 'stage2', calls), ort);
    assert.deepEqual(calls, ['stage1', 'stage2']);
    assert.equal(result.stage2Status, 'complete');
    assert(result.ecg.startsWith(['Bradycardia', 'Tachycardia', 'AFib', 'Normal'][i]));
  }
  calls = [];
  await assert.rejects(predictWindow(signals.Normal,
    { async run() { throw new Error('Stage 1 failure'); } },
    session([0, 0, 0, 5], 4, 'stage2', calls), ort), /Stage 1 failure/);
  assert.deepEqual(calls, []);
  console.log('PASS: activity stops the cascade; rest executes both stages in order for all four classes; stage 1 errors never bypass the gate.');

  const stage1 = await ort.InferenceSession.create(fs.readFileSync(path.join(project, 'public/stage1_resnet.onnx')), { executionProviders: ['wasm'] });
  const stage2 = await ort.InferenceSession.create(fs.readFileSync(path.join(project, 'public/stage2_resnet_4000.onnx')), { executionProviders: ['wasm'] });
  try {
    for (const [name, signal] of Object.entries(signals)) {
      const prediction = await predictWindow(signal, stage1, stage2, ort);
      assert(['skipped', 'complete'].includes(prediction.stage2Status));
      console.log(`${name}: ${prediction.rest} -> ${prediction.ecg}`);
    }
  } finally {
    await stage1.release();
    await stage2.release();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
