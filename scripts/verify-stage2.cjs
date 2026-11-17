const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ort = require('onnxruntime-web/wasm');

function normalized(signal) {
  assert.equal(signal.length, 4000);
  const samples = new Float32Array(signal);
  const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  const variance = samples.reduce((sum, value) => sum + (value - mean) ** 2, 0) / samples.length;
  return samples.map(value => (value - mean) / (Math.sqrt(variance) + 1e-8));
}

async function main() {
  ort.env.wasm.numThreads = 1;
  const publicDir = path.resolve(__dirname, '..', 'public');
  const options = { executionProviders: ['wasm'] };
  const original = await ort.InferenceSession.create(fs.readFileSync(path.join(publicDir, 'stage2_resnet (1).onnx')), options);
  const adapted = await ort.InferenceSession.create(fs.readFileSync(path.join(publicDir, 'stage2_resnet_4000.onnx')), options);
  try {
    const signals = JSON.parse(fs.readFileSync(path.join(publicDir, 'signals.json'), 'utf8'));
    for (const [name, signal] of Object.entries(signals)) {
      const input = new ort.Tensor('float32', normalized(signal), [1, 1, 4000]);
      if (name === 'Normal') {
        await assert.rejects(original.run({ ecg_input: input }), /dimension|5000|shape/i);
      }
      const result = await adapted.run({ ecg_input: input });
      assert.deepEqual(result.logits.dims, [1, 4]);
      const logits = Array.from(result.logits.data);
      assert(logits.every(Number.isFinite));
      console.log(`${name}: 4000 real samples -> 4 finite logits`);
    }
    console.log('PASS: adapted stage 2 executes on all five supplied ECG signals. Accuracy was not evaluated.');
  } finally {
    await original.release();
    await adapted.release();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
