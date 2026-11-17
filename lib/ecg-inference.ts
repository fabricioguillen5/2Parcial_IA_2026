import type * as ort from 'onnxruntime-web/wasm';

export type OrtRuntime = typeof import('onnxruntime-web/wasm');

export const WINDOW_SIZE_STAGE1 = 4000;
export const WINDOW_SIZE_STAGE2 = 4000;
// The original stage 2 model is preserved; this variant declares 4000 inputs.
export const STAGE1_MODEL_URL = '/stage1_resnet_domain.onnx';
export const STAGE2_MODEL_URL = '/stage2_resnet_4000.onnx';
const ECG_CLASSES = ['Bradicardia', 'Taquicardia', 'AFib', 'Normal'];
export type Signals = Record<string, number[]>;
export const DOCUMENTED_REST = 'Reposo documentado en la toma del ECG';
export type Predictions = { rest: string; ecg: string; stage2Status: 'waiting' | 'skipped' | 'complete' | 'error' };

const softmax = (logits: number[]): number[] => {
  const max = Math.max(...logits);
  const exps = logits.map(value => Math.exp(value - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(value => value / sum);
};

const readLogits = (
  results: Record<string, ort.Tensor>,
  expectedClasses: number
): number[] => {
  const tensor = results.logits;
  if (!tensor || tensor.type !== 'float32') {
    throw new Error(`Se esperaba la salida float32 "logits". Salidas: ${Object.keys(results).join(', ')}`);
  }
  if (tensor.dims.length !== 2 || tensor.dims[0] !== 1 || tensor.dims[1] !== expectedClasses) {
    throw new Error(`Forma de salida incorrecta: [${tensor.dims.join(', ')}]; se esperaba [1, ${expectedClasses}].`);
  }
  const logits = Array.from(tensor.data as Float32Array);
  if (logits.length !== expectedClasses || !logits.every(Number.isFinite)) {
    throw new Error('El modelo devolvió logits no válidos.');
  }
  return logits;
};

// Same per-window population standard deviation and epsilon as make_loader in Colab.
// Round raw samples to float32 before normalizing, as the training datasets do.
const normalizeWindow = (signal: number[], length: number): Float32Array => {
  if (signal.length < length) {
    throw new Error(`Se necesitan ${length} muestras reales; se recibieron ${signal.length}.`);
  }
  const samples = new Float32Array(signal.slice(0, length));
  if (!samples.every(Number.isFinite)) throw new Error('La señal contiene valores no finitos.');
  let mean = 0;
  for (const value of samples) mean += value;
  mean /= length;
  let variance = 0;
  for (const value of samples) variance += (value - mean) ** 2;
  const denominator = Math.sqrt(variance / length) + 1e-8;
  return samples.map(value => (value - mean) / denominator);
};

const argmax = (values: number[]): number => {
  let best = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
};

const formatConfidence = (probability: number): string => probability >= 0.999
  ? '>99,9 %'
  : `${(probability * 100).toFixed(1).replace('.', ',')} %`;

export const predictWindow = async (
  signal: number[],
  restSession: ort.InferenceSession,
  ecgSession: ort.InferenceSession,
  runtime: OrtRuntime,
  acquisition: 'documented-rest' | 'unknown' = 'unknown'
): Promise<Predictions> => {
  let rest = DOCUMENTED_REST;
  if (acquisition === 'unknown') {
    const input1 = new runtime.Tensor('float32', normalizeWindow(signal, WINDOW_SIZE_STAGE1), [1, 1, WINDOW_SIZE_STAGE1]);
    const restResults = await restSession.run({ ecg_input: input1 });
    const restProbs = softmax(readLogits(restResults, 2));
    const restClass = argmax(restProbs);
    rest = `${restClass === 0 ? 'Reposo' : 'Actividad'} (${formatConfidence(restProbs[restClass])})`;
    if (restClass === 1) {
      return { rest, ecg: 'Omitida: Stage 1 detectó actividad', stage2Status: 'skipped' };
    }
  }
  const input2 = new runtime.Tensor('float32', normalizeWindow(signal, WINDOW_SIZE_STAGE2), [1, 1, WINDOW_SIZE_STAGE2]);
  const ecgResults = await ecgSession.run({ ecg_input: input2 });
  const ecgProbs = softmax(readLogits(ecgResults, 4));
  const ecgClass = argmax(ecgProbs);
  return {
    rest,
    ecg: `${ECG_CLASSES[ecgClass]} (${formatConfidence(ecgProbs[ecgClass])})`,
    stage2Status: 'complete'
  };
};
