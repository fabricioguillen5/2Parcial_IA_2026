'use client';

import { useState, useEffect, useRef } from 'react';
import type * as ort from 'onnxruntime-web/wasm';

type OrtRuntime = typeof import('onnxruntime-web/wasm');

const WINDOW_SIZE_STAGE1 = 4000;
const WINDOW_SIZE_STAGE2 = 5000;
// Public filenames match the supplied files, including the stage 2 suffix.
const STAGE1_MODEL_URL = '/stage1_resnet.onnx';
const STAGE2_MODEL_URL = '/stage2_resnet%20(1).onnx';
const ECG_CLASSES = ['Bradycardia', 'Tachycardia', 'AFib', 'Normal'];
type Signals = Record<string, number[]>;
type Predictions = { rest: string; ecg: string };

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

const predictWindow = async (
  signal: number[],
  restSession: ort.InferenceSession,
  ecgSession: ort.InferenceSession,
  runtime: OrtRuntime
): Promise<Predictions> => {
  const input1 = new runtime.Tensor('float32', normalizeWindow(signal, WINDOW_SIZE_STAGE1), [1, 1, WINDOW_SIZE_STAGE1]);
  const restResults = await restSession.run({ ecg_input: input1 });
  const restProbs = softmax(readLogits(restResults, 2));
  // Colab class order: 0=Resting, 1=Active. Argmax matches evaluation (including ties).
  const restClass = argmax(restProbs);
  const rest = `${restClass === 0 ? 'Resting' : 'Active'} (${(restProbs[restClass] * 100).toFixed(1)}%)`;
  if (restClass === 1) return { rest, ecg: 'Skipped (Patient Active)' };
  if (signal.length < WINDOW_SIZE_STAGE2) {
    return { rest, ecg: `Se requieren 5000 muestras reales; señal disponible: ${signal.length}.` };
  }
  const input2 = new runtime.Tensor('float32', normalizeWindow(signal, WINDOW_SIZE_STAGE2), [1, 1, WINDOW_SIZE_STAGE2]);
  const ecgResults = await ecgSession.run({ ecg_input: input2 });
  const ecgProbs = softmax(readLogits(ecgResults, 4));
  const ecgClass = argmax(ecgProbs);
  // No manual AFib override: this is the same argmax rule evaluated in Colab.
  return { rest, ecg: `${ECG_CLASSES[ecgClass]} (${(ecgProbs[ecgClass] * 100).toFixed(1)}%)` };
};

async function fetchChecked(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`No se pudo cargar ${url} (HTTP ${response.status}).`);
  return response;
}

function validateSignals(value: unknown): Signals {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('signals.json debe contener un objeto de señales.');
  }
  const signals = value as Record<string, unknown>;
  for (const key of ['AFib', 'Taquicardia', 'Bradicardia', 'Normal', 'Active']) {
    const signal = signals[key];
    if (!Array.isArray(signal) || signal.length < WINDOW_SIZE_STAGE1 || !signal.every(sample => typeof sample === 'number' && Number.isFinite(sample))) {
      throw new Error(`La señal ${key} debe contener al menos 4000 muestras numéricas finitas.`);
    }
  }
  return signals as Signals;
}

export default function Home() {
  const [selectedSignal, setSelectedSignal] = useState('Normal');
  const [rawSignals, setRawSignals] = useState<Signals | null>(null);
  const [displayWindow, setDisplayWindow] = useState<number[]>([]);
  const [predictions, setPredictions] = useState<Predictions>({ rest: 'Waiting...', ecg: 'Waiting...' });
  const [status, setStatus] = useState('Loading data & models...');
  const restSessionRef = useRef<ort.InferenceSession | null>(null);
  const ecgSessionRef = useRef<ort.InferenceSession | null>(null);
  const inferenceRef = useRef<Promise<void> | null>(null);
  const runtimeRef = useRef<OrtRuntime | null>(null);

  useEffect(() => {
    let cancelled = false;
    // Local sessions prevent an older Strict Mode cleanup from clearing a newer session.
    let restSession: ort.InferenceSession | null = null;
    let ecgSession: ort.InferenceSession | null = null;
    async function init() {
      try {
        // useEffect runs only in the browser. A top-level runtime import also
        // executes during Next.js prerendering and breaks WASM URL resolution.
        const runtime = await import('onnxruntime-web/wasm');
        if (cancelled) return;
        runtime.env.wasm.numThreads = 1;
        // Use the exact installed release: /ort/ fails unless its files were copied.
        // JS and WASM must come from the same ONNX Runtime build.
        const runtimeVersion = runtime.env.versions.web;
        if (!runtimeVersion || !/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(runtimeVersion)) {
          throw new Error('No se pudo determinar la versión de onnxruntime-web para cargar WASM.');
        }
        runtime.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${runtimeVersion}/dist/`;
        const [jsonResponse, buffer1, buffer2] = await Promise.all([
          fetchChecked('/signals.json'),
          fetchChecked(STAGE1_MODEL_URL).then(response => response.arrayBuffer()),
          fetchChecked(STAGE2_MODEL_URL).then(response => response.arrayBuffer())
        ]);
        const signals = validateSignals(await jsonResponse.json());
        if (cancelled) return;
        restSession = await runtime.InferenceSession.create(buffer1, { executionProviders: ['wasm'] });
        if (cancelled) { await restSession.release(); return; }
        ecgSession = await runtime.InferenceSession.create(buffer2, { executionProviders: ['wasm'] });
        if (cancelled) {
          await restSession.release(); await ecgSession.release(); return;
        }
        for (const session of [restSession, ecgSession]) {
          if (!session.inputNames.includes('ecg_input') || !session.outputNames.includes('logits')) {
            throw new Error('El archivo ONNX no coincide con los modelos inspeccionados: ecg_input → logits.');
          }
        }
        restSessionRef.current = restSession;
        ecgSessionRef.current = ecgSession;
        runtimeRef.current = runtime;
        setStatus('Ready');
        // Starts the monitoring effect only after both models are ready.
        setRawSignals(signals);
      } catch (error) {
        if (restSession) await restSession.release();
        if (ecgSession) await ecgSession.release();
        if (!cancelled) {
          console.error('Init Error:', error);
          setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }
    void init();
    return () => {
      cancelled = true;
      // Initialization handles cancellation itself until refs have been assigned.
      if (restSessionRef.current === restSession && restSession) {
        restSessionRef.current = null;
        ecgSessionRef.current = null;
        const pending = inferenceRef.current;
        void (async () => {
          await pending;
          await restSession?.release();
          await ecgSession?.release();
        })();
      }
    };
  }, []);

  useEffect(() => {
    const fullSignal = rawSignals?.[selectedSignal];
    const restSession = restSessionRef.current;
    const ecgSession = ecgSessionRef.current;
    const runtime = runtimeRef.current;
    if (!fullSignal || !restSession || !ecgSession || !runtime) return;
    let active = true;
    let scrollIndex = 0;
    let lastInferenceTime = -Infinity;
    setPredictions({ rest: 'Analyzing...', ecg: 'Analyzing...' });
    const tick = () => {
      // Circular scrolling is a demo playback. Never extend short records to 5000.
      const length = Math.min(WINDOW_SIZE_STAGE2, fullSignal.length);
      const windowData = Array.from({ length }, (_, i) => fullSignal[(scrollIndex + i) % fullSignal.length]);
      setDisplayWindow(windowData);
      scrollIndex = (scrollIndex + 20) % fullSignal.length;
      const now = Date.now();
      if (now - lastInferenceTime < 2000 || inferenceRef.current) return;
      lastInferenceTime = now;
      inferenceRef.current = (async () => {
        try {
          const result = await predictWindow(windowData, restSession, ecgSession, runtime);
          // Ignore a prediction from a tab that the user has already left.
          if (active) { setPredictions(result); setStatus('Ready'); }
        } catch (error) {
          if (active) {
            console.error('Inference error:', error);
            setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`);
            setPredictions({ rest: 'Error de inferencia', ecg: 'Revisar señal y modelos' });
          }
        } finally {
          inferenceRef.current = null;
        }
      })();
    };
    tick();
    const timer = setInterval(tick, 40);
    return () => { active = false; clearInterval(timer); };
  }, [rawSignals, selectedSignal]);

  const handleTabClick = (key: string) => setSelectedSignal(key);

  const renderECGPath = () => {
    if (displayWindow.length === 0) return "M 0 100 L 800 100"; 
    const xStep = 800 / displayWindow.length;
    let path = `M 0 100`;
    const maxVal = Math.max(...displayWindow.map(Math.abs)) || 1;
    const yScale = 80 / maxVal; 
    displayWindow.forEach((val, i) => { 
      path += ` L ${i * xStep} ${100 - (val * yScale)}`; 
    });
    return path;
  };

  return (
    <main className="min-h-screen bg-gray-50 flex flex-col items-center p-8 font-sans">
      <div className="w-full max-w-5xl">
        <div className="flex justify-between items-end mb-6">
          <h1 className="text-3xl font-light text-gray-700 tracking-wide">ECG Classifier Demo</h1>
          <span className={`text-sm px-3 py-1 rounded-full ${status === 'Ready' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
            {status}
          </span>
        </div>

        <div className="border-2 border-gray-400 p-6 bg-white shadow-sm">
          <div className="mb-6 relative">
            <h2 className="text-xl font-light text-gray-600 mb-2 absolute top-2 left-4 z-10">ECG Window (Real-Time)</h2>
            <div className="w-full h-64 border border-gray-400 bg-gray-50 overflow-hidden relative">
              <svg viewBox="0 0 800 200" className="w-full h-full stroke-gray-700 stroke-2 fill-none" preserveAspectRatio="none">
                <line x1="0" y1="100" x2="800" y2="100" stroke="#9ca3af" strokeWidth="1" strokeDasharray="5,5" />
                <path d={renderECGPath()} />
              </svg>
            </div>
          </div>

          <div className="flex gap-4 mb-8">
            {['AFib', 'Taquicardia', 'Bradicardia', 'Normal', 'Active'].map((key) => (
              <button key={key} disabled={!rawSignals} onClick={() => handleTabClick(key)}
                className={`flex-1 py-2 px-4 border-2 rounded-sm text-lg font-light transition-colors
                  ${selectedSignal === key ? 'border-gray-600 bg-gray-200 text-gray-900' : 'border-gray-400 text-gray-600 hover:bg-gray-100'}`}>
                {key}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-4 border-t-2 border-gray-200 pt-6">
            <div className="p-4 bg-blue-50 border border-blue-200 rounded">
              <h3 className="text-sm text-blue-600 font-semibold uppercase tracking-wider mb-1">Stage 1: Rest Gatekeeper (ResNet)</h3>
              <p className="text-xl font-medium text-blue-900">{predictions.rest}</p>
            </div>
            <div className="p-4 bg-green-50 border border-green-200 rounded">
              <h3 className="text-sm text-green-600 font-semibold uppercase tracking-wider mb-1">Stage 2: Arrhythmia (ResNet)</h3>
              <p className={`text-xl font-medium ${predictions.ecg.includes('Skipped') ? 'text-gray-500 italic' : 'text-green-900'}`}>
                {predictions.ecg}
              </p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
