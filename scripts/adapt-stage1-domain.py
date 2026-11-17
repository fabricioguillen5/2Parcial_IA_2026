"""Refit the ResNet classification head using wearable data and documented resting clinical ECGs.

Run in a Python environment with numpy, onnx, onnxruntime, and pyarrow.
Raw data and embeddings stay in --work-dir, outside the public application.
"""
import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import pyarrow.parquet as pq


SOURCE = 'https://physionet.org/content/ecg-arrhythmia/1.0.0/'
WINDOW = 4000


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)


def clinical_rows(filename):
    for batch in pq.ParquetFile(filename).iter_batches(batch_size=128, columns=['record_id', 'label', 'signal']):
        yield from batch.to_pylist()


def metrics(y, prediction):
    cm = np.zeros((2, 2), dtype=int)
    for truth, pred in zip(y, prediction):
        cm[int(truth), int(pred)] += 1
    recalls = np.diag(cm) / np.maximum(1, cm.sum(axis=1))
    precisions = np.diag(cm) / np.maximum(1, cm.sum(axis=0))
    f1 = 2 * recalls * precisions / np.maximum(1e-12, recalls + precisions)
    present = cm.sum(axis=1) > 0
    return dict(n=int(len(y)), confusion=cm.tolist(), rest_recall=float(recalls[0]),
                active_recall=float(recalls[1]), macro_f1=float(f1[present].mean()),
                balanced_accuracy=float(recalls[present].mean()))


def fit_domain_mlp(x, y, weights, validation_x, validation_y, validation_origin, base_activity):
    rng = np.random.default_rng(42)
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    a = ((x - mean) / scale).astype(np.float32)
    va = ((validation_x - mean) / scale).astype(np.float32)
    parameters = [
        rng.normal(0, np.sqrt(2 / x.shape[1]), (x.shape[1], 32)).astype(np.float32),
        np.zeros(32, dtype=np.float32),
        rng.normal(0, np.sqrt(2 / 32), (32, 1)).astype(np.float32),
        np.zeros(1, dtype=np.float32),
    ]
    momentum = [np.zeros_like(value) for value in parameters]
    velocity = [np.zeros_like(value) for value in parameters]
    best_score, stale, step = -1, 0, 0
    best = None
    history = []
    for epoch in range(1, 161):
        order = rng.permutation(len(a))
        for start in range(0, len(a), 256):
            index = order[start:start + 256]
            batch = a[index]
            w1, b1, w2, b2 = parameters
            hidden_input = batch @ w1 + b1
            hidden = np.maximum(0, hidden_input)
            z = hidden @ w2 + b2
            probability = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
            delta = (probability - y[index, None]) * weights[index, None] / len(index)
            hidden_delta = (delta @ w2.T) * (hidden_input > 0)
            gradients = [batch.T @ hidden_delta + 1e-4 * w1, hidden_delta.sum(axis=0),
                         hidden.T @ delta + 1e-4 * w2, delta.sum(axis=0)]
            step += 1
            for value, gradient, m, v in zip(parameters, gradients, momentum, velocity):
                m *= 0.9; m += 0.1 * gradient
                v *= 0.999; v += 0.001 * gradient ** 2
                value -= 0.001 * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        w1, b1, w2, b2 = parameters
        gate_logit = (np.maximum(0, va @ w1 + b1) @ w2 + b2)[:, 0]
        clinical = 1 / (1 + np.exp(-np.clip(gate_logit, -40, 40)))
        prediction = ((1 - clinical) * base_activity > 0.5).astype(int)
        wearable = validation_origin == 'wearable'
        wearable_metrics = metrics(validation_y[wearable], prediction[wearable])
        clinical_metrics = metrics(validation_y[~wearable], prediction[~wearable])
        score = (wearable_metrics['rest_recall'] + wearable_metrics['active_recall'] + clinical_metrics['rest_recall']) / 3
        history.append({'epoch': epoch, 'selection_score': score, 'wearable': wearable_metrics, 'clinical': clinical_metrics})
        if score > best_score:
            best_score, stale = score, 0
            best = [value.copy() for value in parameters]
        else:
            stale += 1
        if epoch % 10 == 0:
            print(f'Domain adapter epoch {epoch}: validation group recall {score:.4f}', flush=True)
        if stale >= 30:
            break
    w1, b1, w2, b2 = best
    folded_weights = w1 / scale[:, None]
    folded_bias = b1 - mean @ folded_weights
    return folded_weights.astype(np.float32), folded_bias.astype(np.float32), w2, b2, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rest-dir', required=True, type=Path)
    p.add_argument('--active-dir', required=True, type=Path)
    p.add_argument('--clinical-parquet', required=True, type=Path)
    p.add_argument('--work-dir', required=True, type=Path)
    p.add_argument('--base-model', required=True, type=Path)
    p.add_argument('--examples', required=True, type=Path)
    args = p.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    raw, labels, origins, ids, partitions = [], [], [], [], []
    example_signals = json.loads(args.examples.read_text(encoding='utf-8'))
    excluded_hashes = {digest(s[:WINDOW]) for name, s in example_signals.items() if name != 'Active'}
    sampling_checks = []

    for folder, label in [(args.rest_dir, 0), (args.active_dir, 1)]:
        for file in sorted(folder.rglob('*.csv')):
            rows = list(csv.reader(file.read_text(encoding='utf-8-sig').splitlines(), delimiter=';'))
            timestamps = []
            for row in rows[:512]:
                try:
                    timestamps.append(datetime.fromisoformat(row[0].strip()).timestamp())
                except (ValueError, IndexError):
                    pass
            if len(timestamps) > 2:
                interval = float(np.median(np.diff(timestamps)))
                if interval <= 0:
                    raise ValueError(f'Invalid timestamps in {file.name}')
                sampling_checks.append({'record': file.name, 'estimated_hz': 1 / interval})
            signal = []
            for row in rows:
                try:
                    value = float(row[1])
                except (ValueError, IndexError):
                    continue
                if np.isfinite(value):
                    signal.append(value)
            subject = file.stem.split('_')[0]
            # Keep every window of a wearable subject in the same partition.
            partition = 'test' if subject in {'02', '09'} else 'validation' if subject in {'01', '08'} else 'train'
            for start in range(0, len(signal) - WINDOW + 1, WINDOW):
                window = np.asarray(signal[start:start + WINDOW], dtype=np.float32)
                if window.std() <= 1e-4:
                    continue
                raw.append(window); labels.append(label); origins.append('wearable')
                ids.append(f'wearable:{subject}:{file.name}:{start}')
                partitions.append(partition)
    print(f'Wearable windows: {len(raw)}', flush=True)

    # PhysioNet documents resting ECG acquisition for the complete Shaoxing/Ningbo
    # collection as well: https://physionet.org/content/ecg-arrhythmia/1.0.0/
    candidates = {}
    matched_examples = []
    for row in clinical_rows(args.clinical_parquet):
        record = str(row['record_id'])
        match = re.fullmatch(r'JS(\d+)', record)
        if not match or not 1 <= int(match.group(1)) <= 45551:
            continue
        window = np.asarray(row['signal'][:WINDOW], dtype=np.float32)
        if len(window) != WINDOW or not np.isfinite(window).all() or window.std() <= 1e-8:
            continue
        if digest(window) in excluded_hashes:
            matched_examples.append(record)
            continue
        # Split by record, never by adjacent windows of the same clinical recording.
        score = hashlib.sha256(('stage1-domain-v1:' + record).encode()).hexdigest()
        candidates.setdefault(str(row['label']), []).append((score, record, window))
    for rhythm, rows in sorted(candidates.items()):
        rows = sorted(rows)[:1200]
        for index, (_, record, window) in enumerate(rows):
            partition = 'train' if index < int(0.7 * len(rows)) else 'validation' if index < int(0.85 * len(rows)) else 'test'
            raw.append(window); labels.append(0); origins.append('clinical')
            ids.append(f'clinical:{record}'); partitions.append(partition)
    if not candidates:
        raise ValueError('No documented Chapman-Shaoxing records were identified; do not infer missing activity labels.')
    x = normalize(np.stack(raw))
    y = np.asarray(labels)
    origin = np.asarray(origins)
    split = np.asarray(partitions)
    print(f'Total windows: {len(x)}; clinical examples excluded: {matched_examples}', flush=True)
    for name in ['train', 'validation', 'test']:
        print(name, metrics(y[split == name], np.zeros((split == name).sum(), dtype=int))['confusion'], flush=True)

    model = onnx.load(args.base_model)
    head = model.graph.node[-1]
    if head.op_type != 'Gemm':
        raise ValueError('Expected a final Gemm classification head.')
    initializers = {item.name: item for item in model.graph.initializer}
    original_weights = numpy_helper.to_array(initializers[head.input[1]])
    if original_weights.shape[0] != 2:
        raise ValueError('Expected a two-class Stage 1 model.')
    features_name = head.input[0]
    feature_model = copy.deepcopy(model)
    feature_model.graph.output.append(helper.make_tensor_value_info(features_name, TensorProto.FLOAT, ['batch_size', original_weights.shape[1]]))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(feature_model.SerializeToString(), options, providers=['CPUExecutionProvider'])

    def features(signals):
        chunks = []
        for start in range(0, len(signals), 32):
            chunks.append(session.run([features_name], {'ecg_input': signals[start:start + 32, None, :].astype(np.float32)})[0])
        return np.concatenate(chunks)

    cache = args.work_dir / 'features.npz'
    data_signature = hashlib.sha256(x.tobytes() + args.base_model.read_bytes()).hexdigest()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            valid_cache = str(stored['signature']) == data_signature
            if valid_cache:
                embeddings, inverted = stored['embeddings'], stored['inverted']
    else:
        valid_cache = False
    if not valid_cache:
        print('Extracting frozen ResNet embeddings...', flush=True)
        embeddings = features(x)
        inverted = features(-x[split == 'train'])
        np.savez_compressed(cache, embeddings=embeddings, inverted=inverted, signature=data_signature)
    original_bias = numpy_helper.to_array(initializers[head.input[2]])
    baseline_prediction = (embeddings @ original_weights.T + original_bias).argmax(axis=1)
    training_x = np.concatenate([embeddings[split == 'train'], inverted])
    training_y = np.tile(y[split == 'train'], 2)
    training_origins = np.tile(origin[split == 'train'], 2)
    domain_labels = (training_origins == 'clinical').astype(int)
    weights = np.zeros(len(training_y))
    for mask, mass in [((training_y == 1), 0.25), ((training_y == 0) & (training_origins == 'wearable'), 0.25), ((training_origins == 'clinical'), 0.5)]:
        weights[mask] = mass * len(training_y) / mask.sum()
    base_scores = embeddings @ original_weights.T + original_bias
    base_activity = 1 / (1 + np.exp(-np.clip(base_scores[:, 1] - base_scores[:, 0], -40, 40)))
    vm = split == 'validation'
    w1, b1, w2, b2, history = fit_domain_mlp(training_x, domain_labels, weights,
        embeddings[vm], y[vm], origin[vm], base_activity[vm])
    gate_logits = (np.maximum(0, embeddings @ w1 + b1) @ w2 + b2)[:, 0]
    clinical_probability = 1 / (1 + np.exp(-np.clip(gate_logits, -40, 40)))
    prediction = ((1 - clinical_probability) * base_activity > 0.5).astype(int)
    report = {
        'method': 'Frozen original ResNet activity classifier plus a learned 128-32-1 ReLU acquisition-domain gate. The gate is fitted from ECG embeddings with source labels, balanced across clinical rest, wearable rest and wearable activity. P(activity) = (1 - P(clinical acquisition)) * P(original activity). Clinical resting protocol is documented by PhysioNet. Training-only polarity augmentation; fixed argmax decision. No UI/rhythm label is used as input.',
        'resting_clinical_source': 'https://physionet.org/content/ecg-arrhythmia/1.0.0/',
        'clinical_scope': 'Shaoxing and Ningbo collection; resting ECG acquisition documented by PhysioNet. Unknown identifiers excluded.',
        'example_records_excluded': matched_examples,
        'sampling_checks': sampling_checks,
        'base_model_sha256': hashlib.sha256(args.base_model.read_bytes()).hexdigest(),
        'partitions': {name: {'count': int((split == name).sum()), 'record_ids': [record for record, part in zip(ids, split) if part == name]} for name in ['train', 'validation', 'test']},
        'selection': 'Early stopping by mean validation recall across wearable rest, wearable activity and clinical rest; test labels are not used for fitting or epoch selection.',
        'pretrained_split_limit': 'The original feature extractor training membership cannot be reconstructed from the ONNX file. The evaluation partitions were inspected across development candidates. They measure internal adaptation performance, not an untouched final test or prospective independent clinical validation.',
        'training_history': history,
        'evaluation': {},
    }
    for partition in ['validation', 'test']:
        for source in ['all', 'wearable', 'clinical']:
            mask = (split == partition) & ((origin == source) if source != 'all' else True)
            report['evaluation'][f'{partition}_{source}'] = dict(baseline=metrics(y[mask], baseline_prediction[mask]), adapted=metrics(y[mask], prediction[mask]))
    # Safety check is based on genuine held-out records, never the five demo examples.
    test = report['evaluation']['test_all']
    wearable = report['evaluation']['test_wearable']
    clinical = report['evaluation']['test_clinical']['adapted']
    accepted = (test['adapted']['macro_f1'] > test['baseline']['macro_f1'] + 0.02
                and clinical['rest_recall'] >= 0.95
                and wearable['adapted']['active_recall'] >= 0.80
                and wearable['adapted']['balanced_accuracy'] >= wearable['baseline']['balanced_accuracy'] - 0.03)
    report['accepted'] = bool(accepted)
    (args.work_dir / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['evaluation'], indent=2), flush=True)
    if not accepted:
        raise RuntimeError('Candidate did not meet validation criteria; do not deploy it.')

    prefix = 'domain_adapter/'
    original_output = prefix + 'original_logits'
    head.output[0] = original_output
    additions = {
        'hidden_weights': w1.T.astype(np.float32),
        'hidden_bias': b1.astype(np.float32),
        'weights': w2.T.astype(np.float32),
        'bias': b2.astype(np.float32),
        'one': np.asarray(1, dtype=np.float32),
        'active_index': np.asarray([1], dtype=np.int64),
        'minimum': np.asarray(1e-7, dtype=np.float32),
        'maximum': np.asarray(1, dtype=np.float32),
    }
    for name, array in additions.items():
        model.graph.initializer.append(numpy_helper.from_array(array, prefix + name))
    def node(op, inputs, output, **attributes):
        return helper.make_node(op, inputs, [output], name=output, **attributes)
    model.graph.node.extend([
        node('Gemm', [features_name, prefix + 'hidden_weights', prefix + 'hidden_bias'], prefix + 'hidden_input', transB=1),
        node('Relu', [prefix + 'hidden_input'], prefix + 'hidden'),
        node('Gemm', [prefix + 'hidden', prefix + 'weights', prefix + 'bias'], prefix + 'gate_logits', transB=1),
        node('Sigmoid', [prefix + 'gate_logits'], prefix + 'clinical_probability'),
        node('Softmax', [original_output], prefix + 'original_probability', axis=1),
        node('Gather', [prefix + 'original_probability', prefix + 'active_index'], prefix + 'original_active', axis=1),
        node('Sub', [prefix + 'one', prefix + 'clinical_probability'], prefix + 'wearable_probability'),
        node('Mul', [prefix + 'wearable_probability', prefix + 'original_active'], prefix + 'active_probability'),
        node('Sub', [prefix + 'one', prefix + 'active_probability'], prefix + 'rest_probability'),
        node('Concat', [prefix + 'rest_probability', prefix + 'active_probability'], prefix + 'probabilities', axis=1),
        node('Clip', [prefix + 'probabilities', prefix + 'minimum', prefix + 'maximum'], prefix + 'clipped'),
        node('Log', [prefix + 'clipped'], 'logits'),
    ])
    helper.set_model_props(model, {'stage1_domain_adaptation': report['method'], 'class_order': 'Resting,Active', 'window_samples': '4000', 'resting_clinical_source': SOURCE})
    onnx.checker.check_model(model)
    output = args.work_dir / 'stage1_resnet_domain.onnx'
    onnx.save(model, output)
    adapted_session = ort.InferenceSession(str(output), options, providers=['CPUExecutionProvider'])
    # Confirm serialization matches the fitted two-class scores.
    sample_logits = adapted_session.run(None, {'ecg_input': x[:16, None, :]})[0]
    gate_logits = (np.maximum(0, embeddings[:16] @ w1 + b1) @ w2 + b2)[:, 0]
    clinical_probability = 1 / (1 + np.exp(-np.clip(gate_logits, -40, 40)))
    activity_probability = (1 - clinical_probability) * base_activity[:16]
    expected_logits = np.log(np.clip(np.column_stack([1 - activity_probability, activity_probability]), 1e-7, 1))
    np.testing.assert_allclose(sample_logits, expected_logits, rtol=1e-4, atol=1e-4)
    report['model_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    report['demo_predictions'] = {}
    for name, signal in example_signals.items():
        logits = adapted_session.run(None, {'ecg_input': normalize(np.asarray(signal[:WINDOW], dtype=np.float32))[None, None, :]})[0][0]
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        report['demo_predictions'][name] = {'class': int(probs.argmax()), 'probabilities': probs.tolist()}
    (args.work_dir / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['demo_predictions'], indent=2), flush=True)
    print(f'ACCEPTED: {output}', flush=True)


if __name__ == '__main__':
    main()
