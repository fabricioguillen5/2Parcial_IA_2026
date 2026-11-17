// Preserve every ONNX field and weight; change only the declared input length.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

function varint(buffer, offset) {
  let value = 0;
  let shift = 0;
  const start = offset;
  while (offset < buffer.length) {
    const byte = buffer[offset++];
    value += (byte & 127) * 2 ** shift;
    if (!(byte & 128)) return { value, end: offset, start };
    shift += 7;
    if (shift > 49) throw new Error('Unsupported protobuf varint.');
  }
  throw new Error('Truncated protobuf varint.');
}

function fields(buffer, start = 0, end = buffer.length) {
  const result = [];
  for (let offset = start; offset < end;) {
    const key = varint(buffer, offset);
    offset = key.end;
    const field = { number: Math.floor(key.value / 8), wire: key.value & 7, start: offset };
    if (field.wire === 0) {
      const data = varint(buffer, offset);
      field.value = data.value;
      offset = data.end;
    } else if (field.wire === 2) {
      const size = varint(buffer, offset);
      field.start = size.end;
      offset = size.end + size.value;
    } else if (field.wire === 1) offset += 8;
    else if (field.wire === 5) offset += 4;
    else throw new Error(`Unsupported protobuf wire type ${field.wire}.`);
    assert(offset <= end, 'Invalid protobuf field length.');
    field.end = offset;
    result.push(field);
  }
  return result;
}

function child(buffer, field, number) {
  const found = fields(buffer, field.start, field.end).find(f => f.number === number);
  assert(found, `Missing ONNX field ${number}.`);
  return found;
}

const project = path.resolve(__dirname, '..');
const original = fs.readFileSync(path.join(project, 'public', 'stage2_resnet (1).onnx'));
const graph = fields(original).find(f => f.number === 7);
assert(graph, 'Missing ONNX graph.');
const graphFields = fields(original, graph.start, graph.end);
const nodes = graphFields.filter(f => f.number === 1).map(node => {
  const op = child(original, node, 4);
  return original.toString('utf8', op.start, op.end);
});
assert.deepEqual(nodes.slice(-3), ['GlobalAveragePool', 'Flatten', 'Gemm']);
assert(nodes.every(op => ['Conv', 'Relu', 'MaxPool', 'Add', 'GlobalAveragePool', 'Flatten', 'Gemm'].includes(op)), 'Review length-dependent operators before adaptation.');
const inputs = graphFields.filter(f => f.number === 11);
assert.equal(inputs.length, 1);
const name = child(original, inputs[0], 1);
assert.equal(original.toString('utf8', name.start, name.end), 'ecg_input');
const tensorType = child(original, child(original, inputs[0], 2), 1);
const shape = child(original, tensorType, 2);
const dimensions = fields(original, shape.start, shape.end).filter(f => f.number === 1);
assert.equal(dimensions.length, 3);
assert.equal(child(original, dimensions[1], 1).value, 1);
const length = child(original, dimensions[2], 1);
assert.equal(length.value, 5000);
assert.equal(length.end - length.start, 2);
// Both lengths have a two-byte varint, so all protobuf lengths stay unchanged.
const adapted = Buffer.from(original);
adapted[length.start] = (4000 & 127) | 128;
adapted[length.start + 1] = 4000 >>> 7;
assert.equal(varint(adapted, length.start).value, 4000);
const differences = original.reduce((count, byte, index) => count + (byte !== adapted[index]), 0);
assert.equal(differences, 2, 'Only the two input-length bytes may change.');
fs.writeFileSync(path.join(project, 'public', 'stage2_resnet_4000.onnx'), adapted);
console.log('Created stage2_resnet_4000.onnx: input [batch,1,4000]; all weights and operators preserved.');
