import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { loadDataset, loadPredictions, join as joinCases } from '../src/dataset.mjs';

async function fixture(t, name, lines) {
  const dir = await mkdtemp(join(tmpdir(), 'ragbench-'));
  t.after(() => rm(dir, { recursive: true, force: true }));

  const file = join(dir, name);
  await writeFile(file, Array.isArray(lines) ? lines.map((l) => (typeof l === 'string' ? l : JSON.stringify(l))).join('\n') : lines);

  return file;
}

test('a well formed dataset loads', async (t) => {
  const file = await fixture(t, 'cases.jsonl', [
    { id: 'a', question: 'q1', expected_answer: 'yes' },
    { id: 'b', question: 'q2', relevant_docs: ['doc-1'] },
  ]);

  const { cases, problems } = await loadDataset(file);

  assert.deepEqual(problems, []);
  assert.equal(cases.length, 2);
});

test('blank lines and comments are skipped', async (t) => {
  const file = await fixture(t, 'cases.jsonl', [
    '// the payments suite',
    '',
    JSON.stringify({ id: 'a', question: 'q', expected_answer: 'y' }),
    '',
  ]);

  const { cases, problems } = await loadDataset(file);

  assert.deepEqual(problems, []);
  assert.equal(cases.length, 1);
});

test('a case with nothing to check against is refused', async (t) => {
  // It passes every metric vacuously and quietly lifts the average.
  const file = await fixture(t, 'cases.jsonl', [{ id: 'a', question: 'just a question' }]);

  const { cases, problems } = await loadDataset(file);

  assert.equal(cases.length, 0);
  assert.match(problems[0], /would inflate the average/);
});

test('a duplicate id is refused rather than silently overwriting', async (t) => {
  const file = await fixture(t, 'cases.jsonl', [
    { id: 'a', question: 'q1', expected_answer: 'x' },
    { id: 'a', question: 'q2', expected_answer: 'y' },
  ]);

  const { cases, problems } = await loadDataset(file);

  // Predictions join by id, so a duplicate makes half the suite vanish.
  assert.equal(cases.length, 1);
  assert.match(problems[0], /reuses id "a"/);
});

test('a malformed line names its line number', async (t) => {
  const file = await fixture(t, 'cases.jsonl', [
    JSON.stringify({ id: 'a', question: 'q', expected_answer: 'y' }),
    '{ truncated',
  ]);

  const { cases, problems } = await loadDataset(file);

  assert.equal(cases.length, 1);
  assert.match(problems[0], /:2 is not valid JSON/);
});

test('an unknown field is reported, because it is usually a typo', async (t) => {
  const file = await fixture(t, 'cases.jsonl', [
    { id: 'a', question: 'q', expected_ansewr: 'typo' },
  ]);

  const { problems } = await loadDataset(file);

  assert.ok(problems.some((p) => /unknown field "expected_ansewr"/.test(p)));
});

test('a prediction with no id cannot be matched and is reported', async (t) => {
  const file = await fixture(t, 'preds.jsonl', [{ answer: 'orphan' }]);

  const { predictions, problems } = await loadPredictions(file);

  assert.equal(predictions.size, 0);
  assert.match(problems[0], /has no id/);
});

test('a case with no prediction is scored as empty, not skipped', () => {
  const cases = [
    { id: 'a', question: 'q', expected_answer: 'yes' },
    { id: 'b', question: 'q', expected_answer: 'no' },
  ];
  const predictions = new Map([['a', { id: 'a', answer: 'yes' }]]);

  const { paired, missing } = joinCases(cases, predictions);

  // Skipping it would mean a system that answered nothing scores the same as
  // one that answered perfectly on what it attempted.
  assert.equal(paired.length, 2);
  assert.deepEqual(missing, ['b']);
  assert.equal(paired[1].prediction.answer, '');
});

test('predictions matching no case are reported rather than ignored silently', () => {
  const cases = [{ id: 'a', question: 'q', expected_answer: 'y' }];
  const predictions = new Map([
    ['a', { id: 'a', answer: 'y' }],
    ['ghost', { id: 'ghost', answer: 'y' }],
  ]);

  const { extra } = joinCases(cases, predictions);

  assert.deepEqual(extra, ['ghost']);
});

test('ids are matched as strings, so numeric ids still join', () => {
  const cases = [{ id: 1, question: 'q', expected_answer: 'y' }];
  const predictions = new Map([['1', { id: 1, answer: 'y' }]]);

  const { missing } = joinCases(cases, predictions);

  assert.deepEqual(missing, []);
});
