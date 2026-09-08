import test from 'node:test';
import assert from 'node:assert/strict';

import { evaluate, summarise, toMarkdown, DEFAULT_TOLERANCE } from '../src/gate.mjs';

test('a clean run against a matching baseline passes', () => {
  const verdict = evaluate({ 'recall@k': 0.9 }, { 'recall@k': 0.9 });

  assert.equal(verdict.passed, true);
  assert.deepEqual(verdict.failures, []);
  assert.equal(verdict.comparedToBaseline, true);
});

test('a drop beyond tolerance fails the build', () => {
  const verdict = evaluate({ 'recall@k': 0.7 }, { 'recall@k': 0.9 });

  assert.equal(verdict.passed, false);
  assert.equal(verdict.failures[0].kind, 'regression');
  assert.match(verdict.failures[0].message, /fell 0\.2000/);
});

test('a drop inside tolerance warns rather than fails', () => {
  // Retrieval scores move slightly for reasons that are not your change, and a
  // gate that fires on noise gets disabled within a week.
  const verdict = evaluate({ 'recall@k': 0.895 }, { 'recall@k': 0.9 }, { tolerance: 0.01 });

  assert.equal(verdict.passed, true);
  assert.equal(verdict.warnings.length, 1);
  assert.match(verdict.warnings[0].message, /within the 0.01 tolerance/);
});

test('an improvement is never a failure', () => {
  const verdict = evaluate({ 'recall@k': 0.99 }, { 'recall@k': 0.5 });

  assert.equal(verdict.passed, true);
  assert.equal(verdict.warnings.length, 0);
  assert.ok(verdict.comparisons[0].delta > 0);
});

test('thresholds are checked with or without a baseline', () => {
  const noBaseline = evaluate({ 'recall@k': 0.5 }, null, { thresholds: { 'recall@k': 0.8 } });

  assert.equal(noBaseline.passed, false);
  assert.equal(noBaseline.failures[0].kind, 'threshold');
  assert.equal(noBaseline.comparedToBaseline, false);
});

test('a threshold on a metric the run never produced is a failure, not a pass', () => {
  // Treating it as passing hides that the check never ran, which is the worst
  // possible outcome for a gate.
  const verdict = evaluate({ 'recall@k': 0.9 }, null, { thresholds: { faithfulness: 0.8 } });

  assert.equal(verdict.passed, false);
  assert.equal(verdict.failures[0].kind, 'missing');
  assert.match(verdict.failures[0].message, /was not produced by this run/);
});

test('a first run with no baseline says so rather than implying a comparison', () => {
  const verdict = evaluate({ 'recall@k': 0.9 }, null);

  assert.equal(verdict.passed, true);
  assert.equal(verdict.comparedToBaseline, false);
  assert.deepEqual(verdict.comparisons, []);
  assert.match(summarise(verdict, { 'recall@k': 0.9 }), /no baseline/);
});

test('only shared metrics are compared', () => {
  // A metric that appeared for the first time this run has nothing to regress
  // against, and inventing a baseline of zero would report a false improvement.
  const verdict = evaluate(
    { 'recall@k': 0.9, groundedness: 0.7 },
    { 'recall@k': 0.9 },
  );

  assert.equal(verdict.comparisons.length, 1);
  assert.equal(verdict.comparisons[0].metric, 'recall@k');
});

test('gateOn narrows which metrics can fail the build', () => {
  const verdict = evaluate(
    { 'recall@k': 0.9, exact_match: 0.1 },
    { 'recall@k': 0.9, exact_match: 0.9 },
    { gateOn: ['recall@k'] },
  );

  // exact_match collapsed, but it was not being gated on.
  assert.equal(verdict.passed, true);
});

test('several failures are all reported, not just the first', () => {
  const verdict = evaluate(
    { 'recall@k': 0.5, 'ndcg@k': 0.4 },
    { 'recall@k': 0.9, 'ndcg@k': 0.9 },
  );

  assert.equal(verdict.failures.length, 2);
});

test('the default tolerance is small enough to catch a real regression', () => {
  // A five point drop must fail under the default, or the gate is decorative.
  const verdict = evaluate({ 'recall@k': 0.85 }, { 'recall@k': 0.9 });

  assert.ok(DEFAULT_TOLERANCE < 0.05);
  assert.equal(verdict.passed, false);
});

test('markdown renders a comparison table when there is a baseline', () => {
  const verdict = evaluate({ 'recall@k': 0.7 }, { 'recall@k': 0.9 });
  const md = toMarkdown(verdict, { 'recall@k': 0.7 });

  assert.match(md, /\*\*Failed\.\*\*/);
  assert.match(md, /\| Metric \| Baseline \| Now \| Change \|/);
  assert.match(md, /0\.9000/);
  assert.match(md, /fell 0\.2000/);
});

test('markdown says plainly when nothing was compared', () => {
  const verdict = evaluate({ 'recall@k': 0.9 }, null);
  const md = toMarkdown(verdict, { 'recall@k': 0.9 });

  assert.match(md, /No baseline recorded yet/);
});
