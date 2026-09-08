import test from 'node:test';
import assert from 'node:assert/strict';

import {
  precisionAtK, recallAtK, reciprocalRank, ndcgAtK, hitRate,
  normalise, exactMatch, tokenF1, containsAll, groundedness,
  scoreCase, aggregate,
} from '../src/metrics.mjs';

// <= rather than <, so a tolerance of 0 means "exactly equal" instead of
// "never true", which is what it meant on the first attempt.
const close = (actual, expected, tolerance = 1e-4) =>
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${expected} within ${tolerance}, got ${actual}`,
  );

/* ------------------------------------------------------------- retrieval */

test('precision counts relevant documents among those returned', () => {
  close(precisionAtK(['a', 'b', 'c', 'd'], ['a', 'c']), 0.5);
  close(precisionAtK(['a', 'b', 'c', 'd'], ['a', 'c'], 2), 0.5);
  close(precisionAtK(['a', 'c'], ['a', 'c']), 1);
  close(precisionAtK(['x', 'y'], ['a']), 0);
  close(precisionAtK([], ['a']), 0, 0);
});

test('recall counts relevant documents that were found', () => {
  close(recallAtK(['a', 'b', 'c'], ['a', 'c']), 1);
  close(recallAtK(['a', 'b'], ['a', 'c']), 0.5);
  // Nothing to find means nothing was missed.
  close(recallAtK(['a'], []), 1);
  close(recallAtK(['a', 'b', 'c'], ['a', 'c'], 1), 0.5, 1e-9);
});

test('reciprocal rank rewards putting the answer first', () => {
  close(reciprocalRank(['a', 'b', 'c'], ['a']), 1);
  close(reciprocalRank(['b', 'a', 'c'], ['a']), 0.5);
  close(reciprocalRank(['b', 'c', 'a'], ['a']), 1 / 3);
  close(reciprocalRank(['x', 'y'], ['a']), 0);
});

test('ndcg notices a relevant document sliding down the list', () => {
  // Two relevant, at positions 1 and 3.
  // DCG  = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
  // IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
  close(ndcgAtK(['a', 'b', 'c'], ['a', 'c'], 3), 1.5 / 1.63093);

  // Perfect ordering scores 1.
  close(ndcgAtK(['a', 'c', 'b'], ['a', 'c'], 3), 1);

  // Precision and recall are identical for both of these. ndcg is not, which
  // is the whole reason it is here.
  const early = ndcgAtK(['a', 'x', 'y', 'z'], ['a'], 4);
  const late = ndcgAtK(['x', 'y', 'z', 'a'], ['a'], 4);
  assert.ok(early > late, 'a document found first should score higher than one found last');
  close(precisionAtK(['a', 'x', 'y', 'z'], ['a']), precisionAtK(['x', 'y', 'z', 'a'], ['a']));
});

test('hit rate is the floor below which nothing else matters', () => {
  assert.equal(hitRate(['a', 'b'], ['b']), 1);
  assert.equal(hitRate(['a', 'b'], ['z']), 0);
  assert.equal(hitRate(['a', 'b'], ['b'], 1), 0, 'not within the cutoff');
  assert.equal(hitRate([], []), 1);
});

/* --------------------------------------------------------------- answers */

test('normalisation strips articles, punctuation and case', () => {
  assert.equal(normalise('The Answer.'), 'answer');
  assert.equal(normalise('  A  cat,  and   a dog!  '), 'cat and dog');
  assert.equal(normalise(null), '');
});

test('exact match compares normalised text, not raw strings', () => {
  // Without normalisation these score zero against each other and every number
  // the tool produces afterwards is noise.
  assert.equal(exactMatch('The answer.', 'answer'), 1);
  assert.equal(exactMatch('42', '42'), 1);
  assert.equal(exactMatch('43', '42'), 0);
});

test('token f1 gives partial credit', () => {
  // predicted "cat sat" against expected "cat sat on mat"
  // precision 2/2, recall 2/4, f1 = 2 * 1 * 0.5 / 1.5
  close(tokenF1('the cat sat', 'cat sat on the mat'), 2 / 3);
  close(tokenF1('same words here', 'same words here'), 1);
  close(tokenF1('completely different', 'nothing alike'), 0);
  close(tokenF1('', ''), 1);
  close(tokenF1('something', ''), 0);
});

test('token f1 counts multiplicity, so repetition cannot inflate it', () => {
  // "cat cat cat" must not score as three matches against a single "cat".
  const repeated = tokenF1('cat cat cat', 'cat dog');
  const single = tokenF1('cat', 'cat dog');

  assert.ok(repeated < single, 'padding the answer should not improve the score');
});

test('contains all checks required phrases in normalised form', () => {
  close(containsAll('The total is 1500 CFA, due Friday', ['1500', 'Friday']), 1);
  close(containsAll('The total is 1500 CFA', ['1500', 'Friday']), 0.5);
  close(containsAll('anything', []), 1, 0);
});

test('groundedness catches an answer invented wholesale', () => {
  const context = ['The Benin gateway settles overnight and refunds take one day'];

  const grounded = groundedness('The gateway settles overnight', context);
  const invented = groundedness('Elephants migrate through Patagonia annually', context);

  assert.ok(grounded > 0.9, 'an answer drawn from the context scores high');
  assert.ok(invented < 0.2, 'an answer with no support scores low');
});

/* ------------------------------------------------------------- scoring */

test('a case with no expected answer reports retrieval only', () => {
  const scores = scoreCase(
    { question: 'q', relevant_docs: ['a'] },
    { answer: 'anything', retrieved: ['a', 'b'] },
  );

  // Scoring an absent expectation as zero would look like failure rather than
  // like a partially labelled dataset.
  assert.ok('recall@k' in scores);
  assert.ok(!('exact_match' in scores));
  assert.ok(!('token_f1' in scores));
});

test('a case with no retrieval reports answer quality only', () => {
  const scores = scoreCase(
    { question: 'q', expected_answer: 'yes' },
    { answer: 'yes' },
  );

  assert.equal(scores.exact_match, 1);
  assert.ok(!('recall@k' in scores));
});

test('groundedness only appears when contexts were supplied', () => {
  const without = scoreCase({ question: 'q', expected_answer: 'a' }, { answer: 'a' });
  const with_ = scoreCase({ question: 'q', expected_answer: 'a' }, { answer: 'a', contexts: ['a b c'] });

  assert.ok(!('groundedness' in without));
  assert.ok('groundedness' in with_);
});

test('aggregation skips missing values instead of counting them as zero', () => {
  const aggregated = aggregate([
    { 'recall@k': 1, exact_match: 1 },
    { 'recall@k': 0 },              // no expected answer on this case
    { 'recall@k': 1, exact_match: 0 },
  ]);

  close(aggregated['recall@k'], 2 / 3);
  // Averaged over the two cases that had an expected answer, not over three.
  close(aggregated.exact_match, 0.5);
});

test('aggregation of nothing is empty rather than NaN', () => {
  assert.deepEqual(aggregate([]), {});
  assert.deepEqual(aggregate([{}]), {});
});

test('a perfect run scores one across the board', () => {
  const scores = scoreCase(
    {
      question: 'What does the Benin gateway do?',
      relevant_docs: ['doc-1'],
      expected_answer: 'It settles overnight',
      must_contain: ['overnight'],
    },
    {
      answer: 'It settles overnight',
      retrieved: ['doc-1'],
      contexts: ['It settles overnight'],
    },
  );

  for (const [metric, value] of Object.entries(scores)) {
    close(value, 1, 1e-6);
  }
});
