/**
 * Metrics that need no model.
 *
 * Every commercial eval platform leans on an LLM judge, which costs money per
 * run and returns a slightly different number each time. Neither is acceptable
 * as a build gate: a check that costs a dollar to run gets run less, and a
 * check that drifts cannot distinguish a regression from noise.
 *
 * Everything here is deterministic and free. Given the same dataset and the
 * same predictions it returns the same numbers on any machine, which is what
 * makes "fail the build if recall dropped" a sentence that means something.
 *
 * An LLM judge is still useful for answer quality, and there is a hook for one.
 * It is not on the path that gates your build.
 */

/* ------------------------------------------------------------- retrieval */

/**
 * Of the documents retrieved, how many were relevant.
 *
 * Answers "how much noise is in the context window", which is the metric that
 * predicts hallucination.
 */
export function precisionAtK(retrieved, relevant, k = retrieved.length) {
  const top = retrieved.slice(0, k);
  if (top.length === 0) return 0;

  const relevantSet = new Set(relevant);
  const hits = top.filter((id) => relevantSet.has(id)).length;

  return hits / top.length;
}

/**
 * Of the relevant documents, how many were retrieved.
 *
 * Answers "can the model possibly have got this right", which is the metric
 * that predicts an unanswerable question. A generator cannot cite what
 * retrieval never fetched.
 */
export function recallAtK(retrieved, relevant, k = retrieved.length) {
  if (relevant.length === 0) return 1;

  const top = new Set(retrieved.slice(0, k));
  const found = relevant.filter((id) => top.has(id)).length;

  return found / relevant.length;
}

/** Reciprocal rank of the first relevant document. Rewards ranking, not just presence. */
export function reciprocalRank(retrieved, relevant) {
  const relevantSet = new Set(relevant);

  for (let i = 0; i < retrieved.length; i++) {
    if (relevantSet.has(retrieved[i])) return 1 / (i + 1);
  }

  return 0;
}

/**
 * Normalised discounted cumulative gain, binary relevance.
 *
 * The one that notices when a relevant document slips from position one to
 * position eight, which precision and recall both consider identical.
 */
export function ndcgAtK(retrieved, relevant, k = 10) {
  if (relevant.length === 0) return 1;

  const relevantSet = new Set(relevant);
  const top = retrieved.slice(0, k);

  let dcg = 0;
  top.forEach((id, i) => {
    if (relevantSet.has(id)) dcg += 1 / Math.log2(i + 2);
  });

  // The best achievable ordering: every relevant document first.
  let idcg = 0;
  const ideal = Math.min(relevant.length, k);
  for (let i = 0; i < ideal; i++) idcg += 1 / Math.log2(i + 2);

  return idcg === 0 ? 0 : dcg / idcg;
}

/** Did retrieval find anything relevant at all. The floor below which nothing else matters. */
export function hitRate(retrieved, relevant, k = retrieved.length) {
  if (relevant.length === 0) return 1;

  const relevantSet = new Set(relevant);
  return retrieved.slice(0, k).some((id) => relevantSet.has(id)) ? 1 : 0;
}

/* ---------------------------------------------------------------- answers */

/**
 * SQuAD style normalisation: lowercase, drop articles and punctuation, collapse
 * whitespace. Without it "The answer." and "answer" score zero against each
 * other, and every number the tool produces is noise.
 */
export function normalise(text) {
  return String(text ?? '')
    .toLowerCase()
    .replace(/\b(a|an|the)\b/g, ' ')
    .replace(/[\p{P}\p{S}]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function exactMatch(predicted, expected) {
  return normalise(predicted) === normalise(expected) ? 1 : 0;
}

/**
 * Token level F1. Partial credit, which exact match cannot give.
 *
 * Counts multiplicity, so repeating a word does not inflate the overlap.
 */
export function tokenF1(predicted, expected) {
  const predictedTokens = normalise(predicted).split(' ').filter(Boolean);
  const expectedTokens = normalise(expected).split(' ').filter(Boolean);

  if (predictedTokens.length === 0 && expectedTokens.length === 0) return 1;
  if (predictedTokens.length === 0 || expectedTokens.length === 0) return 0;

  const counts = new Map();
  for (const token of expectedTokens) counts.set(token, (counts.get(token) ?? 0) + 1);

  let overlap = 0;
  for (const token of predictedTokens) {
    const remaining = counts.get(token) ?? 0;
    if (remaining > 0) {
      overlap += 1;
      counts.set(token, remaining - 1);
    }
  }

  if (overlap === 0) return 0;

  const precision = overlap / predictedTokens.length;
  const recall = overlap / expectedTokens.length;

  return (2 * precision * recall) / (precision + recall);
}

/**
 * Did the answer contain every required phrase.
 *
 * The metric to reach for when there is a fact that must appear, such as an
 * amount or a date, and everything around it is free text.
 */
export function containsAll(predicted, phrases = []) {
  if (phrases.length === 0) return 1;

  const haystack = normalise(predicted);
  const found = phrases.filter((phrase) => haystack.includes(normalise(phrase))).length;

  return found / phrases.length;
}

/**
 * Whether the answer stayed inside the retrieved context.
 *
 * A cheap groundedness proxy: the share of content words in the answer that
 * also appear in the context. It does not understand meaning, so it will not
 * catch a fluent misreading, but it does catch an answer invented wholesale,
 * and it costs nothing to run on every case.
 */
export function groundedness(predicted, contexts = []) {
  const answerTokens = normalise(predicted).split(' ').filter((t) => t.length > 2);
  if (answerTokens.length === 0) return 1;

  const contextTokens = new Set(normalise(contexts.join(' ')).split(' '));
  const supported = answerTokens.filter((t) => contextTokens.has(t)).length;

  return supported / answerTokens.length;
}

/* ------------------------------------------------------------ aggregation */

export const RETRIEVAL_METRICS = ['precision@k', 'recall@k', 'mrr', 'ndcg@k', 'hit_rate'];
export const ANSWER_METRICS = ['exact_match', 'token_f1', 'contains_all', 'groundedness'];

/**
 * Score one case. Metrics whose inputs are absent are omitted rather than
 * scored as zero, because a dataset without expected answers should report
 * retrieval quality, not a column of zeroes that looks like failure.
 *
 * @param {object} testCase   { question, relevant_docs, expected_answer, must_contain }
 * @param {object} prediction { answer, retrieved, contexts }
 * @param {object} options    { k }
 */
export function scoreCase(testCase, prediction, { k = 10 } = {}) {
  const scores = {};

  const retrieved = prediction.retrieved ?? [];
  const relevant = testCase.relevant_docs ?? [];

  if (relevant.length > 0 || retrieved.length > 0) {
    scores['precision@k'] = precisionAtK(retrieved, relevant, k);
    scores['recall@k'] = recallAtK(retrieved, relevant, k);
    scores.mrr = reciprocalRank(retrieved, relevant);
    scores['ndcg@k'] = ndcgAtK(retrieved, relevant, k);
    scores.hit_rate = hitRate(retrieved, relevant, k);
  }

  if (testCase.expected_answer !== undefined && testCase.expected_answer !== null) {
    scores.exact_match = exactMatch(prediction.answer, testCase.expected_answer);
    scores.token_f1 = tokenF1(prediction.answer, testCase.expected_answer);
  }

  if (Array.isArray(testCase.must_contain) && testCase.must_contain.length) {
    scores.contains_all = containsAll(prediction.answer, testCase.must_contain);
  }

  if (Array.isArray(prediction.contexts) && prediction.contexts.length) {
    scores.groundedness = groundedness(prediction.answer, prediction.contexts);
  }

  return scores;
}

/**
 * Mean of each metric across cases.
 *
 * Missing values are skipped rather than counted as zero. Averaging a metric
 * over cases that could not produce it silently penalises a dataset for being
 * partially labelled.
 */
export function aggregate(caseScores) {
  const sums = new Map();
  const counts = new Map();

  for (const scores of caseScores) {
    for (const [metric, value] of Object.entries(scores)) {
      if (typeof value !== 'number' || Number.isNaN(value)) continue;
      sums.set(metric, (sums.get(metric) ?? 0) + value);
      counts.set(metric, (counts.get(metric) ?? 0) + 1);
    }
  }

  const out = {};
  for (const [metric, sum] of sums) {
    out[metric] = Number((sum / counts.get(metric)).toFixed(6));
  }

  return out;
}
