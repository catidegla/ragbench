/**
 * Loading and checking the golden dataset.
 *
 * JSON lines rather than one array, because a dataset grows by appending and
 * because a diff of one changed case should be one changed line. It also means
 * a malformed case is one bad line rather than an unparseable file.
 *
 * Validation is strict and names the line number. A silently skipped case makes
 * a suite look like it passed when it did not run.
 */

import { readFile } from 'node:fs/promises';

const KNOWN_CASE_KEYS = new Set([
  'id', 'question', 'expected_answer', 'relevant_docs', 'must_contain', 'tags', 'notes',
]);

const KNOWN_PREDICTION_KEYS = new Set(['id', 'answer', 'retrieved', 'contexts', 'latency_ms', 'cost']);

function parseLines(raw, file) {
  const out = [];
  const problems = [];

  raw.split('\n').forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('//')) return;

    try {
      out.push({ record: JSON.parse(trimmed), line: index + 1 });
    } catch (error) {
      problems.push(`${file}:${index + 1} is not valid JSON: ${error.message}`);
    }
  });

  return { out, problems };
}

/**
 * @returns {Promise<{cases: Array, problems: string[]}>}
 */
export async function loadDataset(file) {
  const { out, problems } = parseLines(await readFile(file, 'utf8'), file);
  const cases = [];
  const seen = new Set();

  for (const { record, line } of out) {
    const where = `${file}:${line}`;

    if (typeof record.question !== 'string' || !record.question.trim()) {
      problems.push(`${where} has no question`);
      continue;
    }

    const id = record.id ?? `case-${line}`;

    if (seen.has(id)) {
      // Duplicate ids silently overwrite each other when predictions are joined
      // by id, so half the suite would vanish without a word.
      problems.push(`${where} reuses id "${id}"`);
      continue;
    }
    seen.add(id);

    if (record.relevant_docs !== undefined && !Array.isArray(record.relevant_docs)) {
      problems.push(`${where} relevant_docs must be an array of document ids`);
      continue;
    }

    if (record.must_contain !== undefined && !Array.isArray(record.must_contain)) {
      problems.push(`${where} must_contain must be an array of strings`);
      continue;
    }

    // Reported before the expectation check, so a misspelled field name is
    // named rather than hidden behind the "no expectation" error it causes.
    for (const key of Object.keys(record)) {
      if (!KNOWN_CASE_KEYS.has(key)) problems.push(`${where} has unknown field "${key}"`);
    }

    // A case with nothing to check against passes every metric vacuously,
    // which quietly inflates the average.
    const hasExpectation =
      record.expected_answer !== undefined ||
      (Array.isArray(record.relevant_docs) && record.relevant_docs.length > 0) ||
      (Array.isArray(record.must_contain) && record.must_contain.length > 0);

    if (!hasExpectation) {
      problems.push(
        `${where} has no expected_answer, relevant_docs or must_contain, so it cannot fail and would inflate the average`,
      );
      continue;
    }

    cases.push({ ...record, id });
  }

  return { cases, problems };
}

/**
 * @returns {Promise<{predictions: Map<string, object>, problems: string[]}>}
 */
export async function loadPredictions(file) {
  const { out, problems } = parseLines(await readFile(file, 'utf8'), file);
  const predictions = new Map();

  for (const { record, line } of out) {
    const where = `${file}:${line}`;

    if (record.id === undefined) {
      problems.push(`${where} has no id, so it cannot be matched to a case`);
      continue;
    }

    if (record.retrieved !== undefined && !Array.isArray(record.retrieved)) {
      problems.push(`${where} retrieved must be an array of document ids`);
      continue;
    }

    for (const key of Object.keys(record)) {
      if (!KNOWN_PREDICTION_KEYS.has(key)) problems.push(`${where} has unknown field "${key}"`);
    }

    predictions.set(String(record.id), record);
  }

  return { predictions, problems };
}

/**
 * Join cases to predictions.
 *
 * A case with no prediction is reported rather than dropped. Dropping it means
 * a system that answered nothing scores the same as one that answered
 * perfectly on the cases it did attempt, which is exactly backwards.
 */
export function join(cases, predictions) {
  const paired = [];
  const missing = [];

  for (const testCase of cases) {
    const prediction = predictions.get(String(testCase.id));

    if (!prediction) {
      missing.push(testCase.id);
      // Counted as an empty answer, so failing to respond costs you the score
      // rather than excusing you from it.
      paired.push({ testCase, prediction: { id: testCase.id, answer: '', retrieved: [] } });
      continue;
    }

    paired.push({ testCase, prediction });
  }

  const extra = [...predictions.keys()].filter((id) => !cases.some((c) => String(c.id) === id));

  return { paired, missing, extra };
}
