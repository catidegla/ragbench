import { test } from 'node:test';
import assert from 'node:assert/strict';

import { auditLabels, compareCorpus, labelWarnings, DEFAULT_TURNOVER } from '../src/labels.mjs';
import * as store from '../src/store.mjs';

const pair = (id, relevant_docs, retrieved) => ({
  testCase: { id, question: `q ${id}`, relevant_docs },
  prediction: { id, answer: 'a', retrieved },
});

/* ------------------------------------------------------------- the audit */

test('a labelled document nothing retrieves is an orphan', () => {
  const audit = auditLabels([
    pair('a', ['doc/1'], ['doc/1', 'doc/9']),
    pair('b', ['doc/2'], ['doc/9']),
  ]);

  assert.equal(audit.labelledCases, 2);
  assert.equal(audit.labelledDocs, 2);
  assert.deepEqual(audit.orphans, [{ docId: 'doc/2', cases: ['b'] }]);
  assert.equal(audit.orphanRate, 0.5);
});

test('a document retrieved for some other question is not an orphan', () => {
  // The signal is reachability across the run, not per case. A label the
  // ranker missed for its own question is a recall failure, which the recall
  // metric already reports, and reporting it twice would make every genuine
  // miss look like a dataset problem.
  const audit = auditLabels([
    pair('a', ['doc/1'], ['doc/9']),
    pair('b', ['doc/9'], ['doc/1']),
  ]);

  assert.deepEqual(audit.orphans, []);
});

test('cases without labels are skipped rather than counted as perfect', () => {
  const audit = auditLabels([
    pair('a', ['doc/1'], ['doc/1']),
    { testCase: { id: 'b', question: 'q' }, prediction: { id: 'b', answer: 'a', retrieved: ['doc/5'] } },
  ]);

  assert.equal(audit.labelledCases, 1);
  assert.equal(audit.labelledDocs, 1);
  assert.deepEqual(audit.orphans, []);
});

test('one orphan blamed on every case that needs it', () => {
  const audit = auditLabels([
    pair('a', ['gone.md'], ['x']),
    pair('b', ['gone.md'], ['x']),
    pair('c', ['x'], ['x']),
  ]);

  assert.deepEqual(audit.orphans, [{ docId: 'gone.md', cases: ['a', 'b'] }]);
});

test('the corpus is the sorted set of everything retrieval returned', () => {
  const audit = auditLabels([
    pair('a', ['b'], ['c', 'a']),
    pair('b', ['b'], ['b', 'a']),
  ]);

  assert.deepEqual(audit.corpus, ['a', 'b', 'c']);
});

test('an empty run audits to nothing rather than dividing by zero', () => {
  const audit = auditLabels([]);

  assert.equal(audit.orphanRate, 0);
  assert.deepEqual(audit.orphans, []);
  assert.deepEqual(audit.corpus, []);
});

test('document ids are compared as strings, so 12 and "12" are one document', () => {
  const audit = auditLabels([pair('a', [12], ['12'])]);

  assert.deepEqual(audit.orphans, []);
});

/* -------------------------------------------------------------- the drift */

test('no previous run reports null rather than stability it cannot see', () => {
  assert.equal(compareCorpus(['a', 'b'], []), null);
});

test('turnover counts both directions against the union', () => {
  const drift = compareCorpus(['a', 'b', 'c'], ['a', 'b', 'd']);

  assert.deepEqual(drift.added, ['c']);
  assert.deepEqual(drift.removed, ['d']);
  assert.equal(drift.turnover, 2 / 4);
});

test('an unchanged corpus is zero turnover', () => {
  assert.equal(compareCorpus(['a', 'b'], ['b', 'a']).turnover, 0);
});

test('a corpus replaced wholesale is complete turnover', () => {
  assert.equal(compareCorpus(['x'], ['y']).turnover, 1);
});

/* ----------------------------------------------------------- the warnings */

test('a single orphan is reported, because it caps a case below 1', () => {
  const warnings = labelWarnings(auditLabels([
    pair('a', ['gone.md'], ['x']),
    pair('b', ['x'], ['x']),
  ]));

  assert.equal(warnings.length, 1);
  assert.equal(warnings[0].kind, 'orphaned-labels');
  assert.match(warnings[0].message, /recall cannot reach 1/);
  assert.match(warnings[0].message, /1 case\(s\)/);
});

test('clean labels and a still corpus produce no warnings at all', () => {
  const audit = auditLabels([pair('a', ['x'], ['x'])]);

  assert.deepEqual(labelWarnings(audit, compareCorpus(['x'], ['x'])), []);
});

test('small churn is furniture and stays quiet', () => {
  // Twenty documents, one swapped. A corpus always moves a little, and a
  // warning that fires every run stops being read.
  const stable = Array.from({ length: 20 }, (_, i) => `doc/${i}`);
  const audit = auditLabels([pair('a', ['doc/0'], stable)]);
  const drift = compareCorpus([...stable.slice(1), 'doc/new'], stable);

  assert.ok(drift.turnover < DEFAULT_TURNOVER, `turnover was ${drift.turnover}`);
  assert.deepEqual(labelWarnings(audit, drift), []);
});

test('churn past the threshold says to re-audit before trusting the comparison', () => {
  const audit = auditLabels([pair('a', ['x'], ['x'])]);
  const warnings = labelWarnings(audit, compareCorpus(['x', 'new'], ['x', 'old']));

  assert.equal(warnings.length, 1);
  assert.equal(warnings[0].kind, 'corpus-drift');
  assert.match(warnings[0].message, /re-auditing/);
});

test('the threshold is the caller\'s to move', () => {
  const stable = Array.from({ length: 20 }, (_, i) => `doc/${i}`);
  const audit = auditLabels([pair('a', ['doc/0'], stable)]);
  const drift = compareCorpus([...stable.slice(1), 'doc/new'], stable);

  assert.deepEqual(labelWarnings(audit, drift), []);
  assert.equal(labelWarnings(audit, drift, { turnover: 0.05 }).length, 1);
});

/* --------------------------------------------------------------- storage */

test('the corpus round trips through the history file', () => {
  const db = store.open(':memory:');

  store.save(db, {
    label: 'main', dataset: 'cases.jsonl', metrics: { 'recall@k': 1 }, cases: 1, corpus: ['a', 'b'],
  });

  assert.deepEqual(store.latest(db, 'main').corpus, ['a', 'b']);
  db.close();
});

test('a run saved without a corpus reads back as null, not an empty corpus', () => {
  // The difference matters: an empty array would compare as "every document
  // left the corpus" against the next run.
  const db = store.open(':memory:');

  store.save(db, { label: 'main', dataset: 'cases.jsonl', metrics: {}, cases: 1 });

  assert.equal(store.latest(db, 'main').corpus, null);
  assert.equal(compareCorpus(['a'], store.latest(db, 'main').corpus ?? []), null);
  db.close();
});

test('a history file written before the column existed keeps its runs', async () => {
  // Exactly what an existing install has on disk. CREATE TABLE IF NOT EXISTS
  // leaves an older table alone, so without the migration every read of corpus
  // comes back undefined and the drift check silently never fires.
  const { DatabaseSync } = await import('node:sqlite');
  const { mkdtemp, rm } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');

  const dir = await mkdtemp(join(tmpdir(), 'ragbench-'));
  const file = join(dir, 'history.db');

  const old = new DatabaseSync(file);
  old.exec(`CREATE TABLE runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT NOT NULL, dataset TEXT NOT NULL,
    cases INTEGER NOT NULL, missing INTEGER NOT NULL DEFAULT 0, metrics TEXT NOT NULL,
    git_ref TEXT, created_at TEXT NOT NULL)`);
  old.prepare('INSERT INTO runs (label, dataset, cases, missing, metrics, created_at) VALUES (?,?,?,?,?,?)')
    .run('main', 'cases.jsonl', 7, 0, '{"recall@k":0.9}', new Date().toISOString());
  old.close();

  const db = store.open(file);

  // The old run survived, and reads back with a null corpus rather than a hole.
  const before = store.latest(db, 'main');
  assert.equal(before.cases, 7);
  assert.equal(before.corpus, null);

  // And the migrated file takes a corpus from here on.
  store.save(db, { label: 'main', dataset: 'cases.jsonl', metrics: {}, cases: 1, corpus: ['a'] });
  assert.deepEqual(store.latest(db, 'main').corpus, ['a']);

  db.close();
  await rm(dir, { recursive: true, force: true });
});
