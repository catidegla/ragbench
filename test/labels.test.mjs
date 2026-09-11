import { test } from 'node:test';
import assert from 'node:assert/strict';

import { auditLabels, compareCorpus, compareOrphans, labelWarnings, DEFAULT_TURNOVER } from '../src/labels.mjs';
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

/* --------------------------------------------------- the orphan delta */

test('the audit hands back a canonical id list for storing', () => {
  const audit = auditLabels([
    pair('a', ['doc/9', 'doc/2'], ['doc/1']),
    pair('b', ['doc/2'], ['doc/1']),
  ]);

  // Sorted by id, and without the cases, because the cases move for their own
  // reasons and would make two identical orphan sets compare as different.
  assert.deepEqual(audit.orphanIds, ['doc/2', 'doc/9']);
});

test('no baseline reports null rather than a delta it cannot compute', () => {
  assert.equal(compareOrphans(['doc/1'], null), null);
});

test('a baseline with no orphans is a baseline, not a missing one', () => {
  // The distinction that makes the whole feature work. An empty array means
  // the previous run had clean labels, and every orphan now is new, which is
  // the single most worth reporting case there is. Treating it as "nothing to
  // compare against" would swallow exactly that.
  const change = compareOrphans(['doc/1'], []);

  assert.deepEqual(change.appeared, ['doc/1']);
  assert.equal(change.net, 1);
});

test('what went unreachable and what came back are both named', () => {
  const change = compareOrphans(['doc/2', 'doc/3'], ['doc/1', 'doc/2']);

  assert.deepEqual(change.appeared, ['doc/3']);
  assert.deepEqual(change.recovered, ['doc/1']);
  assert.equal(change.before, 2);
  assert.equal(change.after, 2);

  // Two in, one out, one back: the count did not move and two things happened.
  assert.equal(change.net, 0);
});

test('the same orphans as yesterday is a delta of nothing, not an absent delta', () => {
  const change = compareOrphans(['doc/1'], ['doc/1']);

  assert.deepEqual(change.appeared, []);
  assert.deepEqual(change.recovered, []);
  assert.equal(change.net, 0);
});

test('audit output compares against stored ids without being reshaped first', () => {
  const audit = auditLabels([pair('a', ['doc/2'], ['doc/1'])]);
  const change = compareOrphans(audit.orphans, ['doc/2']);

  assert.deepEqual(change.appeared, []);
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

test('an orphan that is new since the baseline is named, not just counted', () => {
  const audit = auditLabels([
    pair('a', ['doc/1'], ['doc/9']),
    pair('b', ['doc/2'], ['doc/9']),
  ]);

  const change = compareOrphans(audit.orphanIds, ['doc/1']);
  const [warning] = labelWarnings(audit, null, { orphanChange: change });

  assert.equal(warning.kind, 'orphaned-labels');
  assert.deepEqual(warning.appeared, ['doc/2']);
  // The change leads, because the count behind it is the sentence people
  // have already learned to skip.
  assert.ok(warning.message.startsWith('1 labelled document(s) went unreachable since the baseline'));
  assert.match(warning.message, /doc\/2/);
});

test('a long list of new orphans says how many it is not showing', () => {
  const ids = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7'];
  const audit = auditLabels(ids.map((id, i) => pair(`c${i}`, [id], ['elsewhere'])));
  const [warning] = labelWarnings(audit, null, { orphanChange: compareOrphans(audit.orphanIds, []) });

  assert.match(warning.message, /d1, d2, d3, d4, d5, and 2 more/);
});

test('an orphan that has been there all along is called a standing condition', () => {
  const audit = auditLabels([pair('a', ['doc/1'], ['doc/9'])]);
  const warning = labelWarnings(audit, null, { orphanChange: compareOrphans(audit.orphanIds, ['doc/1']) })[0];

  // The difference between a line worth acting on and a line worth scrolling
  // past, said out loud so nobody has to work it out from the number.
  assert.match(warning.message, /None of them are new since the baseline/);
  assert.deepEqual(warning.appeared, []);
});

test('with no baseline the warning claims nothing about movement', () => {
  const audit = auditLabels([pair('a', ['doc/1'], ['doc/9'])]);
  const [warning] = labelWarnings(audit);

  assert.doesNotMatch(warning.message, /baseline/);
  assert.equal(warning.rate, 1);
});

test('labels coming back gets its own line, so a repair is visible', () => {
  const audit = auditLabels([pair('a', ['doc/1'], ['doc/1'])]);
  const warnings = labelWarnings(audit, null, { orphanChange: compareOrphans(audit.orphanIds, ['doc/1', 'doc/2']) });

  // Nothing is unreachable now, so there is no orphan warning at all, and the
  // only line is the good news.
  assert.equal(warnings.length, 1);
  assert.equal(warnings[0].kind, 'labels-recovered');
  assert.match(warnings[0].message, /doc\/1, doc\/2/);
});

/* --------------------------------------------------------------- storage */

test('the orphan set round trips through the history file', () => {
  const db = store.open(':memory:');

  store.save(db, { label: 'main', dataset: 'd.jsonl', metrics: {}, cases: 1, orphans: ['doc/2'] });
  assert.deepEqual(store.latest(db, 'main').orphans, ['doc/2']);

  db.close();
});

test('a run with clean labels stores an empty set, not an unknown one', () => {
  // These have to read back differently. Empty means the labels were clean and
  // every orphan tomorrow is new; null means the run predates the column and a
  // delta against it would be invented.
  const db = store.open(':memory:');

  store.save(db, { label: 'clean', dataset: 'd.jsonl', metrics: {}, cases: 1, orphans: [] });
  store.save(db, { label: 'old', dataset: 'd.jsonl', metrics: {}, cases: 1 });

  assert.deepEqual(store.latest(db, 'clean').orphans, []);
  assert.equal(store.latest(db, 'old').orphans, null);

  assert.deepEqual(compareOrphans(['doc/1'], store.latest(db, 'clean').orphans).appeared, ['doc/1']);
  assert.equal(compareOrphans(['doc/1'], store.latest(db, 'old').orphans), null);

  db.close();
});

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
