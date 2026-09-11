/**
 * Run history, in one SQLite file.
 *
 * Kept local and committed-adjacent rather than shipped to a service, because
 * the whole point is that a solo developer can gate a build without paying a
 * monthly fee or sending their dataset anywhere.
 *
 * A run is stored under a label. "main" is the baseline a pull request is
 * measured against; a branch name is a run you can look at without disturbing
 * it.
 */

import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

const SCHEMA = `
CREATE TABLE IF NOT EXISTS runs (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  label      TEXT NOT NULL,
  dataset    TEXT NOT NULL,
  cases      INTEGER NOT NULL,
  missing    INTEGER NOT NULL DEFAULT 0,
  metrics    TEXT NOT NULL,
  git_ref    TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS runs_label ON runs(label, created_at DESC);

CREATE TABLE IF NOT EXISTS case_scores (
  run_id  INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  case_id TEXT NOT NULL,
  scores  TEXT NOT NULL,
  PRIMARY KEY (run_id, case_id)
);
`;

export function open(path = '.ragbench/history.db') {
  if (path !== ':memory:') mkdirSync(dirname(path), { recursive: true });

  const db = new DatabaseSync(path);
  db.exec('PRAGMA foreign_keys = ON');
  db.exec(SCHEMA);
  migrate(db);

  return db;
}

/**
 * Columns added after the first release.
 *
 * CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so a
 * history file written by an older version keeps its old shape and every read
 * of a new column comes back undefined. Adding them here means an existing
 * database keeps its runs instead of needing to be thrown away, and the older
 * runs simply carry a null, which the callers already have to handle because
 * the first run of any database has no predecessor either.
 */
function migrate(db) {
  const columns = new Set(db.prepare('PRAGMA table_info(runs)').all().map((c) => c.name));

  if (!columns.has('corpus')) db.exec('ALTER TABLE runs ADD COLUMN corpus TEXT');

  // The orphan set, not just how many there were.
  //
  // The count alone cannot produce a delta worth reading: two runs showing
  // three orphans each can be three entirely different documents, and a
  // reader told "still three" would reasonably assume nothing moved. The
  // corpus already stored here cannot be used to recompute them either,
  // because the labels move between runs as well.
  if (!columns.has('orphans')) db.exec('ALTER TABLE runs ADD COLUMN orphans TEXT');
}

export function save(db, { label, dataset, metrics, cases, missing = 0, gitRef = null, caseScores = [], corpus = null, orphans = null }) {
  const now = new Date().toISOString();

  // The document ids retrieval actually returned, so a later run can tell
  // whether the corpus moved under the labels. Stored as the sorted list
  // rather than a hash, because "something changed" is not actionable and
  // "forty documents left" is. It is bounded by cases times k.
  const result = db
    .prepare('INSERT INTO runs (label, dataset, cases, missing, metrics, git_ref, created_at, corpus, orphans) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)')
    .run(
      label, dataset, cases, missing, JSON.stringify(metrics), gitRef, now,
      corpus ? JSON.stringify(corpus) : null,
      orphans ? JSON.stringify(orphans) : null,
    );

  const runId = Number(result.lastInsertRowid);

  if (caseScores.length) {
    const insert = db.prepare('INSERT INTO case_scores (run_id, case_id, scores) VALUES (?, ?, ?)');
    db.exec('BEGIN');
    try {
      for (const { id, scores } of caseScores) insert.run(runId, String(id), JSON.stringify(scores));
      db.exec('COMMIT');
    } catch (error) {
      db.exec('ROLLBACK');
      throw error;
    }
  }

  return { id: runId, label, createdAt: now };
}

/** The most recent run under a label, or null. */
export function latest(db, label) {
  const row = db.prepare('SELECT * FROM runs WHERE label = ? ORDER BY created_at DESC, id DESC LIMIT 1').get(label);
  return row ? hydrate(row) : null;
}

export function history(db, { label = null, limit = 20 } = {}) {
  const rows = label
    ? db.prepare('SELECT * FROM runs WHERE label = ? ORDER BY id DESC LIMIT ?').all(label, limit)
    : db.prepare('SELECT * FROM runs ORDER BY id DESC LIMIT ?').all(limit);

  return rows.map(hydrate);
}

/**
 * Which cases changed between two runs.
 *
 * The aggregate tells you something moved; this tells you which case, which is
 * the difference between a number to worry about and a bug to fix.
 */
export function regressedCases(db, baselineRunId, currentRunId, { metric = 'recall@k', tolerance = 0 } = {}) {
  const before = new Map(
    db.prepare('SELECT case_id, scores FROM case_scores WHERE run_id = ?').all(baselineRunId)
      .map((r) => [r.case_id, JSON.parse(r.scores)]),
  );

  const changed = [];

  for (const row of db.prepare('SELECT case_id, scores FROM case_scores WHERE run_id = ?').all(currentRunId)) {
    const now = JSON.parse(row.scores);
    const then = before.get(row.case_id);

    if (!then || now[metric] === undefined || then[metric] === undefined) continue;

    const delta = now[metric] - then[metric];
    if (delta < -tolerance) {
      changed.push({ caseId: row.case_id, metric, before: then[metric], now: now[metric], delta });
    }
  }

  return changed.sort((a, b) => a.delta - b.delta);
}

function hydrate(row) {
  return {
    id: row.id,
    label: row.label,
    dataset: row.dataset,
    cases: row.cases,
    missing: row.missing,
    metrics: JSON.parse(row.metrics),
    gitRef: row.git_ref,
    createdAt: row.created_at,
    corpus: row.corpus ? JSON.parse(row.corpus) : null,
    orphans: row.orphans ? JSON.parse(row.orphans) : null,
  };
}
