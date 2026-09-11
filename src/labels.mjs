/**
 * Whether the relevance labels still describe the corpus.
 *
 * Every retrieval metric in this tool is computed against relevant_docs, which
 * somebody wrote by hand against the corpus as it stood on the day they wrote
 * it. The ranker gets re-run on every commit. The labels do not. Documents get
 * re-chunked, re-ided, merged and dropped, and the labels quietly stop
 * pointing at anything that exists, at which point nDCG is applying its
 * discount curve to a stale notion of relevant and every number under it is
 * measuring the wrong thing with great precision.
 *
 * Nothing in here gates a build. A corpus that changed shape is usually
 * somebody doing their job, and a check that failed the build for it would be
 * switched off within a week. What this does is say when the labels are worth
 * re-auditing, which is the only thing that actually fixes them.
 */

/**
 * What the labels look like against what retrieval actually returned.
 *
 * The corpus here is not the real corpus. It is the set of document ids that
 * came back for at least one question in this run, which is all a scorer can
 * see from the outside. That makes the orphan signal weaker than "this
 * document was deleted" and, usefully, more relevant than it: a labelled
 * document that no question retrieves is unreachable at this k whether it was
 * deleted or merely buried, and either way recall for the cases that need it
 * cannot reach 1 no matter what the ranker does.
 *
 * @param {Array<{testCase: object, prediction: object}>} paired
 */
export function auditLabels(paired = []) {
  const corpus = new Set();

  for (const { prediction } of paired) {
    for (const id of prediction?.retrieved ?? []) corpus.add(String(id));
  }

  const orphans = new Map();
  const labelled = new Set();
  let labelledCases = 0;

  for (const { testCase } of paired) {
    const labels = testCase?.relevant_docs ?? [];
    if (!labels.length) continue;

    labelledCases += 1;

    for (const raw of labels) {
      const id = String(raw);
      labelled.add(id);

      if (corpus.has(id)) continue;

      if (!orphans.has(id)) orphans.set(id, []);
      orphans.get(id).push(testCase.id);
    }
  }

  return {
    labelledCases,
    labelledDocs: labelled.size,
    orphans: [...orphans]
      .map(([docId, cases]) => ({ docId, cases }))
      .sort((a, b) => b.cases.length - a.cases.length || a.docId.localeCompare(b.docId)),
    orphanRate: labelled.size ? orphans.size / labelled.size : 0,
    // Sorted so two runs of the same system produce the same stored value, and
    // a diff of the history file is readable.
    corpus: [...corpus].sort(),
    // The same ids again, without the cases that needed them, because this is
    // the form that gets stored and compared across runs. The cases move for
    // their own reasons and would make two identical orphan sets look
    // different.
    orphanIds: [...orphans.keys()].sort(),
  };
}

/**
 * How much the retrieved corpus moved between two runs.
 *
 * Returns null rather than a zero when there is nothing to compare against,
 * because "no previous run" and "nothing changed" are different facts and a
 * first run should not report stability it has no evidence for.
 */
export function compareCorpus(current = [], previous = []) {
  if (!previous.length) return null;

  const now = new Set(current.map(String));
  const before = new Set(previous.map(String));

  const added = [...now].filter((id) => !before.has(id)).sort();
  const removed = [...before].filter((id) => !now.has(id)).sort();
  const union = new Set([...now, ...before]);

  return {
    added,
    removed,
    turnover: union.size ? (added.length + removed.length) / union.size : 0,
  };
}

/** Above this much churn, the labels are worth a look. */
export const DEFAULT_TURNOVER = 0.25;

/**
 * Which labels went unreachable since the baseline, and which came back.
 *
 * The absolute count degrades as a signal exactly when the dataset becomes
 * worth having. Orphans accumulate for benign reasons as a corpus grows, so
 * the number only ever goes up, and a line that reads "3 of 47" every morning
 * stops being read long before it says 4.
 *
 * The change is the thing. "Two more than at the baseline, and here they are"
 * is an event somebody acts on; "three are unreachable" is a standing
 * condition somebody scrolls past.
 *
 * Recovered is reported too, and is not merely symmetry: a label that was
 * unreachable and now is not usually means somebody fixed retrieval, and
 * seeing it confirms the fix reached the thing it was meant to reach.
 */
export function compareOrphans(current = [], baseline = null) {
  if (baseline === null) return null;

  const now = new Set(current.map((o) => (typeof o === 'string' ? o : o.docId)));
  const before = new Set(baseline.map((o) => (typeof o === 'string' ? o : o.docId)));

  const appeared = [...now].filter((id) => !before.has(id)).sort();
  const recovered = [...before].filter((id) => !now.has(id)).sort();

  return {
    appeared,
    recovered,
    before: before.size,
    after: now.size,
    // Positive when the dataset is decaying, negative when somebody is
    // repairing it, and zero when the same documents are unreachable as
    // yesterday, which is a different fact from nothing being wrong.
    net: now.size - before.size,
  };
}

/**
 * Enough ids to act on, and a count for the rest.
 *
 * A warning that prints forty document ids is one nobody finishes reading, and
 * one that silently prints the first five is one that lies about how big the
 * problem is.
 */
function list(ids, show = 5) {
  if (ids.length <= show) return ids.join(', ');
  return `${ids.slice(0, show).join(', ')}, and ${ids.length - show} more`;
}

/**
 * The lines worth putting in front of somebody, and nothing else.
 *
 * Orphans are reported whenever there is one, because a single orphan is not
 * noise: it is a case whose recall is capped below 1 for a reason that has
 * nothing to do with the ranker. Turnover needs a threshold, because a corpus
 * always moves a little and a warning that fires every run is furniture.
 *
 * @returns {Array<{kind: string, message: string}>}
 */
export function labelWarnings(audit, drift = null, { turnover = DEFAULT_TURNOVER, orphanChange = null } = {}) {
  const warnings = [];

  if (audit.orphans.length) {
    const affected = new Set(audit.orphans.flatMap((o) => o.cases));
    const standing =
      `${audit.orphans.length} of ${audit.labelledDocs} labelled documents (${(audit.orphanRate * 100).toFixed(0)} percent) ` +
      `are retrieved for no question in this run, so recall cannot reach 1 for the ` +
      `${affected.size} case(s) that need them, whatever the ranker does.`;

    // The new ones lead when there are any. This is the whole point of holding
    // the previous set: the number is a condition and the change is an event,
    // and putting the event second buries it behind the sentence somebody has
    // already learned to skip.
    const message = orphanChange?.appeared.length
      ? `${orphanChange.appeared.length} labelled document(s) went unreachable since the baseline: ` +
        `${list(orphanChange.appeared)}. ${standing}`
      : orphanChange
        ? `${standing} None of them are new since the baseline, so this is a standing condition ` +
          `rather than something this change caused.`
        : `${standing} They either left the corpus or now rank below k everywhere.`;

    warnings.push({
      kind: 'orphaned-labels',
      count: audit.orphans.length,
      rate: audit.orphanRate,
      appeared: orphanChange?.appeared ?? [],
      message,
    });
  }

  // Worth its own line rather than a clause, because somebody repairing a
  // dataset wants to see the repair land.
  if (orphanChange?.recovered.length) {
    warnings.push({
      kind: 'labels-recovered',
      count: orphanChange.recovered.length,
      message:
        `${orphanChange.recovered.length} labelled document(s) that were unreachable at the baseline ` +
        `are being retrieved again: ${list(orphanChange.recovered)}.`,
    });
  }

  if (drift && drift.turnover >= turnover) {
    warnings.push({
      kind: 'corpus-drift',
      turnover: drift.turnover,
      message:
        `the retrieved corpus turned over ${(drift.turnover * 100).toFixed(0)} percent since the baseline ` +
        `(${drift.added.length} new document(s), ${drift.removed.length} gone). ` +
        `Labels written against the old shape are worth re-auditing on a sample before trusting this comparison.`,
    });
  }

  return warnings;
}
