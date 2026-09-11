/**
 * The build gate.
 *
 * Two kinds of check, and they answer different questions.
 *
 * A **threshold** asks "is this good enough", which you set once and rarely
 * change. A **regression** asks "is this worse than last time", which is the
 * one that catches the change nobody meant to make. Most tools offer only the
 * first, and a suite that sits comfortably above its thresholds will happily
 * absorb a ten point drop without a word.
 *
 * The tolerance exists because retrieval scores move slightly for reasons that
 * are not your change, and a gate that fires on noise gets disabled within a
 * week. It is not there to let real regressions through.
 */

export const DEFAULT_TOLERANCE = 0.01;

/**
 * @param {object} current    aggregated metrics from this run
 * @param {object|null} baseline  aggregated metrics from the reference run
 * @param {object} options
 * @param {object} options.thresholds  metric -> minimum acceptable value
 * @param {number} options.tolerance   how far a metric may fall before it counts
 * @param {string[]} options.gateOn    which metrics gate; empty means all shared ones
 */
export function evaluate(current, baseline, { thresholds = {}, tolerance = DEFAULT_TOLERANCE, gateOn = [] } = {}) {
  const failures = [];
  const warnings = [];
  const comparisons = [];

  for (const [metric, minimum] of Object.entries(thresholds)) {
    const value = current[metric];

    if (value === undefined) {
      // A threshold on a metric the run did not produce is a configuration
      // mistake, and treating it as a pass hides that the check never ran.
      failures.push({
        kind: 'missing',
        metric,
        message: `${metric} has a threshold of ${minimum} but was not produced by this run`,
      });
      continue;
    }

    if (value < minimum) {
      failures.push({
        kind: 'threshold',
        metric,
        value,
        minimum,
        message: `${metric} is ${value.toFixed(4)}, below the ${minimum} threshold`,
      });
    }
  }

  if (baseline) {
    const metrics = gateOn.length
      ? gateOn
      : Object.keys(current).filter((m) => m in baseline);

    for (const metric of metrics) {
      const now = current[metric];
      const before = baseline[metric];

      if (now === undefined || before === undefined) continue;

      const delta = now - before;
      comparisons.push({ metric, before, now, delta });

      if (delta < -tolerance) {
        failures.push({
          kind: 'regression',
          metric,
          value: now,
          baseline: before,
          delta,
          message: `${metric} fell ${Math.abs(delta).toFixed(4)}, from ${before.toFixed(4)} to ${now.toFixed(4)}`,
        });
      } else if (delta < 0) {
        // Inside tolerance. Worth seeing, not worth failing on.
        warnings.push({
          kind: 'drift',
          metric,
          delta,
          message: `${metric} slipped ${Math.abs(delta).toFixed(4)}, within the ${tolerance} tolerance`,
        });
      }
    }
  }

  return {
    passed: failures.length === 0,
    failures,
    warnings,
    comparisons,
    // Stated explicitly so a report can say "no baseline" rather than implying
    // a comparison happened and found nothing.
    comparedToBaseline: Boolean(baseline),
  };
}

/** A one line summary suitable for a commit status. */
export function summarise(verdict, current) {
  if (!verdict.passed) {
    const first = verdict.failures[0];
    return `${verdict.failures.length} check(s) failed: ${first.message}`;
  }

  const headline = ['recall@k', 'ndcg@k', 'token_f1', 'exact_match'].find((m) => m in current);

  if (!headline) return 'All checks passed';

  return verdict.comparedToBaseline
    ? `All checks passed, ${headline} at ${current[headline].toFixed(4)}`
    : `All checks passed, ${headline} at ${current[headline].toFixed(4)} (no baseline to compare against)`;
}

/**
 * Markdown for a pull request comment.
 *
 * `labels` carries the label audit's warnings, which are about the dataset
 * rather than the change, so they appear under their own heading and never
 * affect the verdict line above them.
 */
export function toMarkdown(verdict, current, { title = 'ragbench', labels = [] } = {}) {
  const lines = [`### ${title}`, ''];

  lines.push(verdict.passed ? '**Passed.**' : `**Failed.** ${verdict.failures.length} check(s).`);
  lines.push('');

  if (verdict.comparisons.length) {
    lines.push('| Metric | Baseline | Now | Change |');
    lines.push('| :--- | ---: | ---: | ---: |');

    for (const c of verdict.comparisons) {
      const arrow = c.delta > 0.0001 ? 'up' : c.delta < -0.0001 ? 'down' : 'flat';
      lines.push(
        `| ${c.metric} | ${c.before.toFixed(4)} | ${c.now.toFixed(4)} | ${c.delta >= 0 ? '+' : ''}${c.delta.toFixed(4)} ${arrow} |`,
      );
    }
  } else {
    lines.push('| Metric | Value |');
    lines.push('| :--- | ---: |');
    for (const [metric, value] of Object.entries(current)) {
      lines.push(`| ${metric} | ${value.toFixed(4)} |`);
    }
    if (!verdict.comparedToBaseline) {
      lines.push('');
      lines.push('_No baseline recorded yet, so nothing was compared._');
    }
  }

  if (verdict.failures.length) {
    lines.push('', '**Failures**', '');
    for (const f of verdict.failures) lines.push(`- ${f.message}`);
  }

  if (verdict.warnings.length) {
    lines.push('', '**Within tolerance**', '');
    for (const w of verdict.warnings) lines.push(`- ${w.message}`);
  }

  if (labels.length) {
    lines.push('', '**Labels**', '');
    for (const l of labels) lines.push(`- ${l.message}`);
  }

  return lines.join('\n');
}
