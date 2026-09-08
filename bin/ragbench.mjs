#!/usr/bin/env node
/**
 * ragbench
 *
 * Score a RAG system against a golden dataset, and fail the build when it gets
 * worse. No API key, no service, no per-run cost.
 */

import { readFile, writeFile } from 'node:fs/promises';
import { execFile, spawn } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { loadDataset, loadPredictions, join as joinCases } from '../src/dataset.mjs';
import { scoreCase, aggregate } from '../src/metrics.mjs';
import { evaluate, summarise, toMarkdown, DEFAULT_TOLERANCE } from '../src/gate.mjs';
import * as store from '../src/store.mjs';

const run = promisify(execFile);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const pkg = JSON.parse(await readFile(join(ROOT, 'package.json'), 'utf8'));

const argv = process.argv.slice(2);
const command = argv[0] && !argv[0].startsWith('--') ? argv[0] : null;
const has = (name) => argv.includes(`--${name}`);
const value = (name, fallback = null) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? fallback : argv[i + 1];
};
const all = (name) => argv.reduce((acc, a, i) => (a === `--${name}` ? [...acc, argv[i + 1]] : acc), []);

const tty = process.stdout.isTTY && !process.env.NO_COLOR;
const paint = (code, s) => (tty ? `\x1b[${code}m${s}\x1b[0m` : s);
const c = {
  bold: (s) => paint('1', s),
  dim: (s) => paint('2', s),
  green: (s) => paint('32', s),
  red: (s) => paint('31', s),
  yellow: (s) => paint('33', s),
};

function usage() {
  console.log(`
${c.bold('ragbench')} ${pkg.version}
Score a RAG system against a golden dataset, and fail the build when it gets worse.

  ${c.bold('run')}       score predictions against a dataset
  ${c.bold('gate')}      score, compare to a baseline, exit non-zero on a regression
  ${c.bold('history')}   past runs
  ${c.bold('init')}      write an example dataset to start from

Options
  --dataset <file>       golden cases, JSON lines (default: cases.jsonl)
  --predictions <file>   your system's output, JSON lines
  --exec "<command>"     run this to produce predictions, dataset on stdin
  --label <name>         store the run under this label (default: local)
  --baseline <label>     compare against the latest run with this label
  --threshold m=0.8      minimum for a metric, repeatable
  --tolerance <n>        how far a metric may fall before it fails (default: ${DEFAULT_TOLERANCE})
  --k <n>                cutoff for the @k metrics (default: 10)
  --db <file>            history database (default: .ragbench/history.db)
  --markdown             emit a pull request comment
  --json                 machine readable output
  --no-save              score without recording the run

Examples
  ragbench run --dataset cases.jsonl --predictions out.jsonl
  ragbench run --dataset cases.jsonl --exec "python my_rag.py"
  ragbench gate --baseline main --threshold recall@k=0.8
`);
}

/**
 * Feed the dataset to a command on stdin and read predictions from stdout.
 *
 * spawn rather than execFile: the "input" option belongs to execFileSync, and
 * passing it to the async version does nothing at all, so the child sits
 * waiting on a stdin that never closes and the run hangs forever.
 */
function execPredictions(command, cases) {
  const input = cases.map((testCase) => JSON.stringify(testCase)).join('\n') + '\n';

  return new Promise((resolve, reject) => {
    const child = spawn(command, { shell: true, stdio: ['pipe', 'pipe', 'pipe'] });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => (stdout += chunk));
    child.stderr.on('data', (chunk) => (stderr += chunk));

    child.on('error', reject);

    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`"${command}" exited ${code}${stderr ? `: ${stderr.trim()}` : ''}`));
        return;
      }
      resolve(stdout);
    });

    child.stdin.on('error', () => {
      // A command that ignores stdin and exits closes the pipe under us. That
      // is its right, and its output is still worth reading.
    });

    child.stdin.write(input);
    child.stdin.end();
  });
}

async function collect() {
  const datasetFile = value('dataset', 'cases.jsonl');
  const { cases, problems } = await loadDataset(datasetFile);

  if (problems.length) {
    console.error(c.red(`${problems.length} problem(s) in ${datasetFile}:`));
    for (const problem of problems) console.error(`  ${problem}`);
    process.exit(2);
  }

  if (!cases.length) {
    console.error(c.red(`${datasetFile} has no usable cases.`));
    process.exit(2);
  }

  let predictionsRaw;
  let predictionsFile = value('predictions');

  if (value('exec')) {
    predictionsRaw = await execPredictions(value('exec'), cases);
    predictionsFile = '<exec>';
    // Written out so a failing run can be inspected rather than reproduced.
    await writeFile('.ragbench/last-predictions.jsonl', predictionsRaw).catch(() => {});
  } else if (predictionsFile) {
    predictionsRaw = await readFile(predictionsFile, 'utf8');
  } else {
    console.error('Give --predictions <file> or --exec "<command>".');
    process.exit(2);
  }

  const { predictions, problems: predProblems } = await parsePredictions(predictionsRaw, predictionsFile);

  if (predProblems.length) {
    console.error(c.red(`${predProblems.length} problem(s) in predictions:`));
    for (const problem of predProblems) console.error(`  ${problem}`);
    process.exit(2);
  }

  const { paired, missing, extra } = joinCases(cases, predictions);
  const k = Number(value('k', 10));

  const caseScores = paired.map(({ testCase, prediction }) => ({
    id: testCase.id,
    scores: scoreCase(testCase, prediction, { k }),
  }));

  return {
    datasetFile,
    cases,
    metrics: aggregate(caseScores.map((s) => s.scores)),
    caseScores,
    missing,
    extra,
  };
}

/** loadPredictions reads a file; this takes the text so --exec can share it. */
async function parsePredictions(raw, label) {
  const { writeFile: write } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const tmp = join(tmpdir(), `ragbench-${process.pid}.jsonl`);

  await write(tmp, raw);
  const result = await loadPredictions(tmp);

  return { ...result, problems: result.problems.map((p) => p.replace(tmp, label)) };
}

function parseThresholds() {
  const thresholds = {};

  for (const entry of all('threshold')) {
    if (!entry || !entry.includes('=')) {
      console.error(`--threshold takes metric=value, for example --threshold recall@k=0.8`);
      process.exit(2);
    }
    const [metric, raw] = entry.split('=');
    const parsed = Number(raw);

    if (Number.isNaN(parsed)) {
      console.error(`--threshold ${metric}=${raw} is not a number`);
      process.exit(2);
    }
    thresholds[metric] = parsed;
  }

  return thresholds;
}

async function gitRef() {
  try {
    const { stdout } = await run('git', ['rev-parse', '--short', 'HEAD']);
    return stdout.trim();
  } catch {
    return null;
  }
}

function printMetrics(metrics, comparisons = []) {
  const byMetric = new Map(comparisons.map((x) => [x.metric, x]));

  console.log('');
  for (const [metric, v] of Object.entries(metrics)) {
    const comparison = byMetric.get(metric);
    let delta = '';

    if (comparison) {
      const d = comparison.delta;
      const sign = d >= 0 ? '+' : '';
      delta = Math.abs(d) < 0.0001
        ? c.dim('  no change')
        : d > 0
          ? c.green(`  ${sign}${d.toFixed(4)}`)
          : c.red(`  ${sign}${d.toFixed(4)}`);
    }

    console.log(`  ${metric.padEnd(14)} ${v.toFixed(4)}${delta}`);
  }
  console.log('');
}

const commands = {
  async run() {
    const result = await collect();

    if (has('json')) {
      console.log(JSON.stringify({ metrics: result.metrics, missing: result.missing, cases: result.cases.length }, null, 2));
      return;
    }

    console.log(`\n  ${c.dim(`${result.cases.length} cases from ${result.datasetFile}`)}`);
    if (result.missing.length) {
      console.log(`  ${c.yellow(`${result.missing.length} case(s) had no prediction and were scored as empty`)}`);
    }
    if (result.extra.length) {
      console.log(`  ${c.dim(`${result.extra.length} prediction(s) matched no case and were ignored`)}`);
    }

    printMetrics(result.metrics);

    if (!has('no-save')) {
      const db = store.open(value('db', '.ragbench/history.db'));
      const saved = store.save(db, {
        label: value('label', 'local'),
        dataset: result.datasetFile,
        metrics: result.metrics,
        cases: result.cases.length,
        missing: result.missing.length,
        gitRef: await gitRef(),
        caseScores: result.caseScores,
      });
      db.close();
      console.log(c.dim(`  saved as run ${saved.id} under "${saved.label}"\n`));
    }
  },

  async gate() {
    const result = await collect();
    const db = store.open(value('db', '.ragbench/history.db'));

    const baselineLabel = value('baseline', 'main');
    const baseline = store.latest(db, baselineLabel);

    const verdict = evaluate(result.metrics, baseline?.metrics ?? null, {
      thresholds: parseThresholds(),
      tolerance: Number(value('tolerance', DEFAULT_TOLERANCE)),
    });

    if (!has('no-save')) {
      store.save(db, {
        label: value('label', 'local'),
        dataset: result.datasetFile,
        metrics: result.metrics,
        cases: result.cases.length,
        missing: result.missing.length,
        gitRef: await gitRef(),
        caseScores: result.caseScores,
      });
    }
    db.close();

    if (has('markdown')) {
      console.log(toMarkdown(verdict, result.metrics));
      process.exit(verdict.passed ? 0 : 1);
    }

    if (has('json')) {
      console.log(JSON.stringify({ verdict, metrics: result.metrics }, null, 2));
      process.exit(verdict.passed ? 0 : 1);
    }

    console.log(`\n  ${c.dim(`${result.cases.length} cases, baseline "${baselineLabel}"`)}`);
    if (!verdict.comparedToBaseline) {
      // Said out loud, because otherwise a first run looks like a clean
      // comparison rather than no comparison at all.
      console.log(`  ${c.yellow('no baseline recorded yet, so nothing was compared')}`);
    }

    printMetrics(result.metrics, verdict.comparisons);

    for (const warning of verdict.warnings) console.log(`  ${c.yellow('~')} ${warning.message}`);
    for (const failure of verdict.failures) console.log(`  ${c.red('x')} ${failure.message}`);

    console.log('');
    console.log(verdict.passed ? `  ${c.green(summarise(verdict, result.metrics))}` : `  ${c.red(summarise(verdict, result.metrics))}`);
    console.log('');

    process.exit(verdict.passed ? 0 : 1);
  },

  async history() {
    const db = store.open(value('db', '.ragbench/history.db'));
    const runs = store.history(db, { label: value('label'), limit: Number(value('limit', 20)) });
    db.close();

    if (has('json')) return console.log(JSON.stringify(runs, null, 2));

    if (!runs.length) {
      console.log('\n  No runs recorded yet.\n');
      return;
    }

    console.log('');
    for (const run of runs) {
      const headline = ['recall@k', 'ndcg@k', 'token_f1'].find((m) => m in run.metrics);
      const score = headline ? `${headline} ${run.metrics[headline].toFixed(4)}` : '';

      console.log(
        `  ${String(run.id).padStart(4)}  ${run.createdAt.slice(0, 16).replace('T', ' ')}  ` +
        `${run.label.padEnd(12)} ${c.dim(`${run.cases} cases  ${score}  ${run.gitRef ?? ''}`)}`,
      );
    }
    console.log('');
  },

  async init() {
    const file = value('dataset', 'cases.jsonl');

    const examples = [
      {
        id: 'settlement-timing',
        question: 'When does the Benin gateway settle?',
        expected_answer: 'Overnight',
        relevant_docs: ['payments/settlement.md'],
        must_contain: ['overnight'],
        tags: ['payments'],
      },
      {
        id: 'refund-window',
        question: 'How long do refunds take?',
        expected_answer: 'One business day',
        relevant_docs: ['payments/refunds.md'],
        tags: ['payments'],
      },
    ];

    await writeFile(file, examples.map((e) => JSON.stringify(e)).join('\n') + '\n');

    console.log(`\n  Wrote ${examples.length} example cases to ${file}`);
    console.log(c.dim('\n  Your system should read cases on stdin and write one prediction per line:'));
    console.log(c.dim('    {"id": "settlement-timing", "answer": "...", "retrieved": ["payments/settlement.md"]}'));
    console.log(c.dim(`\n  Then: ragbench run --dataset ${file} --exec "your-command"\n`));
  },
};

if (has('version')) {
  console.log(pkg.version);
} else if (!command || has('help') || command === 'help') {
  usage();
} else if (commands[command]) {
  try {
    await commands[command]();
  } catch (error) {
    console.error(`ragbench: ${error.message}`);
    process.exit(2);
  }
} else {
  console.error(`Unknown command: ${command}`);
  usage();
  process.exit(2);
}
