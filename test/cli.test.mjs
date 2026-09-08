import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const run = promisify(execFile);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const CLI = join(ROOT, 'bin', 'ragbench.mjs');

const CASES = [
  { id: 'a', question: 'When does it settle?', expected_answer: 'Overnight', relevant_docs: ['doc-1'] },
  { id: 'b', question: 'How long for refunds?', expected_answer: 'One day', relevant_docs: ['doc-2'] },
];

const PERFECT = [
  { id: 'a', answer: 'Overnight', retrieved: ['doc-1'] },
  { id: 'b', answer: 'One day', retrieved: ['doc-2'] },
];

const BROKEN = [
  { id: 'a', answer: 'Overnight', retrieved: ['doc-1'] },
  { id: 'b', answer: 'No idea', retrieved: ['doc-9'] },
];

async function cli(args, cwd) {
  try {
    const { stdout, stderr } = await run(process.execPath, [CLI, ...args], { cwd });
    return { code: 0, stdout, stderr };
  } catch (error) {
    return { code: error.code ?? 1, stdout: error.stdout ?? '', stderr: error.stderr ?? '' };
  }
}

async function workspace(t) {
  const dir = await mkdtemp(join(tmpdir(), 'ragbench-cli-'));
  t.after(() => rm(dir, { recursive: true, force: true }));

  const write = (name, rows) => writeFile(join(dir, name), rows.map((r) => JSON.stringify(r)).join('\n') + '\n');

  await write('cases.jsonl', CASES);
  await write('perfect.jsonl', PERFECT);
  await write('broken.jsonl', BROKEN);

  return dir;
}

test('a perfect run scores one across the board', async (t) => {
  const dir = await workspace(t);

  const result = await cli(['run', '--dataset', 'cases.jsonl', '--predictions', 'perfect.jsonl', '--json', '--no-save'], dir);
  const report = JSON.parse(result.stdout);

  assert.equal(result.code, 0);
  assert.equal(report.metrics['recall@k'], 1);
  assert.equal(report.metrics.exact_match, 1);
});

test('gate fails the build on a regression and exits non-zero', async (t) => {
  const dir = await workspace(t);

  await cli(['run', '--dataset', 'cases.jsonl', '--predictions', 'perfect.jsonl', '--label', 'main'], dir);
  const gated = await cli(['gate', '--dataset', 'cases.jsonl', '--predictions', 'broken.jsonl', '--baseline', 'main', '--no-save'], dir);

  // The exit code is the entire point. A gate that reports a regression and
  // exits zero is decorative.
  assert.equal(gated.code, 1);
  assert.match(gated.stdout, /fell/);
});

test('gate passes when nothing regressed', async (t) => {
  const dir = await workspace(t);

  await cli(['run', '--dataset', 'cases.jsonl', '--predictions', 'perfect.jsonl', '--label', 'main'], dir);
  const gated = await cli(['gate', '--dataset', 'cases.jsonl', '--predictions', 'perfect.jsonl', '--baseline', 'main', '--no-save'], dir);

  assert.equal(gated.code, 0);
});

test('a first run with no baseline passes but says nothing was compared', async (t) => {
  const dir = await workspace(t);

  const gated = await cli(['gate', '--dataset', 'cases.jsonl', '--predictions', 'broken.jsonl', '--baseline', 'main', '--no-save'], dir);

  assert.equal(gated.code, 0);
  assert.match(gated.stdout, /no baseline recorded yet/);
});

test('a threshold fails even with no baseline', async (t) => {
  const dir = await workspace(t);

  const gated = await cli(
    ['gate', '--dataset', 'cases.jsonl', '--predictions', 'broken.jsonl', '--threshold', 'recall@k=0.9', '--no-save'],
    dir,
  );

  assert.equal(gated.code, 1);
  assert.match(gated.stdout, /below the 0.9 threshold/);
});

test('a malformed dataset stops the run rather than scoring a subset', async (t) => {
  const dir = await workspace(t);
  await writeFile(join(dir, 'bad.jsonl'), '{ truncated\n');

  const result = await cli(['run', '--dataset', 'bad.jsonl', '--predictions', 'perfect.jsonl', '--no-save'], dir);

  assert.equal(result.code, 2);
  assert.match(result.stderr, /not valid JSON/);
});

test('exec runs a command and reads its predictions from stdout', async (t) => {
  const dir = await workspace(t);

  // A tiny stand-in for a RAG system: reads cases on stdin, answers on stdout.
  await writeFile(
    join(dir, 'fake-rag.mjs'),
    `
    let input = '';
    process.stdin.on('data', (c) => (input += c)).on('end', () => {
      for (const line of input.split('\\n').filter(Boolean)) {
        const c = JSON.parse(line);
        process.stdout.write(JSON.stringify({ id: c.id, answer: c.expected_answer, retrieved: c.relevant_docs }) + '\\n');
      }
    });
    `,
  );

  const result = await cli(
    ['run', '--dataset', 'cases.jsonl', '--exec', `node fake-rag.mjs`, '--json', '--no-save'],
    dir,
  );

  const report = JSON.parse(result.stdout);
  assert.equal(result.code, 0);
  assert.equal(report.metrics.exact_match, 1);
});

test('a missing prediction costs the score rather than being skipped', async (t) => {
  const dir = await workspace(t);
  await writeFile(join(dir, 'partial.jsonl'), JSON.stringify(PERFECT[0]) + '\n');

  const result = await cli(['run', '--dataset', 'cases.jsonl', '--predictions', 'partial.jsonl', '--json', '--no-save'], dir);
  const report = JSON.parse(result.stdout);

  assert.equal(report.missing.length, 1);
  // One perfect answer and one absent one, so half marks rather than full.
  assert.equal(report.metrics.exact_match, 0.5);
});

test('history records runs and shows them back', async (t) => {
  const dir = await workspace(t);

  await cli(['run', '--dataset', 'cases.jsonl', '--predictions', 'perfect.jsonl', '--label', 'main'], dir);
  await cli(['run', '--dataset', 'cases.jsonl', '--predictions', 'broken.jsonl', '--label', 'branch'], dir);

  const history = JSON.parse((await cli(['history', '--json'], dir)).stdout);

  assert.equal(history.length, 2);
  assert.equal(history[0].label, 'branch');
  assert.ok(history[0].metrics['recall@k'] < history[1].metrics['recall@k']);
});

test('markdown output is suitable for a pull request comment', async (t) => {
  const dir = await workspace(t);

  await cli(['run', '--dataset', 'cases.jsonl', '--predictions', 'perfect.jsonl', '--label', 'main'], dir);
  const gated = await cli(
    ['gate', '--dataset', 'cases.jsonl', '--predictions', 'broken.jsonl', '--baseline', 'main', '--markdown', '--no-save'],
    dir,
  );

  assert.equal(gated.code, 1);
  assert.match(gated.stdout, /### ragbench/);
  assert.match(gated.stdout, /\| Metric \| Baseline \| Now \| Change \|/);
});

test('init writes a dataset you can immediately run', async (t) => {
  const dir = await workspace(t);

  const init = await cli(['init', '--dataset', 'fresh.jsonl'], dir);
  assert.equal(init.code, 0);

  const loaded = await cli(['run', '--dataset', 'fresh.jsonl', '--predictions', 'perfect.jsonl', '--json', '--no-save'], dir);

  // The example dataset has to be valid, or the first thing a new user does
  // is hit an error.
  assert.equal(loaded.code, 0);
});

test('help and version behave', async (t) => {
  const dir = await workspace(t);

  assert.match((await cli(['--version'], dir)).stdout.trim(), /^\d+\.\d+\.\d+$/);
  assert.match((await cli([], dir)).stdout, /gate/);
  assert.equal((await cli(['nonsense'], dir)).code, 2);
});
