<div align="center">

# ragbench

**Score a RAG system against a golden dataset, and fail the build when it gets worse.**

No API key. No service. No per-run cost.

[![CI](https://github.com/catidegla/ragbench/actions/workflows/ci.yml/badge.svg)](https://github.com/catidegla/ragbench/actions/workflows/ci.yml)
[![Zero dependencies](https://img.shields.io/badge/dependencies-0-brightgreen)](package.json)
[![Node](https://img.shields.io/badge/node-%E2%89%A522.5-339933)](package.json)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

```bash
npx ragbench gate --baseline main --threshold recall@k=0.8
```

```
  2 cases, baseline "main"

  recall@k       0.5000  -0.5000
  ndcg@k         0.5000  -0.5000
  exact_match    0.5000  -0.5000

  x recall@k fell 0.5000, from 1.0000 to 0.5000

  3 check(s) failed
```

Exit code 1. Your pull request is red, and you know which metric moved before anyone reviews the diff.

## Why this instead of the alternatives

The eval platforms are priced for teams. Confident AI runs Free, Starter at $200 a month and Team at $2,000 with Enterprise above that; Braintrust Pro is $249; Galileo Pro is $100. Each has a free tier, and each meters it: Confident AI's is two seats, one project and five test runs a week, which a build gate exhausts by Tuesday.

The open source libraries, RAGAS and DeepEval, compute good metrics but leave you to build the storage, the comparison and the CI gate yourself. So a solo developer stitches three or four tools together and usually ends up with no gate at all.

Pricing checked on the vendors' own pages on 8 September 2026, and it moves, so check it again before quoting it back at anyone.

The other problem is that almost every metric on offer needs an LLM judge. That costs money per run and returns a slightly different number each time, and neither is acceptable in a build gate. **A check that costs a dollar gets run less, and a check that drifts cannot tell a regression from noise.**

Everything here is deterministic and free. Same dataset, same predictions, same numbers, on any machine.

## Metrics

**Retrieval**, from labelled relevant documents:

| | |
| :--- | :--- |
| `precision@k` | How much noise is in the context window. Predicts hallucination. |
| `recall@k` | Whether the model could possibly have got it right. It cannot cite what retrieval never fetched. |
| `mrr` | Reciprocal rank of the first relevant document. |
| `ndcg@k` | Notices a relevant document sliding from position one to position eight, which precision and recall both consider identical. |
| `hit_rate` | The floor below which nothing else matters. |

**Answers**, from expected text:

| | |
| :--- | :--- |
| `exact_match` | After SQuAD style normalisation, so "The answer." matches "answer". |
| `token_f1` | Partial credit. Counts multiplicity, so padding the answer cannot inflate it. |
| `contains_all` | Required phrases present, for when an amount or a date must appear. |
| `groundedness` | Share of answer content words supported by the retrieved context. A cheap proxy: it will not catch a fluent misreading, but it catches an answer invented wholesale, and it costs nothing. |

A metric whose inputs are absent is **omitted rather than scored zero**, and aggregation skips missing values rather than averaging them in. A partially labelled dataset should report what it can measure, not a column of zeroes that looks like failure.

## The gate

Two checks that answer different questions.

A **threshold** asks "is this good enough". You set it once.

A **regression** asks "is this worse than last time". That is the one that catches the change nobody meant to make, and most tools do not have it. A suite sitting comfortably above its thresholds will absorb a ten point drop without a word.

```bash
ragbench gate --baseline main --tolerance 0.01 --threshold recall@k=0.8
```

The tolerance exists because retrieval scores move slightly for reasons that are not your change, and **a gate that fires on noise gets disabled within a week**. Drops inside it are reported as warnings rather than failures.

A threshold on a metric the run never produced is a **failure**, not a pass. Treating it as passing hides that the check never ran.

## Getting started

```bash
npx ragbench init
```

Writes an example `cases.jsonl`:

```json
{"id": "settlement-timing", "question": "When does the Benin gateway settle?", "expected_answer": "Overnight", "relevant_docs": ["payments/settlement.md"], "must_contain": ["overnight"]}
```

Your system reads cases on stdin and writes one prediction per line. Any language, as long as it speaks JSON lines:

```json
{"id": "settlement-timing", "answer": "It settles overnight", "retrieved": ["payments/settlement.md"]}
```

```bash
ragbench run --dataset cases.jsonl --exec "python my_rag.py"
```

Or produce the file however you like and score it:

```bash
ragbench run --dataset cases.jsonl --predictions out.jsonl
```

## In CI

```yaml
- run: npx ragbench gate --dataset cases.jsonl --exec "python my_rag.py"
                         --baseline main --threshold recall@k=0.8
```

For a pull request comment, `--markdown` emits a table:

| Metric | Baseline | Now | Change |
| :--- | ---: | ---: | ---: |
| recall@k | 1.0000 | 0.5000 | -0.5000 down |
| ndcg@k | 1.0000 | 0.5000 | -0.5000 down |

Record the baseline on merge:

```yaml
- run: npx ragbench run --dataset cases.jsonl --exec "python my_rag.py" --label main
```

History lives in a local SQLite file. Nothing is sent anywhere, which is the point.

## Things it refuses to do quietly

**A case with nothing to check against is refused.** It passes every metric vacuously and lifts your average.

**A duplicate id is refused.** Predictions join by id, so a duplicate makes half the suite vanish without a word.

**A case with no prediction is scored as empty, not skipped.** Skipping means a system that answered nothing scores the same as one that answered perfectly on what it attempted.

**A first run with no baseline says so**, rather than reporting a clean comparison that never happened.

**Every dataset problem is reported in one pass**, with line numbers, so fixing a file is not a game of whack-a-mole.

## What this is not

Not a tracing or observability platform. Langfuse and Phoenix do that well and this does not try.

There is no LLM judge. One would be useful for answer quality and the hook is there, but it will not be on the path that gates your build.

## Testing

```bash
npm test    # 52 tests, nothing to install
```

The metric tests check the arithmetic against hand computed values rather than asserting the code runs. The ndcg test in particular verifies that a document found first scores higher than the same document found last, while precision and recall report both as identical.

## Requirements

Node 22.5 or newer, for the built-in `node:sqlite`. Nothing else.

## License

[MIT](LICENSE)
