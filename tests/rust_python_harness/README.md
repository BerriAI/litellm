# Rust ↔ Python SDK parity harness

This folder is the operator-facing harness for the Rust migration test plan. It runs pytest normally, listens to test events in-process, and redraws a live matrix grouped by testing strategy and SDK-level function.

The matrix always has these SDK columns:

- `ocr / aocr`
- `messages / amessages`
- `responses / aresponses`
- `count_tokens`

Each numbered section in the parity TDD owns one folder below [`strategies/`](strategies/):

| TDD section | Strategy folder |
| --- | --- |
| 1 | `end_to_end/` |
| 2a | `transform_request/` |
| 2b | `transform_response/` |
| 2c | `transform_stream/` |
| 3 | `cassettes/` |
| 4 | `callbacks/` |
| 5 | `manifest_coverage/` |
| 6a | `dual_build_suite/` |
| 6b | `shadow_mode/` |

## Run it

From the repository root:

```bash
poetry run python -m tests.rust_python_harness
```

The default runs every configured test once and updates all matching cells in real time. Narrow a run by strategy, SDK function, or both:

```bash
poetry run python -m tests.rust_python_harness --strategy end_to_end
poetry run python -m tests.rust_python_harness --function messages
poetry run python -m tests.rust_python_harness --strategy end_to_end --function ocr
```

For a guided run, use the interactive picker. It asks which strategy rows and SDK
function columns to include, then hands the terminal to the live dashboard. It never
captures keys while tests are running, so Ctrl-C and pytest debugging remain safe.

```bash
poetry run python -m tests.rust_python_harness --interactive
```

Useful operator options:

```bash
# Inspect coverage and pytest selectors without running anything.
poetry run python -m tests.rust_python_harness --list

# Stable line-oriented output for CI logs or redirected output.
poetry run python -m tests.rust_python_harness --plain

# Forward pytest options. Use the equals form when the value begins with a dash.
poetry run python -m tests.rust_python_harness --pytest-arg=-x
```

The process returns pytest's exit code. A configured selector that collects no test is also a failure. A planned cell has no selector yet and does not fail the run.

The dashboard adapts to narrow terminals, shows elapsed time and unique-test progress,
and prints the three slowest tests when the run ends. Each failure includes a focused
`poetry run pytest ... -q` command. Redirected output and CI automatically use the
line-oriented plain renderer; `--plain` lets you opt into it locally.

## Read the matrix

| Mark | Meaning |
| --- | --- |
| `✓` | All collected tests passed |
| `✗` | At least one test failed |
| `!` | Test setup or teardown failed |
| `↷` | All collected tests skipped |
| `?` | A configured selector did not collect a test |
| `—` | Strategy is planned but has no test yet |
| `n/a` | Strategy does not apply to this SDK function |
| `◐` | The configured tests cover only part of the TDD's parity contract |

The initial end-to-end entries deliberately show `◐`: the repository has Rust bridge tests for OCR, Messages, and Responses websocket plumbing, but those are not yet frozen-Python-oracle comparisons. The remaining TDD cells stay visible as planned work instead of disappearing from a green summary.

## Attach parity tests

Every strategy folder contains a `strategy.json`. Add a pytest file or node ID to the appropriate SDK function's `selectors` list:

```json
{
  "coverage": "complete",
  "selectors": [
    "tests/rust_python_harness/strategies/transform_request/test_messages.py"
  ]
}
```

Selectors use the same syntax as pytest. A file selector aggregates every test in the file; a node selector can target one test or parametrized family. The runner deduplicates selectors, so one test may intentionally prove more than one cell without executing twice.

Use these coverage values:

- `complete`: implements the full strategy contract for that SDK function.
- `partial`: useful coverage exists, but the TDD contract is not fully proven.
- `planned`: no runnable parity test exists yet.
- `not_applicable`: the strategy cannot apply, such as streaming for OCR.

Keep comparison mechanics in shared harness modules and provider/function facts in the owning strategy folder. A Python/Rust mismatch is a test failure; do not normalize away observable return types, exception classes, private response fields, chunk ordering, or callback payload differences merely to make a cell green.

## Architecture

- `catalog.py` validates and loads every strategy manifest.
- `models.py` owns typed strategy, case, coverage, and run-state models.
- `runner.py` maps live pytest events back to one or more matrix cells.
- `ui.py` renders the interactive Rich dashboard and a dependency-free plain fallback.
- `cli.py` handles filtering and preserves pytest exit semantics.

The harness is driven from Python, matching the SDK surface and existing test tooling. Rust remains responsible for the implementation under comparison; the harness does not move provider semantics into the PyO3 bridge.
