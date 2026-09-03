# Expected Structure

```text
tests/rust-python-harness/
├── __main__.py
├── cli/
│   ├── __init__.py
│   ├── catalog.py
│   ├── selection.py
│   └── commands.py
│
├── strategies/
│   ├── e2e_parity/
│   │   ├── __init__.py
│   │   ├── sdk/
│   │   │   ├── ocr/
│   │   │   ├── messages/
│   │   │   ├── chat_completions/
│   │   │   └── responses/
│   │   └── gateway/
│   │
│   ├── trace_parity/
│   │   ├── __init__.py
│   │   ├── sdk/
│   │   └── gateway/
│   │
│   ├── unit_tests_mapping/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── ledger_report.py
│   │   └── mapping_validator.py
│   │
│   ├── unit_tests_parity/
│   │   ├── __init__.py
│   │   └── runner.py
│   │
│   └── unit_tests_rust/
│       ├── __init__.py
│       └── runner.py
│
└── shared/
    ├── parity/
    ├── tracing/
    ├── reporting/
    │   └── strategy.py
    └── unit_runners/
        └── suite_runner.py
```

- A strategy is a folder under `strategies/` with a `strategy.json` manifest, a one-line `AGENTS.md`, and an `__init__.py` exporting exactly one `STRATEGY: StrategyDefinition`; the manifest id must equal the folder name
- `shared/reporting/strategy.py` is the contract: case specs (`SelectorCaseSpec` for pytest-driven cells, `SuiteCaseSpec` for JSON-suite-driven cells), the runner/checker protocols, and `StrategyDefinition`
- Run locally only; no CI integration
- `python -m tests.rust-python-harness run|list|check` selects strategies, lists the catalog, or runs per-strategy consistency checks; every verb shares `--strategy`, `--function`, `--surface`, and `-i`, for example `run --strategy e2e_parity --surface sdk --function ocr --plain`, `list --strategy unit_tests_mapping`, or `check --strategy unit_tests_mapping --function ocr`
- `cli/catalog.py` discovers strategies, validates each manifest against its declared case spec, and orders them; `cli/selection.py` filters and drives the interactive picker; `cli/commands.py` implements the verbs
- `e2e_parity/` compares SDK objects, exceptions, callbacks, and streams, or gateway HTTP responses
- `trace_parity/` compares mapped operations, call counts, and required execution ordering
- E2E and trace strategies run pytest through `shared/reporting/pytest_runner.py`; surface-specific execution lives in their folders
- `unit_tests_mapping/runner.py` keeps the suite model and suite execution; `unit_tests_mapping/ledger_report.py` turns ledger reports into check output; `unit_tests_mapping/mapping_validator.py` matches Python/Rust tests by agreed names or annotations and reports missing or ambiguous counterparts
- `unit_tests_parity/runner.py` runs existing Python unit tests with `LITELLM_RUST=0` and `LITELLM_RUST=1` in separate processes and requires matching outcomes, including failures
- `unit_tests_rust/runner.py` runs Cargo tests; native Rust unit tests stay beside their implementation
- `shared/unit_runners/suite_runner.py` runs the JSON-suite loop behind every suite-driven strategy with nodeids of the form `suite:<strategy_id>:<function>:<path>`
- `shared/` contains reusable parity, tracing, reporting, and unit-runner machinery
- Keep fixtures with their owning API and existing Python tests in their current locations
- Each strategy folder carries an `AGENTS.md` one-liner stating what it should be doing
- Run the harness's own checks with `uv run pytest -o consider_namespace_packages=true tests/rust-python-harness/shared tests/rust-python-harness/cli tests/rust-python-harness/strategies/unit_tests_mapping tests/rust-python-harness/strategies/unit_tests_parity tests/rust-python-harness/strategies/unit_tests_rust tests/test_rust_python_harness.py -q`
