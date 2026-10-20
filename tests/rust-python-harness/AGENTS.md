# Expected Structure

```text
tests/rust-python-harness/
├── __main__.py
├── cli/
│   ├── __init__.py
│   ├── catalog.py
│   └── commands.py
│
├── strategies/
│   ├── e2e_parity/
│   │   ├── __init__.py
│   │   ├── reporting.py
│   │   ├── sdk/
│   │   │   └── responses/
│   │
│   ├── trace_parity/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── reporting.py
│   │   └── sdk/
│   │       ├── chat_completions/
│   │       ├── messages/
│   │       ├── ocr/
│   │       └── transcription/
│   │
│   ├── unit_tests_parity/
│   │   ├── __init__.py
│   │   ├── reporting.py
│   │   └── runner.py
│   │
│   └── unit_tests_rust/
│       ├── __init__.py
│       ├── reporting.py
│       └── runner.py
│
└── shared/
    ├── parity/
    ├── tracing/
    ├── reporting/
    │   └── strategy.py
    └── unit_runners/
        ├── contracts.py
        └── suite_runner.py
```

- A strategy is a folder under `strategies/` with a one-line `AGENTS.md` and an `__init__.py` exporting exactly one `STRATEGY: StrategyDefinition`. Its id must equal the folder name
- `shared/reporting/strategy.py` is the contract: runnable module/suite specs, not-implemented/skipped specs, the runner protocol, and `StrategyDefinition`
- Every `STRATEGY` explicitly classifies every SDK function. Surface-aware strategies declare their surfaces and classify the complete surface-by-function matrix
- Run locally only. No CI integration
- `python -m tests.rust-python-harness run <strategy>|all` runs the selected strategy. `--function` is common, while each strategy exposes only its supported options
- Examples: `run e2e_parity --surface sdk --function ocr`, `run unit_tests_parity --function ocr --pytest-arg=-x`, or `run all --function ocr`
- `cli/catalog.py` discovers strategies, validates their Python definitions, and orders them. `cli/__init__.py` builds the Click command tree. `cli/commands.py` runs selected cases
- `e2e_parity/` compares SDK objects, exceptions, callbacks, and streams, or gateway HTTP responses
- `e2e_parity/sdk/<route>/` holds the pytest parity files that send one public SDK call through the Python implementation and through the native route against the same recorded provider response, then compare dispatch, the provider request, the response and the success log. Every known gap is a strict xfail naming the gap. Run them with `uv run pytest -o consider_namespace_packages=true tests/rust-python-harness/strategies/e2e_parity/sdk/responses -q`. They reuse the recording server and isolation helpers from `tests/test_litellm_rust/support`, and they never belong in `tests/test_litellm_rust`, which is native only
- `trace_parity/` profiles the Python call stack and prints every collected Python call under `litellm/`. It never collects Rust spans and never rebuilds the native extension
- E2E and trace strategies load their registered module cases and run surface-specific execution from their folders
- `shared/unit_runners/contracts.py` owns the typed per-function unit contracts consumed by `unit_tests_parity` and `unit_tests_rust`
- `unit_tests_parity/runner.py` runs each contract's `unit_parity_scope` with `LITELLM_RUST=0` and `LITELLM_RUST=1` in separate processes and requires matching outcomes, including failures. Exclusions require a reason in the contract
- `unit_tests_rust/runner.py` runs each contract's focused Cargo test suite. Native Rust unit tests stay beside their implementation
- `shared/unit_runners/suite_runner.py` runs typed suites registered in code with nodeids of the form `suite:<strategy_id>:<function>:<suite>`
- Every strategy declares its report sections and presentation in its own `reporting.py`. Shared reporting code only provides reusable models and cell-formatting primitives
- `shared/` contains reusable parity, tracing, reporting primitives, and unit-runner machinery
- Keep fixtures with their owning API and existing Python tests in their current locations
- Each strategy folder carries an `AGENTS.md` one-liner stating what it should be doing
- Run the harness's own checks with `uv run pytest -o consider_namespace_packages=true tests/rust-python-harness/shared tests/rust-python-harness/cli tests/rust-python-harness/strategies/trace_parity tests/rust-python-harness/strategies/unit_tests_parity tests/rust-python-harness/strategies/unit_tests_rust -q`
