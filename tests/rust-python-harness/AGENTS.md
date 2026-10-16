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
│   ├── drift/
│   │   ├── __init__.py
│   │   ├── AGENTS.md
│   │   ├── command.py
│   │   ├── drift.json
│   │   └── runner.py
│   │
│   ├── e2e_parity/
│   │   ├── __init__.py
│   │   ├── reporting.py
│   │   ├── sdk/
│   │   │   └── ocr/
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

- Every strategy is a folder under `strategies/` with an `AGENTS.md` and an `__init__.py` exporting exactly one `STRATEGY`. Its id must equal the folder name. SDK strategies export `StrategyDefinition`; the repository-source drift strategy exports `RepositoryStrategyDefinition`
- `shared/reporting/strategy.py` is the contract: runnable module/suite specs, not-implemented/skipped specs, the runner protocol, and `StrategyDefinition`
- Every SDK `StrategyDefinition` explicitly classifies every SDK function. Surface-aware SDK strategies declare their surfaces and classify the complete surface-by-function matrix. Repository strategies use their own options rather than SDK `--function` or `--surface` selection
- Run locally only. No CI integration
- `uv run python -m tests.rust-python-harness run drift --target HEAD` inventories committed Python and Rust changes using `strategies/drift/drift.json`. See `strategies/drift/AGENTS.md` for usage, exploratory `--since` comparisons, and manual snapshot checkpoints with independent unknown, drifted, or aligned states
- `python -m tests.rust-python-harness run <strategy>` runs the selected strategy with its supported options. SDK strategies share `--function`. `run all` runs SDK execution tests; invoke `run drift` explicitly for the repository change inventory
- Examples: `run e2e_parity --surface sdk --function ocr`, `run unit_tests_parity --function ocr --pytest-arg=-x`, or `run all --function ocr`
- `cli/catalog.py` discovers strategies, validates their Python definitions, and orders them. `cli/__init__.py` builds the Click command tree. `cli/commands.py` runs selected cases
- `e2e_parity/` compares SDK objects, exceptions, callbacks, and streams, or gateway HTTP responses
- `trace_parity/` profiles the Python call stack and prints every collected Python call under `litellm/`. It never collects Rust spans and never rebuilds the native extension
- E2E and trace strategies load their registered module cases and run surface-specific execution from their folders
- `shared/unit_runners/contracts.py` owns the typed per-function unit contracts consumed by `unit_tests_parity` and `unit_tests_rust`
- `unit_tests_parity/runner.py` runs each contract's `unit_parity_scope` with `LITELLM_RUST=0` and `LITELLM_RUST=1` in separate processes and requires matching outcomes, including failures. Exclusions require a reason in the contract
- `unit_tests_rust/runner.py` runs each contract's focused Cargo test suite. Native Rust unit tests stay beside their implementation
- `shared/unit_runners/suite_runner.py` runs typed suites registered in code with nodeids of the form `suite:<strategy_id>:<function>:<suite>`
- Every SDK strategy declares its report sections and presentation in its own `reporting.py`. Shared reporting code only provides reusable models and cell-formatting primitives. Drift emits its repository change report directly
- `shared/` contains reusable parity, tracing, reporting primitives, and unit-runner machinery
- Keep fixtures with their owning API and existing Python tests in their current locations
- Each test strategy folder carries an `AGENTS.md` one-liner stating what it should be doing. The repository-source drift strategy keeps its usage guide in `strategies/drift/AGENTS.md`
- Run the harness's own checks with `uv run pytest -o consider_namespace_packages=true tests/rust-python-harness/shared tests/rust-python-harness/cli tests/rust-python-harness/strategies/drift tests/rust-python-harness/strategies/trace_parity tests/rust-python-harness/strategies/unit_tests_parity tests/rust-python-harness/strategies/unit_tests_rust -q`
