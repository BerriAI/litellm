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
│   ├── unit_tests_mapping/
│   │   ├── __init__.py
│   │   ├── contracts.py
│   │   ├── cases/
│   │   │   └── ocr.py
│   │   ├── mapping_report.py
│   │   ├── mappings.py
│   │   ├── mapping_validator.py
│   │   ├── reporting.py
│   │   └── runner.py
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
        └── suite_runner.py
```

- A strategy is a folder under `strategies/` with a one-line `AGENTS.md` and an `__init__.py` exporting exactly one `STRATEGY: StrategyDefinition`; its id must equal the folder name
- `shared/reporting/strategy.py` is the contract: runnable module/suite specs, not-implemented/skipped specs, the runner protocol, and `StrategyDefinition`
- Every `STRATEGY` explicitly classifies every SDK function; surface-aware strategies declare their surfaces and classify the complete surface-by-function matrix
- Run locally only; no CI integration
- `python -m tests.rust-python-harness run <strategy>|all` runs the selected strategy; `--function` is common, while each strategy exposes only its supported options
- Examples: `run e2e_parity --surface sdk --function ocr`, `run unit_tests_parity --function ocr --pytest-arg=-x`, or `run all --function ocr`
- `cli/catalog.py` discovers strategies, validates their Python definitions, and orders them; `cli/__init__.py` builds the Click command tree; `cli/commands.py` runs selected cases
- `e2e_parity/` compares SDK objects, exceptions, callbacks, and streams, or gateway HTTP responses
- `trace_parity/` compares mapped operations, call counts, and required execution ordering; before running it rebuilds the native bridge with the `trace-parity` feature whenever `litellm-rust` sources are newer than the installed extension (`shared/native_build.py`)
- E2E and trace strategies load their registered module cases and run surface-specific execution from their folders
- `unit_tests_mapping/contracts.py` owns typed harness-side mapping contracts, per-function contracts live below `cases/`, and `mappings.py` exports the registry; live test discovery derives unmapped Python and Rust-only tests without an exhaustive manifest
- `unit_tests_mapping/runner.py` validates confirmed mappings against the live Python and Rust inventories and attaches the derived status report
- `unit_tests_parity/runner.py` runs each contract's `unit_parity_scope` with `LITELLM_RUST=0` and `LITELLM_RUST=1` in separate processes and requires matching outcomes, including failures; exclusions require a reason in the contract
- `unit_tests_rust/runner.py` runs each contract's focused Cargo test suite; native Rust unit tests stay beside their implementation
- `shared/unit_runners/suite_runner.py` runs typed suites registered in code with nodeids of the form `suite:<strategy_id>:<function>:<suite>`
- Every strategy declares its report sections and presentation in its own `reporting.py`; shared reporting code only provides reusable models and cell-formatting primitives
- `shared/` contains reusable parity, tracing, reporting primitives, and unit-runner machinery
- Keep fixtures with their owning API and existing Python tests in their current locations
- Each strategy folder carries an `AGENTS.md` one-liner stating what it should be doing
- Run the harness's own checks with `uv run pytest -o consider_namespace_packages=true tests/rust-python-harness/shared tests/rust-python-harness/cli tests/rust-python-harness/strategies/unit_tests_mapping tests/rust-python-harness/strategies/unit_tests_parity tests/rust-python-harness/strategies/unit_tests_rust tests/test_rust_python_harness.py -q`
