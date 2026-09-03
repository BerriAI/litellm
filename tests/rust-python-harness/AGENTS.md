# Expected Structure

```text
tests/rust-python-harness/
├── __main__.py
│
├── strategies/
│   ├── e2e_parity/
│   │   ├── runner.py
│   │   ├── sdk/
│   │   │   ├── ocr/
│   │   │   ├── messages/
│   │   │   ├── chat_completions/
│   │   │   └── responses/
│   │   └── gateway/
│   │
│   ├── trace_parity/
│   │   ├── runner.py
│   │   ├── sdk/
│   │   └── gateway/
│   │
│   ├── unit_tests_mapping/
│   │   ├── runner.py
│   │   └── mapping_validator.py
│   │
│   ├── unit_tests_parity/
│   │   └── runner.py
│   │
│   └── unit_tests_rust/
│       └── runner.py
│
└── shared/
    ├── parity/
    ├── tracing/
    ├── reporting/
    └── unit_runners/
```

- Run locally only; no CI integration
- `__main__.py` selects strategies and combines their reports; each strategy also runs independently
- `e2e_parity/` compares SDK objects, exceptions, callbacks, and streams, or gateway HTTP responses
- `trace_parity/` compares mapped operations, call counts, and required execution ordering
- E2E and trace runners share orchestration across `sdk/` and `gateway/`; surface-specific execution lives in those folders
- `unit_tests_mapping/runner.py` validates Python/Rust test mappings against collected test inventories without running the selected tests
- `unit_tests_mapping/mapping_validator.py` matches Python/Rust tests by agreed names or annotations and reports missing or ambiguous counterparts
- `unit_tests_parity/runner.py` runs existing Python unit tests with `LITELLM_RUST=0` and `LITELLM_RUST=1` in separate processes and requires matching outcomes, including failures
- `unit_tests_rust/runner.py` runs Cargo tests; native Rust unit tests stay beside their implementation
- `shared/` contains reusable parity, tracing, reporting, and unit-runner machinery
- Keep fixtures with their owning API and existing Python tests in their current locations
- Each strategy folder carries an `AGENTS.md` one-liner stating what it should be doing
