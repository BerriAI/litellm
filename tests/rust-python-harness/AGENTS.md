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
│   └── unit_tests/
│       ├── runner.py
│       ├── mapping_validator.py
│       ├── python_runner.py
│       └── rust_runner.py
│
└── shared/
    ├── parity/
    ├── tracing/
    └── reporting/
```

- Run locally only; no CI integration
- `__main__.py` selects strategies and combines their reports; each strategy also runs independently
- `e2e_parity/` compares SDK objects, exceptions, callbacks, and streams, or gateway HTTP responses
- `trace_parity/` compares mapped operations, call counts, and required execution ordering
- E2E and trace runners share orchestration across `sdk/` and `gateway/`; surface-specific execution lives in those folders
- `unit_tests/runner.py` combines mapping validation, Python test runs, and native Rust test runs
- `mapping_validator.py` matches Python/Rust tests by agreed names or annotations and reports missing or ambiguous counterparts
- `python_runner.py` runs existing Python tests with Rust disabled and enabled in separate processes, verifies backend selection, and compares results
- `rust_runner.py` runs Cargo tests; native Rust unit tests stay beside their implementation
- `shared/` contains reusable parity, tracing, and reporting machinery
- Keep fixtures with their owning API and existing Python tests in their current locations
