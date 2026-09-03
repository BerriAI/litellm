# Rust/Python migration harness

This local harness follows [the agreed structure](AGENTS.md). The root command selects strategies and combines their reports. Each strategy has an independent entry point

```text
strategies/
  e2e_parity/runner.py
    sdk/ocr/fixtures/
    sdk/messages/
    sdk/chat_completions/
    sdk/responses/
    gateway/
  existing_e2e_test_sdk/runner.py
  trace_parity/runner.py
    sdk/
    gateway/
  unit_tests/
    runner.py
    mapping_validator.py
    python_runner.py
    rust_runner.py
shared/
  parity/
  tracing/
  reporting/
```

## Run locally

```bash
uv run python -m tests.rust-python-harness --list
uv run python -m tests.rust-python-harness --function ocr --plain
uv run python -m tests.rust-python-harness --strategy e2e_parity --surface sdk --function ocr --plain
uv run python -m tests.rust-python-harness.strategies.e2e_parity.runner --function ocr --plain
uv run python -m tests.rust-python-harness.strategies.trace_parity.runner --plain
uv run python -m tests.rust-python-harness.strategies.unit_tests.runner --plain
uv run python -m tests.rust-python-harness.strategies.existing_e2e_test_sdk.runner --function transcription --plain
```

Use `--interactive` for strategy and function selection, `--pytest-arg=-x` to stop pytest on its first failure, and `--coverage` to write Python coverage under `target/rust-python-harness/`. The harness enables pytest namespace-package discovery only for its own invocations

This harness has no CI execution. A configured test that fails or disappears makes the command fail. An unconfigured strategy cell remains planned and contributes no passing evidence. Interruptions and collection errors stop execution; ordinary test failures remain in the combined report while later strategies run

## Strategy responsibilities

E2E parity compares SDK objects, exceptions, callbacks, streams, and provider requests. Gateway tests compare HTTP responses. Both surfaces use the same strategy runner and keep execution details and fixtures in their own folders. OCR has recorded sync/async SDK coverage; the existing Messages and Responses bridge checks remain partial

Trace parity compares operation names through an explicit Python/Rust mapping, call counts, and required completion-before-start ordering with `shared/tracing/compare.py`. Surface tests supply captured operation intervals. No production trace instrumentation or trace case is configured yet

Unit testing combines test mapping validation, separate Python processes with Rust disabled and enabled, backend verification, result comparison, and native Cargo tests. Native tests stay beside their Rust implementation. Existing Python tests stay at their original paths. No complete Python/native unit mapping is configured yet, so these cells remain planned

The existing E2E SDK strategy retains the live provider tests configured upstream. It runs OCR, Chat Completions, and Transcription checks from their existing paths and reports them separately from parity tests. These tests require provider credentials

## Configure cases

Each strategy has a `strategy.json`. Its `functions` object defines SDK cases for OCR, Messages, Responses, Count Tokens, Chat Completions, and Transcription. E2E and trace manifests also accept a `gateway` object keyed by API name. A case has `coverage`, `selectors`, and an optional `note`

```json
{
  "coverage": "partial",
  "selectors": ["tests/rust-python-harness/strategies/e2e_parity/sdk/ocr/test_sdk_parity.py"]
}
```

Selectors use pytest file or node syntax. A selector ending in `/` includes tests recursively from that directory

Use `planned` with no selectors until an executable contract exists, `partial` for incomplete coverage, `complete` for the full contract, and `not_applicable` when a strategy does not apply. The dashboard shows passing evidence separately from coverage completeness and LOC coverage

Unit cases use `unit_suite` instead of `selectors`, pointing to a repository-relative JSON file with this shape:

```json
{
  "python_selectors": ["tests/test_api.py::test_decode"],
  "cargo_manifest": "litellm-rust/Cargo.toml",
  "cargo_package": "litellm-core",
  "cargo_filter": "ocr::",
  "backend": {
    "environment_variable": "LITELLM_USE_RUST_OCR",
    "probe": "tests.rust-python-harness.strategies.unit_tests.python_runner:ocr_backend"
  },
  "mappings": [{"python": "tests/test_api.py::test_decode", "rust": "ocr::test_decode"}]
}
```

Names match automatically when the collected Python and Rust test names agree. Explicit `mappings` handle different names, class names, and parametrized cases. Missing or ambiguous counterparts fail validation in either direction. The Cargo filter must select the same behavior as the Python selectors

The backend probe returns `python` or `rust` and runs at startup and before every test call, after fixtures have run. The OCR probe verifies the dispatch flag and native extension availability. Surface tests must also assert that calls reach their intended implementation to catch per-call fallback. Python outcomes must agree, and failed runs remain failures even if both backends fail identically

## OCR fixtures

Fixtures, provider configuration, input strategies, and recording commands live in [the OCR package](strategies/e2e_parity/sdk/ocr/fixtures/README.md). Record with provider credentials:

```bash
uv run python -m tests.rust-python-harness.strategies.e2e_parity.sdk.ocr.fixtures.record --examples 1000
```

`LITELLM_OCR_FIXTURE_DIR` and `--fixture-dir` override the default directory. Shared recording, replay, comparison, streaming, and cassette persistence live in `shared/parity/`

Run the harness's own checks locally:

```bash
uv run pytest -o consider_namespace_packages=true tests/rust-python-harness/shared tests/rust-python-harness/strategies/unit_tests tests/test_rust_python_harness.py -q
```

Existing OCR parity gaps remain visible: invalid-model provider errors differ, Reducto lacks a native contract, and the expanded Azure corpus exposes duplicate Content-Type headers. Moving the harness does not change provider responses or weaken assertions
