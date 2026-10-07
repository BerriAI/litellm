This directory holds only the tests that cannot be written in the Rust code

- Native route contracts against local peers: inference, Messages, OCR, secrets, cache, trace storage, fork guard, tiktoken encodings
- Every file needs the compiled `litellm.rust_bridge._native` extension: the conftest imports it at collection, so a missing extension is a collection error, not a skip
- Rollout is enabled by the test itself (monkeypatch or test-owned rules), never by the job env
- Helpers live in `tests/_support` and `tests/integration/_support/native`
- These files are GitHub-owned via `GITHUB_FILES` in `tests/integration/run.py` and run only through `make test-rust-extension` against the installed wheel
- Trace query API contracts live in `tests/integration/observability/test_trace_query_api.py`
- Pure in-process binding tests go to `tests/unit`, under `rust_bridge` or the production module's path
