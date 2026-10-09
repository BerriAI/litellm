# e2e harness tests

Tests of the harness under `tests/e2e/` (the transport, the clients, the fixture bundle and replay edge, the stack lock, the IdP launcher, the coverage collector, the JUnit properties, the load aggregation helpers and the Claude Code driver), not of the product. They live outside `tests/e2e/` because the Buildkite e2e run copies that folder into the runner image and runs every test in it, so a harness test in there counts as a product test in the nightly numbers. Nothing here needs a proxy, provider keys or the network

The layout mirrors `tests/e2e/`: `test_e2e_http.py` covers `tests/e2e/e2e_http.py`, `logging/test_datadog_reader.py` covers `tests/e2e/logging/datadog_reader.py`, and `claude_code/` covers the driver, builder, probe and version resolver. Put a new harness test under the folder that mirrors the suite folder whose module it covers

Run them from the repo root. `e2e_config` reads `LITELLM_MASTER_KEY` at import and any value will do, the CI lane sets a dummy:

```bash
LITELLM_MASTER_KEY=sk-harness uv run pytest tests/e2e_harness
```

`pytest.ini` here puts `tests/e2e` and the suite folders whose modules are under test on the path, so imports look exactly as they do inside the suite (`from e2e_http import ...`, `from batch_cleanup import ...`). `claude_code/test_request_determinism.py` drives the real `claude` CLI. Deselect it with `-m "not cli_determinism"` when the CLI is not installed

Rules: no `e2e` marker and no `@meta`, since nothing here drives the proxy. `@pytest.mark.covers` only where the test proves the collector or the JUnit properties read it. Inputs via arguments or env vars (setting an env var through pytest's `monkeypatch` fixture is fine, patching a function, class or module is not). And the same typing bar as the suite, `make lint-e2e-basedpyright` covers this folder and allows zero errors. The raw HTTP client ban (`tests/code_coverage_tests/check_e2e_no_raw_requests.py`) applies here too

CI: the `python` job in `.github/workflows/test-linting.yml` runs this folder whenever anything under `tests/e2e/` (except `ui/`) or `tests/e2e_harness/` changes, with the `claude` CLI installed. The CircleCI `provider_replay_harness` job also runs the provider-edge and fixture tests at the root of this folder next to `tests/code_coverage_tests/test_provider_replay_harness.py`, which imports helpers from `test_provider_edge.py`
