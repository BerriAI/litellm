# Proxy CLI integration tests

These tests invoke real binaries and subprocesses (for example the installed
`litellm-proxy` console script) rather than mocks. They live here, outside
`tests/test_litellm/`, because that tree is reserved for mock-only unit tests
that must run deterministically in CI without external binaries.

## Running

They are skipped by default. Opt in with an environment variable:

```bash
LITELLM_RUN_INTEGRATION_TESTS=1 pytest tests/proxy_cli_integration_tests/
```

You also need the package installed so the console scripts exist on PATH:

```bash
pip install -e '.[proxy]'
```

Individual tests additionally `pytest.skip` themselves when a required binary
(such as `litellm-proxy`) is not present in the active environment, so a partial
install degrades to a skip rather than a failure.

## Markers

Tests here are tagged with the `integration` marker (registered in
`pyproject.toml`). To run only integration-marked tests:

```bash
LITELLM_RUN_INTEGRATION_TESTS=1 pytest -m integration
```
