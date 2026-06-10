# Mock LLM Endpoints (for local dev + CI)

This directory contains small, self-contained mock servers used by tests to
avoid hitting real provider APIs (and to avoid relying on third-party
hosting services like Railway).

## `example_openai_endpoint/`

A vendored copy of [BerriAI/example_openai_endpoint](https://github.com/BerriAI/example_openai_endpoint),
a FastAPI app that implements stub OpenAI- / Anthropic- / Vertex- / Bedrock-
compatible endpoints (chat completions, embeddings, audio, batches, etc.).
Many tests across the repo reference its public deployment at
`https://exampleopenaiendpoint-production.up.railway.app/`.

The Railway deployment has had multiple outages
(see <https://status.railway.com/historical>) that break CI. The goal of
vendoring it here is to make the same stub server runnable in the same
container as CI (and locally on a developer machine), so tests do not depend
on any external service being up.

### Running locally

The easiest way to start the server (installs `slowapi` + friends into the
project venv first, then runs the server on `:8090`):

```bash
make mock-server          # foreground
PORT=18090 make mock-server   # alt port
```

If you don't want to use the Makefile, you can run the server directly with
any Python interpreter that has the deps installed:

```bash
python -m pip install -r tests/mock_endpoints/example_openai_endpoint/requirements.txt
python tests/mock_endpoints/example_openai_endpoint/main.py
```

Or use the bash helper (handy for CI / shell scripts — it starts the server
in the background, polls until it's ready, and writes logs to `/tmp`):

```bash
./tests/mock_endpoints/start_mock_server.sh                # foreground
./tests/mock_endpoints/start_mock_server.sh --background   # background
```

### Using it from a pytest suite

The recommended pattern is the session-scoped fixture defined in
[`tests/mock_endpoints/conftest.py`](./conftest.py). Any test file (or a
suite-level `conftest.py`) can opt in like so:

```python
import pytest

pytest_plugins = ("tests.mock_endpoints.conftest",)


def test_something(mock_openai_endpoint_server):
    base_url = mock_openai_endpoint_server  # e.g. "http://127.0.0.1:53892"
    ...
```

The fixture:

- Picks a free port automatically (so suites can run in parallel without
  colliding).
- Boots the vendored server as a subprocess.
- Waits until `/chat/completions` returns 200 before yielding the URL.
- Exposes the URL as `LITELLM_MOCK_OPENAI_BASE_URL` in the environment so
  any code that calls `tests.mock_endpoints.MOCK_OPENAI_BASE_URL` picks it
  up automatically.
- Tears the subprocess down at session end.

See [`tests/test_litellm/mock_endpoints/test_mock_openai_endpoint_server.py`](../test_litellm/mock_endpoints/test_mock_openai_endpoint_server.py)
for a working end-to-end example.

### Quick sanity check

```bash
curl -s http://127.0.0.1:8090/chat/completions \
  -H 'Authorization: Bearer sk-test' \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}'
```

### Pointing tests at the local server

Tests that currently hard-code
`https://exampleopenaiendpoint-production.up.railway.app/` can be pointed at
the local mock by changing the URL to `http://127.0.0.1:8090/`. New tests
should prefer this local mock over the Railway deployment.

### Keeping the vendored copy in sync

The upstream source of truth is still
<https://github.com/BerriAI/example_openai_endpoint>. To pull the latest
version into this repo:

```bash
curl -fsSL -o tests/mock_endpoints/example_openai_endpoint/main.py \
  https://raw.githubusercontent.com/BerriAI/example_openai_endpoint/main/main.py
curl -fsSL -o tests/mock_endpoints/example_openai_endpoint/batch_and_files_api.py \
  https://raw.githubusercontent.com/BerriAI/example_openai_endpoint/main/batch_and_files_api.py
curl -fsSL -o tests/mock_endpoints/example_openai_endpoint/requirements.txt \
  https://raw.githubusercontent.com/BerriAI/example_openai_endpoint/main/requirements.txt
curl -fsSL -o tests/mock_endpoints/example_openai_endpoint/Dockerfile \
  https://raw.githubusercontent.com/BerriAI/example_openai_endpoint/main/Dockerfile
```

These files should be copied **as-is** from upstream so the two repos stay
in sync; do not edit them in place. If an endpoint needs to change, change
it upstream first, then re-vendor.
