# Migrating HTTP clients to httpx2

LiteLLM now uses `httpx2` for HTTP clients, transports, responses, timeouts and exceptions. Python 3.10 remains supported. Installers select a compatible `httpx2` release for that interpreter

Applications that pass custom HTTP clients or timeout objects must construct them with `httpx2`. Float timeouts and the existing LiteLLM HTTP handler names are unchanged

```python
import httpx2
import litellm

with httpx2.Client(timeout=httpx2.Timeout(60, connect=5)) as client:
    litellm.client_session = client
```

Use `httpx2.AsyncClient` for `litellm.aclient_session`. Catch `httpx2` exceptions when working directly with LiteLLM's HTTP handlers. Provider calls continue to expose LiteLLM's public exception types

The base SDK can be installed without legacy `httpx` when using OpenAI 3

```bash
python -m pip install litellm 'openai>=3,<4'
```

OpenAI 2.47 and newer remain supported for integrations such as the optional semantic router that still require OpenAI 2. Those installations retain OpenAI's legacy `httpx` dependency. Optional packages such as Prisma, Langfuse and the A2A SDK can also bring legacy `httpx` into a proxy installation

## HTTP mocks

Tests use `pytest-httpx2` to intercept `httpcore2` through RESPX. The shared `respx_mock` fixture and `tests._support.respx_httpx2` namespace select the backend for both HTTP client families. Explicit RESPX contexts and decorators must use `using="httpcore2_and_legacy"`

RESPX still requires its own legacy response objects. Use `response_for_respx` for a route's return value and `side_effect_for_respx` for its callback or response sequence. These test-only adapters preserve streaming and cleanup without changing runtime module imports
