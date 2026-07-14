"""Generic configuration for live e2e tests against a running LiteLLM proxy.

Shared by every e2e suite under tests/e2e/. Values come from the
environment so the same tests run against localhost or a deployed proxy.
"""

import os
import uuid

PROXY_BASE_URL = os.environ.get("LITELLM_PROXY_URL", "http://localhost:4000").rstrip("/")
MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-1234")

# Control-plane (management/admin) base URL. In a split control-plane/data-plane
# deployment the LLM data plane (PROXY_BASE_URL: /chat, /embeddings, native
# passthrough) and the management API (keys, users, teams, orgs, budgets, spend,
# model info, /openapi.json) are served by *different* services. The suite drives
# both through one Transport that routes by path (see transport.SplitTransport).
# Defaults to PROXY_BASE_URL so a monolithic proxy serving everything on one URL
# behaves exactly as before.
CONTROL_PLANE_BASE_URL = os.environ.get(
    "LITELLM_CONTROL_PLANE_URL", PROXY_BASE_URL
).rstrip("/")

UI_USERNAME = os.environ.get("E2E_UI_USERNAME", "admin")
UI_PASSWORD = os.environ.get("E2E_UI_PASSWORD", MASTER_KEY)

CHEAP_ANTHROPIC_MODEL = os.environ.get("E2E_CHEAP_ANTHROPIC_MODEL", "claude-haiku-4-5")

# Jaeger query API of the compose stack's OTEL trace destination (the `jaeger`
# service in docker-compose.yml maps it to host 16686). Trace-completeness tests
# read exported spans back through it.
OTEL_QUERY_URL = os.environ.get("E2E_OTEL_QUERY_URL", "http://localhost:16686").rstrip("/")

# Writes on the proxy are eventually consistent (e.g. spend rows flush on
# proxy_batch_write_at, ~60s). Read-backs poll to this deadline, never sleep-once.
POLL_TIMEOUT = float(os.environ.get("E2E_POLL_TIMEOUT", "120"))
POLL_INTERVAL = float(os.environ.get("E2E_POLL_INTERVAL", "5"))
REQUEST_TIMEOUT = float(os.environ.get("E2E_REQUEST_TIMEOUT", "60"))


def unique_marker() -> str:
    """A short unique token per call/run, so concurrent runs and the shared
    response cache never collide on prompts, tags, or customer ids."""
    return uuid.uuid4().hex[:12]
