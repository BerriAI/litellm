"""Generic configuration for live e2e tests against a running LiteLLM proxy.

Shared by every e2e suite under tests/e2e/. Values come from the
environment so the same tests run against localhost or a deployed proxy.
"""

import os
import uuid

PROXY_BASE_URL = os.environ.get("LITELLM_PROXY_URL", "http://localhost:4000").rstrip("/")
MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-1234")

# Control-plane (management/admin) base URL. Defaults to PROXY_BASE_URL so a
# single path-routing host (stage ALB, compose monolith) works for both planes.
# Set LITELLM_CONTROL_PLANE_URL only when management is a different base than
# the LLM host and you are not going through an ingress that path-routes.
CONTROL_PLANE_BASE_URL = os.environ.get(
    "LITELLM_CONTROL_PLANE_URL", PROXY_BASE_URL
).rstrip("/")

UI_USERNAME = os.environ.get("E2E_UI_USERNAME", "admin")
UI_PASSWORD = os.environ.get("E2E_UI_PASSWORD", MASTER_KEY)

# Dashboard base for playwright. Defaults to PROXY_BASE_URL so one ALB/monolith
# host covers /ui as well. Override E2E_UI_BASE_URL only if the UI is elsewhere.
UI_BASE_URL = os.environ.get("E2E_UI_BASE_URL", PROXY_BASE_URL).rstrip("/")

CHEAP_ANTHROPIC_MODEL = os.environ.get("E2E_CHEAP_ANTHROPIC_MODEL", "claude-haiku-4-5")
CHEAP_OPENAI_MODEL = os.environ.get("E2E_CHEAP_OPENAI_MODEL", "gpt-5.5")

# Jaeger query API of the compose stack's OTEL trace destination (the `jaeger`
# service in docker-compose.yml maps it to host 16686). Trace-completeness tests
# read exported spans back through it.
OTEL_QUERY_URL = os.environ.get("E2E_OTEL_QUERY_URL", "http://localhost:16686").rstrip("/")

# Real-DataDog read-back (no local sink - destination fakes cannot be deployed
# on the cluster): the proxy delivers with DD_API_KEY as in production, and the
# tests read ingested events back through the DataDog Logs Search API, which
# additionally needs an application key. On the cluster the secret manager
# injects both; locally tests/e2e/.env provides them.
DD_SITE = os.environ.get("DD_SITE", "datadoghq.com").strip()
DD_API_KEY = os.environ.get("DD_API_KEY", "").strip()
DD_APP_KEY = os.environ.get("DD_APP_KEY", "").strip()
# After the first event is searchable, keep watching this long for a late
# duplicate before the exactly-one assertion: real-DataDog ingestion jitter can
# make one call's two events searchable tens of seconds apart, and a duplicate
# that surfaces late IS the bug (LIT-4447), so one poll interval is not enough.
DD_SETTLE_SECONDS = float(os.environ.get("E2E_DD_SETTLE_SECONDS", "30"))
# DataDog Logs Search `from` window (relative to now). Wide enough for a suite
# run plus ingestion lag; override if a long CI queue needs a wider lookback.
DD_SEARCH_FROM = os.environ.get("E2E_DD_SEARCH_FROM", "now-30m").strip() or "now-30m"
# The Logs Search API budget is tight - 2 requests per 10s org-wide
# (x-ratelimit-name logs_public_search_api) - so read-backs pace their search
# calls at this interval instead of POLL_INTERVAL, and back off when a 429
# still slips through (the budget is shared with anything else searching).
DD_SEARCH_INTERVAL = float(os.environ.get("E2E_DD_SEARCH_INTERVAL", "10"))

# Writes on the proxy are eventually consistent (e.g. spend rows flush on
# proxy_batch_write_at, ~60s). Read-backs poll to this deadline, never sleep-once.
POLL_TIMEOUT = float(os.environ.get("E2E_POLL_TIMEOUT", "120"))
POLL_INTERVAL = float(os.environ.get("E2E_POLL_INTERVAL", "5"))
REQUEST_TIMEOUT = float(os.environ.get("E2E_REQUEST_TIMEOUT", "60"))

# Locust throughput load test (tests/e2e/load). Defaults mirror the concurrency
# and pass criterion the load test is meant to guard; every knob is overridable
# so the same test can be pointed at any deployment (a beefy stage cluster wants
# more users and a higher RPS floor than a laptop compose stack). LOAD_MIN_RPS is
# the SLO the run must clear and LOAD_MAX_FAILURE_RATIO caps the share of requests
# allowed to error before the throughput number is considered meaningless.
LOAD_USERS = int(os.environ.get("E2E_LOAD_USERS", "750"))
LOAD_SPAWN_RATE = float(os.environ.get("E2E_LOAD_SPAWN_RATE", "50"))
LOAD_DURATION_SECONDS = float(os.environ.get("E2E_LOAD_DURATION_SECONDS", "60"))
LOAD_MIN_RPS = float(os.environ.get("E2E_LOAD_MIN_RPS", "355"))
LOAD_MAX_FAILURE_RATIO = float(os.environ.get("E2E_LOAD_MAX_FAILURE_RATIO", "0.01"))


def unique_marker() -> str:
    """A short unique token per call/run, so concurrent runs and the shared
    response cache never collide on prompts, tags, or customer ids."""
    return uuid.uuid4().hex[:12]
