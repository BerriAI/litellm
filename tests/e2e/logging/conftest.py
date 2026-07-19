"""Fixtures for the logging e2e suite.

Missing proxy, provider keys, or integration credentials are hard failures.
Never pytest.skip from this suite for environment gaps.
"""

from __future__ import annotations

import os

import pytest

from logging_client import LangfuseCreds, LoggingClient, build_logging_client, load_langfuse_creds
from datadog_reader import DdLogsReader, build_dd_logs_reader
from models import LiteLLMParamsBody
from otel_client import OtelReader, build_otel_reader
from proxy_client import ProxyClient

# Static name used by test_langfuse_e2e.py team/key/org cases.
_LANGFUSE_DRIVER_MODEL = "e2e-langfuse-haiku"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. logging.langfuse.success.logs_spend",
    )


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> LoggingClient:
    """The logging suite's client: holds the shared ProxyClient so `resources` /
    `scoped_key` clean up keys and teams, and adds `/metrics` scraping plus
    Langfuse read-back."""
    # Register a cheap Anthropic driver so langfuse cases do not depend on a
    # static gateway model_list entry or a flaky Gemini quota.
    proxy.create_model(
        _LANGFUSE_DRIVER_MODEL,
        LiteLLMParamsBody(
            model="anthropic/claude-haiku-4-5-20251001",
            api_key="os.environ/ANTHROPIC_API_KEY",
        ),
    )
    return build_logging_client(proxy)


@pytest.fixture(scope="session")
def otel_reader() -> OtelReader:
    """Read-back client for the compose stack's Jaeger trace destination."""
    return build_otel_reader()


@pytest.fixture(scope="session")
def dd_logs() -> DdLogsReader:
    """Read-back client for the real DataDog Logs Search API (keys from the
    secret manager on the cluster, tests/e2e/.env locally)."""
    return build_dd_logs_reader()


@pytest.fixture
def datadog_creds() -> None:
    """Require Datadog shipping credentials. Hard-fail when absent; never skip."""
    if not (os.getenv("DD_API_KEY") and os.getenv("DD_SITE")):
        pytest.fail(
            "Datadog e2e requires DD_API_KEY and DD_SITE; missing credentials is a hard failure, not a skip"
        )


@pytest.fixture(scope="session")
def langfuse_creds() -> LangfuseCreds:
    """Require real Langfuse cloud credentials for team callback + trace poll."""
    return load_langfuse_creds()
