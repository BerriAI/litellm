"""Fixtures for the logging e2e suite.

Missing proxy, provider keys, or integration credentials are hard failures.
Never pytest.skip from this suite for environment gaps.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from logging_client import (
    LangfuseCreds,
    LoggingClient,
    PhoenixCreds,
    build_logging_client,
    load_langfuse_creds,
    load_phoenix_creds,
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. logging.langfuse.success.logs_spend",
    )


@pytest.fixture(scope="session")
def client() -> Iterator[LoggingClient]:
    """The logging suite's client: holds the shared Gateway so `resources` /
    `scoped_key` clean up keys and teams, and adds `/metrics` scraping plus
    Langfuse and Phoenix read-back. The models registered at build time are
    queued for deletion and removed at session end."""
    logging_client, model_cleanup = build_logging_client()
    yield logging_client
    model_cleanup.teardown()


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


@pytest.fixture(scope="session")
def phoenix_creds() -> PhoenixCreds:
    """Arize Phoenix read-back target: the local compose `phoenix` service by
    default, or PHOENIX_BASE_URL / PHOENIX_PROJECT_NAME / PHOENIX_API_KEY for a
    deployed instance. An unreachable Phoenix is a hard failure at poll time."""
    return load_phoenix_creds()
