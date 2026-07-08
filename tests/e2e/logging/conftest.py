"""Fixtures for the Datadog logging suite.

These tests drive the Datadog batch-send path (#25663) directly against the real
Datadog logs intake with synthetic events - no LLM calls, no proxy, no log
read-back - so they need only the shipping credentials DD_API_KEY + DD_SITE
(DD_SERVICE is an optional tag). No Datadog Application key is required, and they
skip when the shipping credentials are absent from the environment.
"""

import os

import pytest

from logging_client import LoggingClient, build_logging_client


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. logging.datadog.success.writes_object",
    )


@pytest.fixture(scope="session")
def client() -> LoggingClient:
    """The logging suite's client: holds the shared Gateway so `resources` /
    `scoped_key` clean up keys, and adds `/metrics` scraping."""
    return build_logging_client()


@pytest.fixture
def datadog_creds() -> None:
    """Gate the suite on the Datadog shipping credentials. The DataDogLogger is built
    inside each async test, not here, because its __init__ schedules a periodic-flush
    task via asyncio.create_task and so needs a running event loop."""
    if not (os.getenv("DD_API_KEY") and os.getenv("DD_SITE")):
        pytest.skip("set DD_API_KEY and DD_SITE to run the Datadog logging suite")
