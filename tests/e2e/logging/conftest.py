"""Fixtures for the logging e2e suite.

The Datadog tests exercise the real delivery path: the proxy ships each request's
StandardLoggingPayload to Datadog on its ``datadog`` callback, and the tests read
those events back out of the Datadog Logs Search API to prove they landed. That
read-back needs all three credentials - ``DD_API_KEY`` and ``DD_SITE`` (which the
proxy also ships with) plus ``DD_APP_KEY`` (the Logs Search API rejects reads that
carry only an API key) - so the suite skips when any is absent.

The shared ``resources`` / ``scoped_key`` fixtures come from the root e2e conftest
via the GatewayProvider protocol (LoggingClient exposes ``.gateway``).
"""

import pytest

from logging_client import DatadogClient, LoggingClient, build_logging_client


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "covers: registry cell a test covers, e.g. logging.datadog.success.exports_metric",
    )


@pytest.fixture(scope="session")
def client() -> LoggingClient:
    """The logging suite's client: holds the shared Gateway (so `resources` /
    `scoped_key` clean up keys), the Datadog read-back client, and `/metrics`
    scraping."""
    return build_logging_client()


@pytest.fixture(scope="session")
def datadog(client: LoggingClient) -> DatadogClient:
    """The Datadog read-back client, or skip when the credentials are absent."""
    if client.datadog is None:
        pytest.skip("set DD_API_KEY, DD_SITE and DD_APP_KEY to run the Datadog logging suite")
    return client.datadog
