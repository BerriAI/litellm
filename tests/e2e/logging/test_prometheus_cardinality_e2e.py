"""Live e2e: Prometheus request metrics grow one series per virtual key.

The proxy exposes ``/metrics`` (prometheus is in the callbacks and
``require_auth_for_metrics_endpoint`` is off in the e2e config). The counter
``litellm_requests_metric_total`` carries an ``api_key_alias`` label, so driving
traffic through keys with distinct aliases must produce a distinct labeled series
per alias. This is the per-key cardinality contract: a regression that stops
stamping ``api_key_alias`` (or collapses every key onto one series) would drop
the aliases and fail here.

Scraping goes through ``transport.probe`` (raw text) and is parsed with
prometheus_client; the metric is eventually consistent (it increments on the
success-logging callback), so the scrape polls to a deadline.
"""

from __future__ import annotations

import time

import pytest
from prometheus_client.parser import text_string_to_metric_families

from e2e_config import unique_marker
from lifecycle import ResourceManager
from logging_client import LoggingClient

pytestmark = pytest.mark.e2e

DRIVER_MODEL = "gemini-2.5-flash"
REQUESTS_METRIC = "litellm_requests_metric_total"
ALIAS_LABEL = "api_key_alias"
DISTINCT_KEYS = 3


def _aliases_in_metric(exposition: str, metric: str, label: str) -> frozenset[str]:
    """The set of ``label`` values present on ``metric`` samples in a scrape."""
    return frozenset(
        sample.labels[label]
        for family in text_string_to_metric_families(exposition)
        for sample in family.samples
        if sample.name == metric and label in sample.labels
    )


class TestPrometheusPerKeyCardinality:
    @pytest.mark.covers("logging.prometheus.success.exports_metric", exercised_on=[])
    def test_distinct_key_aliases_produce_distinct_series(
        self, client: LoggingClient, resources: ResourceManager
    ) -> None:
        aliases = tuple(f"e2e-prom-{unique_marker()}" for _ in range(DISTINCT_KEYS))
        for alias in aliases:
            key = client.key_with_alias(alias, models=[DRIVER_MODEL])
            resources.defer(lambda k=key: client.delete_key(k))
            response = client.chat(key, DRIVER_MODEL, f"reply with one word {alias}")
            assert response.model, f"driver call for {alias} returned no model: {response}"

        wanted = frozenset(aliases)
        deadline = time.monotonic() + client.proxy.poll_timeout
        seen: frozenset[str] = frozenset()
        while time.monotonic() < deadline:
            seen = _aliases_in_metric(client.scrape_metrics(), REQUESTS_METRIC, ALIAS_LABEL)
            if wanted <= seen:
                break
            time.sleep(client.proxy.poll_interval)

        missing = wanted - seen
        assert not missing, (
            f"{REQUESTS_METRIC} is missing a per-key series for aliases {sorted(missing)}; "
            f"each distinct {ALIAS_LABEL} must grow its own series"
        )
