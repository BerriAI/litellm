"""Live e2e: Prometheus request metrics grow one series per virtual key.

The proxy exposes ``/metrics`` (prometheus is in the callbacks and
``require_auth_for_metrics_endpoint`` is off in the e2e config). The counter
``litellm_requests_metric_total`` carries an ``api_key_alias`` label, so driving
traffic through keys with distinct aliases must produce a distinct labeled series
per alias. This is the per-key cardinality contract: a regression that stops
stamping ``api_key_alias`` (or collapses every key onto one series) would drop
the aliases and fail here.

Scraping goes through ``transport.probe`` (raw text) and is parsed with
prometheus_client. ``/metrics`` is per-pod behind a round-robin LB and the metric
is eventually consistent (it increments on the success-logging callback), so the
poll re-drives each still-missing alias with fresh traffic and unions the aliases
seen across scrapes until the deadline.
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

        def provisioned_key(alias: str) -> str:
            key = client.key_with_alias(alias, models=[DRIVER_MODEL])
            resources.defer(lambda: client.delete_key(key))
            return key

        def drive(alias: str, key: str) -> None:
            response = client.chat(key, DRIVER_MODEL, f"reply with one word {alias} {unique_marker()}")
            assert response.model, f"driver call for {alias} returned no model: {response}"

        keys_by_alias = {alias: provisioned_key(alias) for alias in aliases}
        for alias, key in keys_by_alias.items():
            drive(alias, key)

        wanted = frozenset(aliases)
        deadline = time.monotonic() + client.proxy.poll_timeout
        seen: frozenset[str] = frozenset()
        while time.monotonic() < deadline:
            seen = seen | _aliases_in_metric(client.scrape_metrics(), REQUESTS_METRIC, ALIAS_LABEL)
            if wanted <= seen:
                break
            for alias in sorted(wanted - seen):
                drive(alias, keys_by_alias[alias])
            time.sleep(client.proxy.poll_interval)

        missing = wanted - seen
        assert not missing, (
            f"{REQUESTS_METRIC} never exposed a per-key series for aliases {sorted(missing)} "
            f"on any scraped pod within the deadline despite repeated driver calls; "
            f"each distinct {ALIAS_LABEL} must grow its own series"
        )
