from __future__ import annotations

import time

import pytest
from prometheus_client.parser import text_string_to_metric_families

from e2e_config import unique_marker
from lifecycle import ResourceManager
from logging_client import LoggingClient

pytestmark = pytest.mark.e2e

DRIVER_MODEL = "gemini-2.5-flash"
QUEUE_TIME_METRIC = "litellm_request_queue_time_seconds"
ALIAS_LABEL = "api_key_alias"


def _observation_count(exposition: str, alias: str) -> float | None:
    return next(
        (
            sample.value
            for family in text_string_to_metric_families(exposition)
            for sample in family.samples
            if sample.name == f"{QUEUE_TIME_METRIC}_count" and sample.labels.get(ALIAS_LABEL) == alias
        ),
        None,
    )


class TestPrometheusRequestQueueTime:
    @pytest.mark.covers("logging.prometheus.success.records_queue_time")
    def test_queue_time_histogram_records_an_observation(
        self, client: LoggingClient, resources: ResourceManager
    ) -> None:
        alias = f"e2e-queue-time-{unique_marker()}"
        key = client.key_with_alias(alias, models=[DRIVER_MODEL])
        resources.defer(lambda: client.delete_key(key))

        response = client.chat(key, DRIVER_MODEL, f"reply with one word {alias}")
        assert response.model, f"driver call returned no model: {response}"

        deadline = time.monotonic() + client.proxy.poll_timeout
        count: float | None = None
        while time.monotonic() < deadline:
            count = _observation_count(client.scrape_metrics(), alias)
            if count is not None and count > 0:
                break
            time.sleep(client.proxy.poll_interval)

        assert count is not None, (
            f"{QUEUE_TIME_METRIC} has no series for {ALIAS_LABEL}={alias}; the histogram was "
            f"never observed for a request that succeeded"
        )
        assert count > 0, (
            f"{QUEUE_TIME_METRIC} series for {alias} exists but recorded {count} observations; "
            f"the metric is registered yet never written"
        )
