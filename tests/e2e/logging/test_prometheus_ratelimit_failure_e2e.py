"""Live e2e: a rate-limit rejection is attributed to the gateway, not the vendor.

Operators page on 429s, and the response to one depends entirely on who produced
it: a vendor 429 means back off or fail over, while a gateway 429 means the
customer's own key limit is set too low. `litellm_proxy_failed_requests_metric`
has to make that difference readable, and it has to name the provider, otherwise
a multi-provider deployment cannot tell which upstream is saturating (PR #27687).

The two are distinguished by `exception_class`: the gateway's own limiter raises
`HTTPException`, while a vendor rejection surfaces its provider error class (for
example `Openai.RateLimitError`). This drives a key over its own RPM limit, which
is unambiguously a gateway-side rejection, and requires the resulting series to
be labelled that way and to carry a provider.

The metric is written on the failure-logging callback, so the scrape polls to a
deadline rather than sleeping once.
"""

from __future__ import annotations

import time

import pytest
from prometheus_client.parser import text_string_to_metric_families

from e2e_config import unique_marker
from lifecycle import ResourceManager
from logging_client import LoggingClient
from models import KeyGenerateBody

pytestmark = pytest.mark.e2e

DRIVER_MODEL = "gemini-2.5-flash"
FAILURE_METRIC = "litellm_proxy_failed_requests_metric_total"
GATEWAY_EXCEPTION_CLASS = "HTTPException"
RPM_LIMIT = 1
CALLS = 3


def _rate_limited_series(exposition: str, alias: str) -> tuple[dict[str, str], ...]:
    """Every 429 sample on the failure metric belonging to our key."""
    return tuple(
        sample.labels
        for family in text_string_to_metric_families(exposition)
        for sample in family.samples
        if sample.name == FAILURE_METRIC
        and sample.labels.get("api_key_alias") == alias
        and sample.labels.get("exception_status") == "429"
    )


class TestPrometheusRateLimitAttribution:
    @pytest.mark.covers("logging.prometheus.failure.attributes_rate_limit_source")
    def test_gateway_rate_limit_is_labelled_with_source_and_provider(
        self, client: LoggingClient, resources: ResourceManager
    ) -> None:
        alias = f"e2e-rl-attr-{unique_marker()}"
        key = client.proxy.generate_key(
            KeyGenerateBody(
                key_alias=alias,
                models=[DRIVER_MODEL],
                user_id=f"e2e-{alias}",
                rpm_limit=RPM_LIMIT,
            )
        )
        resources.defer(lambda: client.delete_key(key))

        statuses = tuple(
            client.chat_raw(key, DRIVER_MODEL, f"say ok {unique_marker()}").status_code
            for _ in range(CALLS)
        )
        assert 429 in statuses, (
            f"driving {CALLS} calls against an rpm_limit of {RPM_LIMIT} produced no 429; "
            f"statuses were {statuses}, so the limiter never rejected and there is "
            f"nothing for the metric to attribute"
        )

        deadline = time.monotonic() + client.proxy.poll_timeout
        series: tuple[dict[str, str], ...] = ()
        while time.monotonic() < deadline:
            series = _rate_limited_series(client.scrape_metrics(), alias)
            if series:
                break
            time.sleep(client.proxy.poll_interval)

        assert series, (
            f"{FAILURE_METRIC} has no 429 series for api_key_alias={alias}; a rate-limit "
            f"rejection that is never counted cannot be alerted on"
        )

        classes = {labels.get("exception_class") for labels in series}
        assert classes == {GATEWAY_EXCEPTION_CLASS}, (
            f"a gateway rpm rejection must be attributed to the gateway as "
            f"{GATEWAY_EXCEPTION_CLASS}, got {sorted(c or '' for c in classes)}; "
            f"a vendor error class here would send operators chasing the upstream"
        )

        providers = {labels.get("api_provider") for labels in series}
        assert providers and all(providers), (
            f"{FAILURE_METRIC} 429 series carries an empty api_provider ({providers}); "
            f"a multi-provider deployment cannot tell which upstream the rejection was for"
        )
