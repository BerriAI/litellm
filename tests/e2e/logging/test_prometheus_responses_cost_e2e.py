"""The cost a logging callback sees must be the cost the spend row bills.

GitHub issue #41976: a successful sync ``/v1/responses`` call reached callbacks with
``StandardLoggingPayload.response_cost == 0`` while ``LiteLLM_SpendLogs.spend`` for the
same request id was positive. Prometheus is the callback under test here because its
``litellm_spend_metric`` counter is fed straight from ``standard_logging_object.response_cost``
and is readable from the proxy itself, so no external sink is needed. A fresh key with one
request makes the per-alias counter equal that request's callback-visible cost.
"""

from __future__ import annotations

import json
import math
import time

import pytest
from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from lifecycle import ResourceManager
from logging_client import LoggingClient, first_ok
from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel, ConfigDict

pytestmark = pytest.mark.e2e

SPEND_METRIC = "litellm_spend_metric"
ALIAS_LABEL = "api_key_alias"


class _ResponsesBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str


def _callback_spend(exposition: str, alias: str) -> float | None:
    values = [
        sample.value
        for family in text_string_to_metric_families(exposition)
        for sample in family.samples
        if sample.name == f"{SPEND_METRIC}_total" and sample.labels.get(ALIAS_LABEL) == alias
    ]
    return sum(values) if values else None


def _poll_callback_spend(client: LoggingClient, alias: str) -> float | None:
    deadline = time.monotonic() + client.proxy.poll_timeout
    while True:
        spend = _callback_spend(client.scrape_metrics(), alias)
        if spend is not None or time.monotonic() >= deadline:
            return spend
        time.sleep(client.proxy.poll_interval)


class TestResponsesCallbackCost:
    @pytest.mark.covers("logging.prometheus.success.logs_spend", exercised_on=["responses"])
    def test_responses_callback_cost_equals_spend_log_row(
        self, client: LoggingClient, resources: ResourceManager
    ) -> None:
        alias = f"e2e-responses-cost-{unique_marker()}"
        key = client.key_with_alias(alias, models=[CHEAP_OPENAI_MODEL])
        resources.defer(lambda: client.delete_key(key))

        outcome = first_ok(
            client,
            lambda: client.responses_raw(key, CHEAP_OPENAI_MODEL, f"reply with one word {unique_marker()}"),
        )
        response_id = _ResponsesBody.model_validate(json.loads(outcome.body)).id
        assert response_id, f"/v1/responses returned no id: {outcome.body[:200]}"

        spend_row = client.poll_proxy_spend_for_key(key, response_id=response_id)
        assert spend_row is not None and spend_row.spend is not None and spend_row.spend > 0, (
            f"no positive-spend row for request_id={response_id}, got {spend_row!r}"
        )

        callback_spend = _poll_callback_spend(client, alias)
        assert callback_spend is not None, (
            f"{SPEND_METRIC} has no series for {ALIAS_LABEL}={alias} although the spend row billed {spend_row.spend}"
        )
        assert math.isclose(callback_spend, spend_row.spend, rel_tol=1e-9), (
            f"callback saw response_cost={callback_spend} but SpendLogs billed {spend_row.spend} "
            f"for request_id={response_id}"
        )
