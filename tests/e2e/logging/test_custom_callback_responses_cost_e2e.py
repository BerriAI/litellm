"""The response_cost a config-registered CustomLogger receives must be the cost the spend row bills.

GitHub issue #41976: a successful sync, non-streaming ``/v1/responses`` call reached callbacks with
``kwargs["standard_logging_object"]["response_cost"] == 0`` while ``LiteLLM_SpendLogs.spend`` for the
same request id was positive. The gateway registers ``callback_cost_capture.capture`` from this
directory as a plain ``callbacks:`` entry, so this is the exact payload a Langfuse, Datadog, or
customer CustomLogger is handed. The capture file path comes from ``E2E_CALLBACK_COST_CAPTURE_FILE``.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest
from callback_cost_capture import CAPTURE_FILE_ENV, CapturedCost
from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from lifecycle import ResourceManager
from logging_client import LoggingClient, first_ok
from pydantic import BaseModel, ConfigDict

pytestmark = pytest.mark.e2e


class _ResponsesBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str


def _capture_file() -> Path:
    path = os.environ.get(CAPTURE_FILE_ENV)
    assert path, f"{CAPTURE_FILE_ENV} is unset: the gateway must register callback_cost_capture.capture"
    return Path(path)


def _captured_rows(path: Path, request_id: str) -> list[CapturedCost]:
    if not path.exists():
        return []
    rows = [CapturedCost.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row.request_id == request_id]


def _poll_captured(client: LoggingClient, path: Path, request_id: str) -> list[CapturedCost]:
    deadline = time.monotonic() + client.proxy.poll_timeout
    while True:
        rows = _captured_rows(path, request_id)
        if rows or time.monotonic() >= deadline:
            return rows
        time.sleep(client.proxy.poll_interval)


class TestCustomCallbackResponsesCost:
    @pytest.mark.covers("logging.custom_callback.success.logs_spend", exercised_on=["responses"])
    def test_sync_responses_callback_payload_cost_equals_spend_log_row(
        self, client: LoggingClient, resources: ResourceManager
    ) -> None:
        capture_file = _capture_file()
        alias = f"e2e-callback-cost-{unique_marker()}"
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

        captured = _poll_captured(client, capture_file, response_id)
        assert captured, f"CustomLogger saw no success event for request_id={response_id} in {capture_file}"
        for row in captured:
            assert row.response_cost is not None and math.isclose(row.response_cost, spend_row.spend, rel_tol=1e-9), (
                f"{row.hook} ({row.call_type}) saw standard_logging_object.response_cost={row.response_cost} "
                f"but SpendLogs billed {spend_row.spend} for request_id={response_id}"
            )
