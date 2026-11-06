import json
import time
import uuid
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway
from tests.integration._support.timing import assert_enforced_within
from tests.integration._support.wire import Reply, Request, wire_server

RETRY_AFTER_SECONDS: Final = 2
CALLER_TIMEOUT_SECONDS: Final = 4
NUM_RETRIES: Final = 3
CALLER_RESPONSE_SLACK_SECONDS: Final = 1.5


def _unavailable(_: Request) -> Reply:
    return Reply(
        status=503,
        body=json.dumps({"error": {"message": "overloaded", "type": "server_error"}}).encode(),
        headers={"retry-after": str(RETRY_AFTER_SECONDS)},
    )


def _chat(gateway: Gateway, model: str, *, headers: dict[str, str], body: dict[str, JsonValue]) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"deadline {uuid.uuid4().hex}"}], **body},
        headers=headers,
    )


@pytest.mark.parametrize(
    "headers, body",
    [
        pytest.param({"x-litellm-timeout": str(CALLER_TIMEOUT_SECONDS)}, {}, id="x-litellm-timeout header"),
        pytest.param({}, {"timeout": CALLER_TIMEOUT_SECONDS}, id="timeout in the body"),
    ],
)
def test_retries_and_retry_after_sleeps_stop_at_the_callers_timeout(
    gateway: Gateway, headers: dict[str, str], body: dict[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario, wire_server(_unavailable) as wire:
        model: Final = scenario.model(api_base=wire.url, num_retries=NUM_RETRIES)
        started: Final = time.monotonic()
        response: Final = _chat(gateway, model, headers=headers, body=body)
        elapsed: Final = time.monotonic() - started
        assert response.status_code == 503, response.text
        attempts: Final = len(wire.drain())
        assert attempts == 2, (attempts, elapsed)
        assert_enforced_within(elapsed, configured=CALLER_TIMEOUT_SECONDS, slack=CALLER_RESPONSE_SLACK_SECONDS)
