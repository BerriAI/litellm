import json
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest

from litellm.integrations.lago import LagoLogger
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.utils import ModelResponse, Usage

CALL_TIME: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def lago_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAGO_API_KEY", "lago-key")
    monkeypatch.setenv("LAGO_API_BASE", "https://lago.example")
    monkeypatch.setenv("LAGO_API_EVENT_CODE", "llm-usage")
    monkeypatch.delenv("LAGO_API_CHARGE_BY", raising=False)


def test_lago_event_bills_the_response_token_usage_to_the_end_user(lago_env: None) -> None:
    sent: Final[list[httpx.Request]] = []

    def lago(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={})

    logger: Final = LagoLogger()
    logger.sync_http_handler = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(lago)))
    response: Final = ModelResponse(model="gpt-4o", usage=Usage(prompt_tokens=12, completion_tokens=5, total_tokens=17))

    logger.log_success_event(
        kwargs={
            "model": "gpt-4o",
            "response_cost": 0.002,
            "litellm_params": {"metadata": {}, "proxy_server_request": {"body": {"user": "end-user-1"}}},
        },
        response_obj=response,
        start_time=CALL_TIME,
        end_time=CALL_TIME,
    )

    assert [str(request.url) for request in sent] == ["https://lago.example/api/v1/events"]
    event: Final = json.loads(sent[0].content)["event"]
    assert event["external_subscription_id"] == "end-user-1"
    assert event["code"] == "llm-usage"
    assert event["properties"] == {
        "model": "gpt-4o",
        "response_cost": 0.002,
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    }
