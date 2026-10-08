"""A `no-log: true` request is still billed, but its content must never be persisted or published.

`_ProxyDBLogger` is deliberately let through the `should_run_callback` no-log filter so the proxy can
bill the request. These tests pin the other half of that contract: the spend row keeps the cost
fields and loses the prompt/response content, in both the DB write and the collector publish.
"""

import datetime
import json
from datetime import timezone
from typing import Final

import pytest

import litellm
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

PROMPT_SECRET: Final = "PROMPT-SECRET-3f9a1c"
COMPLETION_SECRET: Final = "COMPLETION-SECRET-7b2d4e"


def _sl_object() -> dict:
    return {
        "messages": [{"role": "user", "content": PROMPT_SECRET}],
        "response": {"choices": [{"message": {"role": "assistant", "content": COMPLETION_SECRET}}]},
        "metadata": {"user_api_key": "hashed-key"},
        "model_map_information": None,
        "request_tags": [],
        "response_cost": 0.001,
    }


def _kwargs(no_log: bool) -> dict:
    return {
        "model": "gpt-4o-mini",
        "call_type": "acompletion",
        "litellm_params": {
            "no-log": no_log,
            "metadata": {"user_api_key": "hashed-key"},
            "proxy_server_request": {
                "url": "http://localhost:4000/v1/chat/completions",
                "body": {"messages": [{"role": "user", "content": PROMPT_SECRET}]},
            },
        },
        "standard_logging_object": _sl_object(),
        "response_cost": 0.001,
    }


def _response() -> litellm.ModelResponse:
    return litellm.ModelResponse(
        id="chatcmpl-no-log",
        choices=[{"message": {"role": "assistant", "content": COMPLETION_SECRET}}],
        usage=litellm.Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
    )


def _payload(monkeypatch, no_log: bool, store_prompts: bool) -> dict:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": store_prompts})
    now: Final = datetime.datetime.now(timezone.utc)
    return get_logging_payload(
        kwargs=_kwargs(no_log=no_log), response_obj=_response(), start_time=now, end_time=now
    )


@pytest.mark.parametrize("store_prompts", [True, False])
def test_no_log_request_persists_no_content(monkeypatch, store_prompts: bool):
    payload: Final = _payload(monkeypatch, no_log=True, store_prompts=store_prompts)
    stored: Final = json.dumps(payload, default=str)

    assert PROMPT_SECRET not in stored, "prompt leaked into the no-log spend row"
    assert COMPLETION_SECRET not in stored, "completion leaked into the no-log spend row"


def test_no_log_request_still_records_cost(monkeypatch):
    """The point of letting the spend writer through: the request is still billed."""
    payload: Final = _payload(monkeypatch, no_log=True, store_prompts=True)

    assert payload["request_id"]
    assert payload["model"] == "gpt-4o-mini"
    assert payload["total_tokens"] == 7
    assert payload["spend"] == 0.001


def test_logged_request_still_persists_content_when_prompt_storage_is_on(monkeypatch):
    """Control: the suppression is scoped to no-log, it is not a blanket redaction."""
    payload: Final = _payload(monkeypatch, no_log=False, store_prompts=True)

    # The request body carries the prompt for any call type; the SL `response` column carries the
    # completion. (`messages` is only populated for `_arealtime` calls, so it is not asserted here.)
    assert PROMPT_SECRET in payload["proxy_server_request"]
    assert COMPLETION_SECRET in payload["response"]


class _CapturingProducer:
    def __init__(self) -> None:
        self.published: list[bytes] = []

    async def publish(self, line: bytes) -> None:
        self.published.append(line)


@pytest.mark.asyncio
@pytest.mark.parametrize("no_log", [True, False])
async def test_collector_event_carries_no_content_for_no_log_requests(monkeypatch, no_log: bool):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"store_prompts_in_spend_logs": True})
    producer: Final = _CapturingProducer()
    logger: Final = _ProxyDBLogger(spend_event_producer=producer)
    now: Final = datetime.datetime.now(timezone.utc)

    await logger.async_log_success_event(_kwargs(no_log=no_log), _response(), now, now)

    assert producer.published, "expected the collector (offload) path to be taken"
    published: Final = producer.published[0].decode()
    if no_log:
        assert PROMPT_SECRET not in published
        assert COMPLETION_SECRET not in published
    else:
        assert PROMPT_SECRET in published
