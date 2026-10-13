"""The /usage/ai/chat tools hand the model the reason the daily activity query rejected its dates, the same check
the daily activity routes answer 400 with, instead of a generic fetch error."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, object_value
from integration._support.upstream import delete_scenario, register_scenario
from integration.compatibility._status_code_audit import Upstream, json_response, owned_gateway
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(180)

_TOOLS: Final = ("get_usage_data", "get_team_usage_data", "get_tag_usage_data")


@dataclass(frozen=True, slots=True)
class _UsageChat:
    gateway: Gateway
    identity: str


@pytest.fixture(scope="module")
def usage_chat(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_UsageChat]:
    """A proxy whose usage chat model (`openai/...`, called straight through litellm) is the scripted upstream."""
    identity: Final = f"audit-usage-chat-{uuid.uuid4().hex}"
    handle: Final = register_scenario(identity, json_response({}))
    try:
        with owned_gateway(
            tmp_path_factory.mktemp("audit-usage-chat"),
            {"model_list": [], "router_settings": {"num_retries": 0}},
            environment={"OPENAI_API_BASE": handle.api_base(), "OPENAI_API_KEY": identity},
        ) as owned:
            yield _UsageChat(owned, identity)
    finally:
        delete_scenario(handle)


def _tool_calls(start_date: str, end_date: str) -> dict[str, JsonValue]:
    arguments: Final = json.dumps({"start_date": start_date, "end_date": end_date})
    return {
        "id": "chatcmpl-usage",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": f"call-{tool}", "type": "function", "function": {"name": tool, "arguments": arguments}}
                        for tool in _TOOLS
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _tool_results(usage_chat: _UsageChat, start_date: str, end_date: str) -> tuple[dict[str, str], tuple[str, ...]]:
    """Runs one chat turn whose model calls every tool with these dates; returns the tool messages the follow-up
    model call carried, keyed by tool, and the tool statuses the SSE stream reported."""
    register_scenario(usage_chat.identity, json_response(_tool_calls(start_date, end_date)))
    Upstream(usage_chat.gateway.upstream_url).drain()
    with usage_chat.gateway.client.stream(
        "POST",
        "/usage/ai/chat",
        json={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "How much did we spend?"}]},
        headers={"Authorization": f"Bearer {usage_chat.gateway.key}"},
        timeout=60,
    ) as response:
        assert response.status_code == 200
        events: Final = tuple(
            object_value(JSON_OBJECT.validate_json(line.removeprefix("data: ")))
            for line in response.iter_lines()
            if line.startswith("data: ")
        )
    calls: Final = Upstream(usage_chat.gateway.upstream_url).calls(usage_chat.identity)
    assert len(calls) == 2, calls
    messages: Final = object_value(calls[1]["body"]).get("messages")
    assert isinstance(messages, list), calls[1]
    tool_messages: Final = {
        str(message["tool_call_id"]).removeprefix("call-"): str(message["content"])
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    }
    statuses: Final = tuple(str(event.get("status")) for event in events if event.get("type") == "tool_call")
    return tool_messages, statuses


_REVERSED: Final = "Invalid date range: end_date must be on or after start_date"
_NOT_CANONICAL: Final = "Invalid date range: start_date and end_date must be valid YYYY-MM-DD dates"
_NO_TEAM_DATA: Final = "No Team usage data found for the given date range."
_NO_TAG_DATA: Final = "No Tag usage data found for the given date range."


@pytest.mark.parametrize(
    ("start_date", "end_date", "team_message", "tag_message"),
    (
        pytest.param("2026-10-09", "2026-10-01", _REVERSED, _REVERSED, id="reversed"),
        pytest.param("2026-13-01", "2026-10-09", _NOT_CANONICAL, _NOT_CANONICAL, id="invalid-month"),
        pytest.param("2001/01/01", "2001-01-09", _NOT_CANONICAL, _NOT_CANONICAL, id="slashes"),
        pytest.param("2001-01-01", "2001-01-09", _NO_TEAM_DATA, _NO_TAG_DATA, id="valid"),
    ),
)
def test_usage_chat_tools_answer_with_the_query_layers_date_check(
    usage_chat: _UsageChat, start_date: str, end_date: str, team_message: str, tag_message: str
) -> None:
    """The team and tag tools give the model the daily activity routes' 400 reason; the global tool, whose query
    takes these dates without that check, still answers with its usage summary."""
    tool_messages, statuses = _tool_results(usage_chat, start_date, end_date)
    assert tool_messages["get_usage_data"].startswith("Total Spend: $"), tool_messages
    assert tool_messages["get_team_usage_data"] == team_message, tool_messages
    assert tool_messages["get_tag_usage_data"] == tag_message, tool_messages
    assert statuses == ("running", "complete") * len(_TOOLS), statuses
