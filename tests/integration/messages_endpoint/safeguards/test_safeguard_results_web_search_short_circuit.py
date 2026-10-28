"""A web search answered by the interception short circuit never reaches a backend or the classifier, yet auto mode
still receives an ``available`` result with no tool uses to judge."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Final

import pytest
from integration._support.client import list_value, object_value
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    CHAT_MODEL,
    TAVILY_TARGET,
    Rig,
    auto_mode_body,
    available,
    json_body,
    deployment,
    nonce,
    safeguards_rig,
    sse_events,
    stream_results,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

WEB_SEARCH_TOOLS: Final[list[JsonValue]] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}]
DATABRICKS_CLAUDE_MODEL: Final = "safeguards-databricks-claude"
SEARCHING_MODELS: Final = (CHAT_MODEL, DATABRICKS_CLAUDE_MODEL)


LITELLM_SETTINGS: Final[Mapping[str, JsonValue]] = {
    "callbacks": ["websearch_interception"],
    "websearch_interception_params": {
        "enabled_providers": ["fireworks_ai", "databricks"],
        "search_tool_name": "tavily-search",
    },
}


def search_tools(wire_url: str) -> Mapping[str, JsonValue]:
    tavily: Final[Mapping[str, JsonValue]] = {
        "search_tool_name": "tavily-search",
        "litellm_params": {
            "search_provider": "tavily",
            "api_key": "synthetic-tavily-key",
            "api_base": f"{wire_url}/tavily",
        },
    }
    return {"search_tools": [dict(tavily)]}


def databricks_claude(wire_url: str) -> Sequence[Mapping[str, JsonValue]]:
    return (
        deployment(
            DATABRICKS_CLAUDE_MODEL, "databricks/databricks-claude-sonnet-4-6", f"{wire_url}/dbx/serving-endpoints"
        ),
    )


@pytest.fixture(scope="module")
def searching(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("web-search")
    with safeguards_rig(
        directory, litellm_settings=LITELLM_SETTINGS, top_level=search_tools, extra_models=databricks_claude
    ) as rig:
        yield rig


def search_text(content: JsonValue) -> str:
    blocks: Final = tuple(object_value(block) for block in list_value(content))
    return "".join(str(block["text"]) for block in blocks if block["type"] == "text")


def streamed_text(raw: bytes) -> str:
    deltas: Final = tuple(
        object_value(event["delta"]) for event in sse_events(raw) if event.get("type") == "content_block_delta"
    )
    return "".join(str(delta["text"]) for delta in deltas if delta.get("type") == "text_delta")


def assert_one_search_and_nothing_else(rig: Rig, request_nonce: str) -> None:
    received: Final = rig.wire.drain()
    targets: Final = [request.target for request in received]
    assert targets == [TAVILY_TARGET], targets
    assert received[0].headers.get("authorization") == "Bearer synthetic-tavily-key", received[0].headers


@pytest.mark.parametrize("model", SEARCHING_MODELS)
def test_short_circuited_search_is_available_with_no_tool_uses(searching: Rig, model: str) -> None:
    request_nonce: Final = nonce()
    body: Final = json_body(
        searching.post(auto_mode_body(model, request_nonce, tools=WEB_SEARCH_TOOLS, prompt="search the web for"))
    )
    assert request_nonce in search_text(body["content"]), body
    assert body["safeguard_results"] == available({}), body
    assert_one_search_and_nothing_else(searching, request_nonce)


@pytest.mark.parametrize("model", SEARCHING_MODELS)
def test_short_circuited_search_stream_is_available_with_no_tool_uses(searching: Rig, model: str) -> None:
    request_nonce: Final = nonce()
    status, raw = searching.stream(
        auto_mode_body(model, request_nonce, tools=WEB_SEARCH_TOOLS, prompt="search the web for")
    )
    assert status == 200, raw
    assert request_nonce in streamed_text(raw), raw
    assert stream_results(raw) == available({}), raw
    assert_one_search_and_nothing_else(searching, request_nonce)
