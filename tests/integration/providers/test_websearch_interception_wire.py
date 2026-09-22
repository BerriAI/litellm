import json
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request, wire_server

_QUERY: Final = "integration capped search"
_TEXT_BLOCK: Final = {"type": "text", "text": "searching once more"}
_NOT_INTERCEPTED: Final = "native tool reached the provider"
_SEARCH_RESULT_BLOCK: Final = {
    "type": "web_search_result",
    "url": "https://owned.invalid/a",
    "title": "Owned result",
    "page_age": None,
    "encrypted_content": "",
    "snippet": "owned snippet",
}


def _search_tool_use(identity: str) -> dict[str, object]:
    return {"type": "tool_use", "id": identity, "name": "litellm_web_search", "input": {"query": _QUERY}}


def _anthropic_reply(identity: str, content: list[dict[str, object]], stop_reason: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": content,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }
        ).encode()
    )


@contextmanager
def _intercepting_search_tool(gateway: Gateway, search_api_base: str) -> Iterator[None]:
    name: Final = "integration-searxng-" + uuid.uuid4().hex
    created: Final = gateway.post(
        "/search_tools",
        {
            "search_tool": {
                "search_tool_name": name,
                "litellm_params": {"search_provider": "searxng", "api_base": search_api_base},
            }
        },
    )
    settings: Final = {"enabled": True, "enabled_providers": ["anthropic"], "search_tool_name": name}
    gateway.post("/config/update", {"litellm_settings": {"websearch_interception_params": settings}})
    try:
        yield
    finally:
        gateway.post("/config/update", {"litellm_settings": {"websearch_interception_params": {"enabled": False}}})
        gateway.request("DELETE", f"/search_tools/{created['search_tool_id']}")


@pytest.mark.covers(
    "other.provider_wire.anthropic.websearch_interception_capped_loop_ends_turn_without_internal_tool_use"
)
def test_capped_websearch_interception_loop_ends_turn_instead_of_exposing_internal_tool_use(gateway: Gateway) -> None:
    identity: Final = "websearch-wire-" + uuid.uuid4().hex
    searched: Final = threading.Event()

    def respond(request: Request) -> Reply:
        parts: Final = urlsplit(request.target)
        if request.method == "GET" and parts.path == "/search":
            assert parse_qs(parts.query)["q"] == [_QUERY], request.target
            searched.set()
            return Reply(
                body=json.dumps(
                    {
                        "results": [
                            {"title": "Owned result", "url": "https://owned.invalid/a", "content": "owned snippet"}
                        ]
                    }
                ).encode()
            )
        assert request.method == "POST" and parts.path == "/v1/messages", request.target
        body: Final = json.loads(request.body)
        if any(tool.get("type") == "web_search_20250305" for tool in body["tools"]):
            return _anthropic_reply(identity, [{"type": "text", "text": _NOT_INTERCEPTED}], "end_turn")
        assert [tool["name"] for tool in body["tools"]] == ["litellm_web_search"], body["tools"]
        return _anthropic_reply(identity, [_TEXT_BLOCK, _search_tool_use(identity)], "tool_use")

    def send(model: str) -> httpx.Response:
        return gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": identity + " attempt " + uuid.uuid4().hex}],
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
            },
        )

    def searched_through_proxy(response: httpx.Response) -> bool:
        return searched.is_set() and _NOT_INTERCEPTED not in response.text

    with wire_server(respond) as wire, gateway.scenario() as scenario, _intercepting_search_tool(gateway, wire.url):
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929", api_base=wire.url, api_key="synthetic-anthropic-key"
        )
        response: Final = eventually(lambda: send(model), searched_through_proxy, seconds=40)
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["stop_reason"] == "end_turn", response.text
        content: Final = body["content"]
        assert [block["type"] for block in content] == ["server_tool_use", "web_search_tool_result", "text"], (
            response.text
        )
        assert content[0]["name"] == "web_search" and content[0]["input"] == {"query": _QUERY}, response.text
        assert content[1]["tool_use_id"] == content[0]["id"], response.text
        assert content[1]["content"] == [_SEARCH_RESULT_BLOCK], response.text
        assert content[2] == _TEXT_BLOCK, response.text
        targets: Final = tuple((request.method, urlsplit(request.target).path) for request in wire.drain())
        assert targets[-3:] == (("POST", "/v1/messages"), ("GET", "/search"), ("POST", "/v1/messages")), targets
