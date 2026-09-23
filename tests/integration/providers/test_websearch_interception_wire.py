import json
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

BEDROCK_MODEL: Final = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
INVOKE_TARGET: Final = f"/model/{BEDROCK_MODEL}/invoke"
SEARCH_TARGET: Final = "/tavily/search"
SEARCH_RESULT: Final = {
    "title": "Synthetic result",
    "url": "https://example.test/result",
    "content": "the snippet text",
}


def sse_events(text: str) -> tuple[tuple[str, dict[str, object]], ...]:
    frames: Final = tuple(frame for frame in text.split("\n\n") if frame.strip())
    return tuple(
        (
            next(line.removeprefix("event: ") for line in frame.splitlines() if line.startswith("event: ")),
            json.loads(next(line.removeprefix("data: ") for line in frame.splitlines() if line.startswith("data: "))),
        )
        for frame in frames
    )


@pytest.mark.covers("other.provider_wire.bedrock.websearch_interception_streamed_capped_turn_ends_with_native_results")
def test_streamed_web_search_turn_capped_by_max_agentic_loops_ends_turn_with_snippets_and_ordered_blocks(
    gateway: Gateway, tmp_path: Path
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request.target
        body: Final = json.loads(request.body)
        if request.target == SEARCH_TARGET:
            assert request.headers["authorization"] == "Bearer synthetic-tavily-key"
            assert body["query"] == "query-0", body
            return Reply(body=json.dumps({"query": "query-0", "results": [SEARCH_RESULT]}).encode())
        assert request.target == INVOKE_TARGET
        assert request.headers["authorization"] == "Bearer synthetic-bedrock-token"
        assert [tool["name"] for tool in body["tools"]] == ["litellm_web_search"], body["tools"]
        assert "stream" not in body, body
        depth: Final = sum(
            1
            for message in body["messages"]
            if isinstance(message["content"], list)
            for block in message["content"]
            if block["type"] == "tool_result"
        )
        if depth == 1:
            assert body["messages"][2]["content"] == [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_0",
                    "content": "Title: Synthetic result\nURL: https://example.test/result\nSnippet: the snippet text",
                }
            ], body["messages"]
        return Reply(
            body=json.dumps(
                {
                    "id": f"msg_{depth}",
                    "type": "message",
                    "role": "assistant",
                    "model": BEDROCK_MODEL,
                    "content": [
                        {"type": "text", "text": f"turn-{depth}"},
                        {
                            "type": "tool_use",
                            "id": f"toolu_{depth}",
                            "name": "litellm_web_search",
                            "input": {"query": f"query-{depth}"},
                        },
                    ],
                    "stop_reason": "tool_use",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 4},
                }
            ).encode()
        )

    with wire_server(respond) as wire:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["search_tools"] = [
            {
                "search_tool_name": "integration-search",
                "litellm_params": {
                    "search_provider": "tavily",
                    "api_key": "synthetic-tavily-key",
                    "api_base": wire.url + "/tavily",
                },
            }
        ]
        config["litellm_settings"].update(
            {
                "callbacks": ["websearch_interception"],
                "websearch_interception_params": {
                    "enabled_providers": ["bedrock"],
                    "search_tool_name": "integration-search",
                    "max_agentic_loops": 1,
                },
            }
        )
        path: Final = tmp_path / "websearch.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model=f"bedrock/{BEDROCK_MODEL}",
                api_key="synthetic-bedrock-token",
                api_base=wire.url,
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=wire.url,
            )
            response: Final = candidate.request(
                "POST",
                "/v1/messages",
                {
                    "model": model,
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [{"role": "user", "content": "search control"}],
                    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                },
            )
            assert response.status_code == 200, response.text
            events: Final = sse_events(response.text)
            assert [name for name, _ in events][:1] == ["message_start"], response.text
            assert [name for name, _ in events][-2:] == ["message_delta", "message_stop"], response.text
            for position, (name, event) in enumerate(events):
                if name == "content_block_stop":
                    assert event["index"] in {
                        earlier_event["index"]
                        for earlier, earlier_event in events[:position]
                        if earlier == "content_block_start"
                    }, response.text
            started: Final = tuple(event["content_block"] for name, event in events if name == "content_block_start")
            search_ids: Final = tuple(block["id"] for block in started if block["type"] == "server_tool_use")
            assert search_ids and all(search_id.startswith("srvtoolu_") for search_id in search_ids), response.text
            assert started[-1] == {"type": "text", "text": ""}, response.text
            assert started[:-1] == tuple(
                block
                for search_id in search_ids
                for block in (
                    {"type": "server_tool_use", "id": search_id, "name": "web_search", "input": {"query": "query-0"}},
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": search_id,
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.test/result",
                                "title": "Synthetic result",
                                "page_age": None,
                                "encrypted_content": "",
                                "snippet": "the snippet text",
                            }
                        ],
                    },
                )
            ), response.text
            assert (
                "".join(event["delta"]["text"] for name, event in events if name == "content_block_delta") == "turn-1"
            ), response.text
            assert [event["delta"]["stop_reason"] for name, event in events if name == "message_delta"] == [
                "end_turn"
            ], response.text
            assert "litellm_web_search" not in response.text, response.text
            assert [request.target for request in wire.drain()] == [INVOKE_TARGET, SEARCH_TARGET, INVOKE_TARGET]


import threading
import uuid
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from integration._support.client import Gateway, eventually

_QUERY: Final = "integration capped search"
_TEXT_BLOCK: Final = {"type": "text", "text": "searching once more"}
_NOT_INTERCEPTED: Final = "native tool reached the provider"
_FINAL_BLOCK: Final = {"type": "text", "text": "answered from the stored backend"}
_OWNED_RESULT_TEXT: Final = "Title: Owned result\nURL: https://owned.invalid/a\nSnippet: owned snippet"
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


@pytest.mark.covers(
    "other.provider_wire.anthropic.websearch_interception_capped_loop_ends_turn_without_internal_tool_use"
)
def test_capped_websearch_interception_loop_ends_turn_instead_of_exposing_internal_tool_use(
    gateway: Gateway, tmp_path: Path
) -> None:
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

    def send(candidate: Gateway, model: str) -> httpx.Response:
        return candidate.request(
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

    with wire_server(respond) as wire:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["search_tools"] = [
            {
                "search_tool_name": "integration-searxng",
                "litellm_params": {"search_provider": "searxng", "api_base": wire.url},
            }
        ]
        config["litellm_settings"].update(
            {
                "callbacks": ["websearch_interception"],
                "websearch_interception_params": {
                    "enabled": True,
                    "enabled_providers": ["anthropic"],
                    "search_tool_name": "integration-searxng",
                },
            }
        )
        path: Final = tmp_path / "websearch.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            response: Final = eventually(lambda: send(candidate, model), searched_through_proxy, seconds=40)
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


@pytest.mark.covers("other.provider_wire.anthropic.websearch_interception_uses_database_search_tool_backend")
def test_database_created_search_tool_backend_receives_the_intercepted_query_over_a_same_named_config_tool(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = "websearch-db-" + uuid.uuid4().hex
    tool_name: Final = "integration-db-searxng-" + uuid.uuid4().hex
    searched: Final = threading.Event()

    def respond(request: Request) -> Reply:
        parts: Final = urlsplit(request.target)
        if request.method == "GET" and parts.path == "/database/search":
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
        results: Final = [
            block
            for message in body["messages"]
            if isinstance(message["content"], list)
            for block in message["content"]
            if block["type"] == "tool_result"
        ]
        if not results:
            return _anthropic_reply(identity, [_TEXT_BLOCK, _search_tool_use(identity)], "tool_use")
        assert results == [{"type": "tool_result", "tool_use_id": identity, "content": _OWNED_RESULT_TEXT}], results
        return _anthropic_reply(identity, [_FINAL_BLOCK], "end_turn")

    def send(candidate: Gateway, model: str) -> httpx.Response:
        return candidate.request(
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

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        created: Final = gateway.post(
            "/search_tools",
            {
                "search_tool": {
                    "search_tool_name": tool_name,
                    "litellm_params": {"search_provider": "searxng", "api_base": wire.url + "/database"},
                }
            },
        )
        scenario.cleanups.callback(gateway.request, "DELETE", f"/search_tools/{created['search_tool_id']}")
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["search_tools"] = [
            {
                "search_tool_name": tool_name,
                "litellm_params": {"search_provider": "searxng", "api_base": wire.url + "/config"},
            }
        ]
        config["litellm_settings"].update(
            {
                "callbacks": ["websearch_interception"],
                "websearch_interception_params": {
                    "enabled": True,
                    "enabled_providers": ["anthropic"],
                    "search_tool_name": tool_name,
                },
            }
        )
        path: Final = tmp_path / "websearch-db.yaml"
        path.write_text(yaml.safe_dump(config))
        environment: Final = {"ANTHROPIC_API_BASE": wire.url}
        with owned_proxy(gateway, tmp_path, environment, config=path) as candidate, candidate.scenario() as models:
            model: Final = models.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=wire.url, api_key="synthetic-anthropic-key"
            )
            response: Final = eventually(lambda: send(candidate, model), searched_through_proxy, seconds=40)
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["stop_reason"] == "end_turn", response.text
        assert body["content"][-1] == _FINAL_BLOCK, response.text
        found: Final = [
            (result["url"], result["title"])
            for block in body["content"]
            if block["type"] == "web_search_tool_result"
            for result in block["content"]
        ]
        assert found == [("https://owned.invalid/a", "Owned result")], response.text
        assert "litellm_web_search" not in response.text, response.text
        targets: Final = tuple((request.method, urlsplit(request.target).path) for request in wire.drain())
        assert targets[-3:] == (("POST", "/v1/messages"), ("GET", "/database/search"), ("POST", "/v1/messages")), (
            targets
        )
