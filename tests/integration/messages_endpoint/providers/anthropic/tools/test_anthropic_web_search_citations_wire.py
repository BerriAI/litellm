import uuid
from typing import Final

from integration._support import claude_code as cc
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server

WEB_SEARCH_TOOL: Final = {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}


def _web_search_stream(identity: str) -> tuple[bytes, ...]:
    return (
        cc.sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": cc.FABLE,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 20, "output_tokens": 1, "server_tool_use": {"web_search_requests": 1}},
                },
            },
        ),
        cc.sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
            },
        ),
        cc.sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"query": "current LiteLLM version"}'},
            },
        ),
        cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        cc.sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [
                        {
                            "type": "web_search_result",
                            "title": "litellm releases",
                            "url": "https://example.com/litellm",
                            "page_age": None,
                            "encrypted_content": "enc_ws_1",
                        }
                    ],
                },
            },
        ),
        cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 1}),
        cc.sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        ),
        cc.sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "1.104.0"}},
        ),
        cc.sse_frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {
                    "type": "citations_delta",
                    "citation": {
                        "type": "web_search_result_location",
                        "url": "https://example.com/litellm",
                        "title": "litellm releases",
                        "cited_text": "version 1.104.0",
                        "encrypted_index": "eidx_1",
                    },
                },
            },
        ),
        cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 2}),
        cc.sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 15, "server_tool_use": {"web_search_requests": 1}},
            },
        ),
        cc.sse_frame("message_stop", {"type": "message_stop"}),
    )


def test_web_search_tool_passthrough_and_cited_response(gateway: Gateway) -> None:
    identity: Final = f"msg_ws_{uuid.uuid4().hex}"
    base: Final = cc.frontier_request(
        f"cache-bust-{uuid.uuid4().hex}",
        "high",
        64000,
        prompt_text="Use web search to find the current LiteLLM version and answer in one word",
    )
    request_body: Final = {
        **base,
        "tools": [*base["tools"], WEB_SEARCH_TOOL],
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        expected: Final = {**request_body, "model": cc.FABLE}
        assert body == expected, {
            key: (expected.get(key), body.get(key))
            for key in expected.keys() | body.keys()
            if expected.get(key) != body.get(key)
        }
        assert body["tools"][-1] == WEB_SEARCH_TOOL
        assert len({tool["name"] for tool in body["tools"]}) == len(body["tools"]), body["tools"]
        return Reply(content_type="text/event-stream", chunks=_web_search_stream(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{cc.FABLE}",
            api_base=wire.url,
            api_key=cc.ANTHROPIC_API_KEY,
            input_cost_per_token=1e-6,
            output_cost_per_token=5e-6,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        events: Final = cc.sse_events(response.text)
        started: Final = [
            (data["index"], data["content_block"]["type"]) for event, data in events if event == "content_block_start"
        ]
        assert started == [(0, "server_tool_use"), (1, "web_search_tool_result"), (2, "text")], started
        citations: Final = [
            data["delta"] for event, data in events if data.get("delta", {}).get("type") == "citations_delta"
        ]
        assert len(citations) == 1 and citations[0]["citation"]["url"] == "https://example.com/litellm", citations
        start_usage: Final = events[0][1]["message"]["usage"]
        assert start_usage["server_tool_use"]["web_search_requests"] == 1, start_usage
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        token_cost: Final = 20 * 1e-6 + 15 * 5e-6
        assert float(rows[0]["spend"]) >= token_cost, dict(rows[0])
