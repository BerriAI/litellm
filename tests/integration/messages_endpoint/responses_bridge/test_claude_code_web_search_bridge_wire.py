import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc
from pydantic import JsonValue

_WEB_SEARCH_TOOL: Final = {
    "name": "WebSearch",
    "description": "Search the web. Returns result blocks with titles and URLs.",
    "input_schema": cc.schema(
        {
            "query": cc.field("The search query to use", type="string", minLength=2),
            "allowed_domains": cc.field(
                "Only include search results from these domains", type="array", items={"type": "string"}
            ),
            "blocked_domains": cc.field(
                "Never include search results from these domains", type="array", items={"type": "string"}
            ),
        },
        ("query",),
    ),
}


def test_web_search_tool_and_cited_output_on_responses_bridge(gateway: Gateway) -> None:
    request_body: Final = {
        **cc.frontier_request(
            f"cache-bust-{uuid.uuid4().hex}",
            "high",
            64000,
            prompt_text="Use web search to find the current LiteLLM version and answer in one word",
        ),
        "stream": False,
    }
    request_body["tools"] = [*request_body["tools"], _WEB_SEARCH_TOOL]

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses", request.target
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        tools: Final = body["tools"]
        assert tools[-1] == {
            "type": "function",
            "name": "WebSearch",
            "strict": False,
            "description": _WEB_SEARCH_TOOL["description"],
            "parameters": _WEB_SEARCH_TOOL["input_schema"],
        }, tools[-1]
        return Reply(
            body=cc.responses_completed(
                "ws",
                cc.OPENAI_BACKEND,
                (
                    {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "1.104.0",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://example.com/litellm",
                                        "title": "litellm releases",
                                        "start_index": 0,
                                        "end_index": 7,
                                    }
                                ],
                            }
                        ],
                    },
                ),
                {"input_tokens": 41, "output_tokens": 5, "total_tokens": 46},
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        payload: Final = cc.JSON_OBJECT.validate_json(response.content)
        assert payload["content"] == [{"type": "text", "text": "1.104.0"}], payload["content"]
        assert len(wire.drain()) == 1
