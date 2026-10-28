import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

BEDROCK_MODEL: Final = "us.anthropic.claude-opus-5-v1:0"
TOKEN: Final = "synthetic-bedrock-bearer"
SNIPPET: Final = "synthetic snippet about the integration harness"
INTERCEPTED_TURN: Final = (
    {"type": "server_tool_use", "id": "srvtoolu_synthetic", "name": "web_search", "input": {"query": "harness docs"}},
    {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_synthetic",
        "content": [
            {
                "type": "web_search_result",
                "url": "https://example.test/harness",
                "title": "Harness",
                "page_age": None,
                "encrypted_content": "",
                "snippet": SNIPPET,
            },
        ],
    },
    {"type": "text", "text": "The harness is documented at example.test"},
)
FLATTENED_TURN: Final = (
    {
        "type": "text",
        "text": f"Web search results for 'harness docs':\n\nTitle: Harness\nURL: https://example.test/harness\nSnippet: {SNIPPET}",
    },
    {"type": "text", "text": "The harness is documented at example.test"},
)
REPLY: Final = json.dumps(
    {
        "id": "msg_synthetic_replay",
        "type": "message",
        "role": "assistant",
        "model": BEDROCK_MODEL,
        "content": [{"type": "text", "text": "replay accepted"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 3},
    }
).encode()


def bedrock_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == f"/model/{BEDROCK_MODEL}/invoke"
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body: Final = json.loads(request.body)
    assert body["messages"] == [
        {"role": "user", "content": "where is the harness documented"},
        {"role": "assistant", "content": list(FLATTENED_TURN)},
        {"role": "user", "content": "and what does it say"},
    ], request.body.decode()
    assert "tools" not in body, request.body.decode()
    return Reply(body=REPLY)


@pytest.mark.covers("providers.bedrock_messages.replayed_intercepted_web_search_turn_is_flattened_to_text")
def test_replayed_intercepted_web_search_turn_reaches_bedrock_as_text_and_answers(gateway: Gateway) -> None:
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/{BEDROCK_MODEL}",
            api_key=TOKEN,
            api_base=wire.url,
            aws_region_name="us-east-1",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "where is the harness documented"},
                    {"role": "assistant", "content": list(INTERCEPTED_TURN)},
                    {"role": "user", "content": "and what does it say"},
                ],
            },
            headers={"x-api-key": gateway.key, "anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"] == [{"type": "text", "text": "replay accepted"}], response.text
        assert len(wire.drain()) == 1
