import uuid
from typing import Final

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc

_MODEL: Final = cc.SONNET


def test_claude_code_streaming_request_reaches_anthropic_intact_and_streams_back(gateway: Gateway) -> None:
    identity: Final = f"msg_cc_{uuid.uuid4().hex}"
    request_body: Final = cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}")
    cli_beta: Final = frozenset(cc.CLI_BETA.split(","))

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == cc.ANTHROPIC_API_KEY
        assert request.headers["anthropic-version"] == "2023-06-01"
        upstream_beta: Final = frozenset(request.headers.get("anthropic-beta", "").split(","))
        assert cli_beta <= upstream_beta, request.headers.get("anthropic-beta")
        body: Final = cc.JSON_OBJECT.validate_json(request.body)
        expected: Final = {**request_body, "model": _MODEL}
        assert body == expected, {
            key: (expected.get(key), body.get(key))
            for key in expected.keys() | body.keys()
            if expected.get(key) != body.get(key)
        }
        return Reply(
            content_type="text/event-stream",
            chunks=cc.text_stream(identity, _MODEL, "PONG", {"input_tokens": 12, "output_tokens": 4}),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key),
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), dict(response.headers)
        events: Final = cc.sse_events(response.text)
        assert [event for event, _ in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert events[2][1]["delta"] == {"type": "text_delta", "text": "PONG"}
        assert events[4][1]["delta"]["stop_reason"] == "end_turn"
        assert events[4][1]["usage"]["output_tokens"] == 4
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["prompt_tokens"] == 12 and rows[0]["completion_tokens"] == 4
