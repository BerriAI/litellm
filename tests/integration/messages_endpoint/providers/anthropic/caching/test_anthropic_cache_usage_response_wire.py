import uuid
from typing import Final

from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_MODEL: Final = "claude-sonnet-4-5"
_ANTHROPIC_USAGE: Final = {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_read_input_tokens": 400,
    "cache_creation_input_tokens": 300,
    "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200},
}


def _client_body(stream: bool) -> dict[str, JsonValue]:
    return {**cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}"), "stream": stream}


def test_streamed_cache_read_and_write_tokens_reach_the_client_unchanged(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return Reply(
            chunks=cc.message_stream(
                f"msg_{uuid.uuid4().hex}",
                _MODEL,
                ({"type": "text", "text": "PONG"},),
                _ANTHROPIC_USAGE,
                final_usage=_ANTHROPIC_USAGE,
            ),
            content_type="text/event-stream",
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request("POST", "/v1/messages", {**_client_body(stream=True), "model": model})
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
    assert cc.streamed_start_usage(response.text) == {
        "input_tokens": 100,
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 300,
        "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200},
    }, response.text
    assert cc.streamed_usage(response.text) == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 300,
        "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200},
    }, response.text


def test_non_streamed_cache_read_and_write_tokens_reach_the_client_unchanged(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return Reply(
            body=cc.message_reply(
                f"msg_{uuid.uuid4().hex}", _MODEL, ({"type": "text", "text": "PONG"},), _ANTHROPIC_USAGE
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request("POST", "/v1/messages", {**_client_body(stream=False), "model": model})
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
    assert response.json()["usage"] == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 400,
        "cache_creation_input_tokens": 300,
        "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200},
    }, response.text
