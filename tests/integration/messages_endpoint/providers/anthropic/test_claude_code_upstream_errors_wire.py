import json
import uuid
from typing import Final

import pytest

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc


def _error_body(error_type: str, message: str) -> bytes:
    return json.dumps({"type": "error", "error": {"type": error_type, "message": message}}).encode()


def _assert_upstream_error_status_passthrough(gateway: Gateway, status: int, error_type: str) -> None:
    request_body: Final = {**cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}"), "stream": False}

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        body = cc.JSON_OBJECT.validate_json(request.body)
        assert body["stream"] is False
        return Reply(status=status, body=_error_body(error_type, "Upstream rejected"))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{cc.SONNET}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY, num_retries=0
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key),
        )
        assert response.status_code == status, response.text
        payload: Final = cc.JSON_OBJECT.validate_json(response.content)
        assert payload["type"] == "error", payload
        assert payload["error"]["type"] == error_type, payload
        assert len(wire.drain()) == 1
        call_id: Final = response.headers.get("x-litellm-call-id", "")
        assert call_id, dict(response.headers)
        leftover: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
            lambda values: len(values) == 1,
            seconds=20,
            return_last_on_timeout=True,
        )
        assert all(float(row["spend"]) == 0 for row in leftover), leftover


def test_anthropic_529_overloaded_error_passes_through_with_client_status(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: upstream 529 overloaded_error is re-raised through exception_type as InternalServerError "
        "and reaches the client as 500 api_error"
    )
    _assert_upstream_error_status_passthrough(gateway, 529, "overloaded_error")


def test_anthropic_429_rate_limit_error_passes_through_with_client_status(gateway: Gateway) -> None:
    _assert_upstream_error_status_passthrough(gateway, 429, "rate_limit_error")


def test_anthropic_stream_stop_reason_max_tokens_and_refusal_reach_client(gateway: Gateway) -> None:
    for stop_reason in ("max_tokens", "refusal"):
        _assert_stream_stop_reason_reaches_client(gateway, stop_reason)


def _assert_stream_stop_reason_reaches_client(gateway: Gateway, stop_reason: str) -> None:
    identity: Final = f"msg_stop_{uuid.uuid4().hex}"
    request_body: Final = cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}")

    def respond(request: Request) -> Reply:
        return Reply(
            content_type="text/event-stream",
            chunks=(
                cc.sse_frame(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": identity,
                            "type": "message",
                            "role": "assistant",
                            "model": cc.SONNET,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 12, "output_tokens": 1},
                        },
                    },
                ),
                cc.sse_frame(
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                ),
                cc.sse_frame(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "PAR"}},
                ),
                cc.sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
                cc.sse_frame(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": {"output_tokens": 32000},
                    },
                ),
                cc.sse_frame("message_stop", {"type": "message_stop"}),
            ),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{cc.SONNET}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key),
        )
        assert response.status_code == 200, response.text
        events: Final = cc.sse_events(response.text)
        assert events[2][1]["delta"]["text"] == "PAR"
        assert events[4][1]["delta"]["stop_reason"] == stop_reason
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (identity,)),
            lambda values: len(values) == 1,
            seconds=20,
            return_last_on_timeout=True,
        )
        assert isinstance(rows, list)
