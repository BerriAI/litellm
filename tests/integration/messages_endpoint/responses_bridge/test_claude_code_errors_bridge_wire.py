import json
import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc


def test_openai_429_error_comes_back_in_anthropic_shape(gateway: Gateway) -> None:
    request_body: Final = {**cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}"), "stream": False}

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses", request.target
        return Reply(
            status=429,
            body=json.dumps(
                {"error": {"type": "rate_limit_error", "message": "Rate limit reached", "code": "rate_limit_exceeded"}}
            ).encode(),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY, num_retries=0
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key),
        )
        assert response.status_code == 429, response.text
        payload: Final = cc.JSON_OBJECT.validate_json(response.content)
        assert payload["type"] == "error", payload
        assert payload["error"]["type"] == "rate_limit_error", payload
        assert len(wire.drain()) == 1


def test_incomplete_responses_completion_maps_to_max_tokens(gateway: Gateway) -> None:
    request_body: Final = {**cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}"), "stream": False}

    def respond(request: Request) -> Reply:
        assert request.target == "/responses", request.target
        return Reply(
            body=cc.responses_completed(
                "inc",
                cc.OPENAI_BACKEND,
                (
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "PAR", "annotations": []}],
                    },
                ),
                {"input_tokens": 41, "output_tokens": 5, "total_tokens": 46},
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key),
        )
        assert response.status_code == 200, response.text
        payload: Final = cc.JSON_OBJECT.validate_json(response.content)
        assert payload["stop_reason"] == "max_tokens", payload
        assert len(wire.drain()) == 1
