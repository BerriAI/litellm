import uuid
from typing import Final

import pytest

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc


def test_count_tokens_forwards_to_anthropic_and_bills_nothing(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: /v1/messages/count_tokens on an anthropic deployment runs the internal token_counter "
        "and never forwards to the provider"
    )
    request_body: Final = {
        key: value
        for key, value in cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}").items()
        if key not in ("stream", "max_tokens", "thinking", "output_config")
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages/count_tokens", request.target
        body = cc.JSON_OBJECT.validate_json(request.body)
        assert body == {**request_body, "model": cc.FABLE}, body
        return Reply(body=b'{"input_tokens": 37}')

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{cc.FABLE}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key),
        )
        assert response.status_code == 200, response.text
        assert cc.JSON_OBJECT.validate_json(response.content) == {"input_tokens": 37}
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
