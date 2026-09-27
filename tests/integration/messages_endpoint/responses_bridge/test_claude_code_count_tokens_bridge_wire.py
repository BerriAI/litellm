import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint import _claude_code as cc


def test_count_tokens_on_openai_deployment_returns_token_count(gateway: Gateway) -> None:
    request_body: Final = {
        key: value
        for key, value in cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}").items()
        if key not in ("stream", "max_tokens", "thinking", "output_config")
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses/input_tokens", request.target
        return Reply(body=b'{"object": "response.input_tokens", "input_tokens": 37}')

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{cc.OPENAI_BACKEND}", api_base=wire.url, api_key=cc.OPENAI_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers=cc.cli_headers(gateway.key),
        )
        assert response.status_code == 200, response.text
        payload: Final = cc.JSON_OBJECT.validate_json(response.content)
        assert isinstance(payload.get("input_tokens"), int), payload
