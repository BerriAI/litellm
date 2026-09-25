import json
from pathlib import Path
from typing import Final
from uuid import uuid4

import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-5.4-mini"
_API_KEY: Final = "synthetic-openai-key"
_CLIENT_HEADER: Final = "x-my-new-header"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OUTPUT_MESSAGE: Final[dict[str, JsonValue]] = {
    "type": "message",
    "id": "msg_forwarded",
    "status": "completed",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "header wire control", "annotations": []}],
}
_RESPONSE: Final = json.dumps(
    {
        "id": "resp_forwarded",
        "object": "response",
        "status": "completed",
        "created_at": 1700000000,
        "model": _BACKEND,
        "output": [_OUTPUT_MESSAGE],
        "usage": {
            "input_tokens": 9,
            "output_tokens": 3,
            "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
).encode()


def _forwarding_config(directory: Path) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["general_settings"]["forward_client_headers_to_llm_api"] = True
    path: Final = directory / "forwarding.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.mark.covers("providers.responses_api.forwarded_client_headers_reach_the_provider")
def test_client_x_header_is_forwarded_to_the_provider_on_responses(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"hello-from-client-{uuid4().hex}"
    prompt: Final = f"forward my header {marker}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        assert request.headers.get(_CLIENT_HEADER) == marker, dict(request.headers)
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND and body["input"] == prompt, request.body
        return Reply(body=_RESPONSE)

    with (
        wire_server(respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_forwarding_config(tmp_path)) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = candidate.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": prompt, "stream": False},
            headers={_CLIENT_HEADER: marker},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["output"] == [
            {
                **_OUTPUT_MESSAGE,
                "phase": None,
                "content": [
                    {"type": "output_text", "text": "header wire control", "annotations": [], "logprobs": None}
                ],
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
