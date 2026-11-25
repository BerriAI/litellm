import json
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

# Body shape from a customer report (Pylon #12371) and a live OpenAI repro on 2026-11-24
_MODERATION_BLOCKED: Final[dict[str, JsonValue]] = {
    "error": {
        "message": "Your request was rejected by the safety system. If you believe this is an error, contact us "
        "at help.openai.com and include the request ID req_integration.",
        "type": "image_generation_user_error",
        "param": None,
        "code": "moderation_blocked",
    }
}
_FALLBACK_IMAGE: Final = "ZmFsbGJhY2s="


def test_moderation_blocked_image_prompt_triggers_content_policy_fallback(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        if request.target.startswith("/blocked/"):
            return Reply(status=400, body=json.dumps(_MODERATION_BLOCKED).encode())
        return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _FALLBACK_IMAGE}]}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        blocked: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/blocked", api_key="synthetic-openai-key"
        )
        fallback: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/fallback", api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {
                "model": blocked,
                "prompt": "a transparent lighthouse at dawn",
                "content_policy_fallbacks": [{blocked: [fallback]}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"][0]["b64_json"] == _FALLBACK_IMAGE
        assert [request.target for request in wire.drain()] == [
            "/blocked/images/generations",
            "/fallback/images/generations",
        ]
