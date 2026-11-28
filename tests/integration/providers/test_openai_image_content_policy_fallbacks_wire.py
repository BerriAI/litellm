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


def test_moderation_blocked_chat_on_an_openai_compatible_provider_stays_a_bad_request(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return Reply(status=400, body=json.dumps(_MODERATION_BLOCKED).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        blocked: Final = scenario.model(
            model="hosted_vllm/served-model", api_base=f"{wire.url}/blocked", api_key="synthetic-vllm-key"
        )
        fallback: Final = scenario.model(
            model="hosted_vllm/served-model", api_base=f"{wire.url}/fallback", api_key="synthetic-vllm-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": blocked,
                "messages": [{"role": "user", "content": "describe a lighthouse"}],
                "content_policy_fallbacks": [{blocked: [fallback]}],
            },
        )
        assert response.status_code == 400, response.text
        assert [request.target for request in wire.drain()] == ["/blocked/chat/completions"]


_PNG: Final = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff"
    b"\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _blocked_then_image(request: Request) -> Reply:
    if request.target.startswith("/blocked/"):
        return Reply(status=400, body=json.dumps(_MODERATION_BLOCKED).encode())
    return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": _FALLBACK_IMAGE}]}).encode())


def test_moderation_blocked_image_edit_triggers_content_policy_fallback(gateway: Gateway) -> None:
    with wire_server(_blocked_then_image) as wire, gateway.scenario() as scenario:
        blocked: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/blocked", api_key="synthetic-openai-key"
        )
        fallback: Final = scenario.model(
            model="openai/gpt-image-1", api_base=f"{wire.url}/fallback", api_key="synthetic-openai-key"
        )
        key: Final = scenario.key(router_settings={"content_policy_fallbacks": [{blocked: [fallback]}]})
        response: Final = gateway.request_multipart(
            "/v1/images/edits",
            {"model": blocked, "prompt": "turn the lighthouse green"},
            {"image": ("lighthouse.png", _PNG, "image/png")},
            key=key,
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"][0]["b64_json"] == _FALLBACK_IMAGE
        assert [request.target for request in wire.drain()] == ["/blocked/images/edits", "/fallback/images/edits"]


def test_moderation_blocked_openai_chat_stays_a_bad_request(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        return Reply(status=400, body=json.dumps(_MODERATION_BLOCKED).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        blocked: Final = scenario.model(
            model="openai/gpt-6.2-sol", api_base=f"{wire.url}/blocked", api_key="synthetic-openai-key"
        )
        fallback: Final = scenario.model(
            model="openai/gpt-6.2-sol", api_base=f"{wire.url}/fallback", api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": blocked,
                "messages": [{"role": "user", "content": "describe a lighthouse"}],
                "content_policy_fallbacks": [{blocked: [fallback]}],
            },
        )
        assert response.status_code == 400, response.text
        assert [request.target for request in wire.drain()] == ["/blocked/chat/completions"]


def test_moderation_blocked_image_edit_on_hosted_vllm_stays_a_bad_request(gateway: Gateway) -> None:
    with wire_server(_blocked_then_image) as wire, gateway.scenario() as scenario:
        blocked: Final = scenario.model(
            model="hosted_vllm/served-image-model", api_base=f"{wire.url}/blocked", api_key="synthetic-vllm-key"
        )
        fallback: Final = scenario.model(
            model="hosted_vllm/served-image-model", api_base=f"{wire.url}/fallback", api_key="synthetic-vllm-key"
        )
        key: Final = scenario.key(router_settings={"content_policy_fallbacks": [{blocked: [fallback]}]})
        response: Final = gateway.request_multipart(
            "/v1/images/edits",
            {"model": blocked, "prompt": "turn the lighthouse green"},
            {"image": ("lighthouse.png", _PNG, "image/png")},
            key=key,
        )
        assert response.status_code == 400, response.text
        assert [request.target for request in wire.drain()] == ["/blocked/v1/images/edits"]
