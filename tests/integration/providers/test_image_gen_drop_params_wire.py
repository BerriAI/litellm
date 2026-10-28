import json
import threading
from types import MappingProxyType
from typing import Final, Literal

import httpx
import openai
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROMPT: Final = "a transparent lighthouse at dawn"


@pytest.mark.asyncio
@pytest.mark.parametrize("router_retries", (0, 1))
@pytest.mark.parametrize("sdk_retries", (0, 5))
@pytest.mark.parametrize("outcome", ("503", "timeout", "success"))
async def test_proxy_image_retry_attempts_are_owned_only_by_router(
    monkeypatch: pytest.MonkeyPatch,
    router_retries: int,
    sdk_retries: int,
    outcome: Literal["503", "timeout", "success"],
) -> None:
    from litellm.proxy import proxy_server

    release: Final = threading.Event()

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/images/generations"
        assert request.headers["authorization"] == "Bearer synthetic-image-key"
        assert _JSON_OBJECT.validate_json(request.body) == {"model": "gpt-image-1-mini", "prompt": "a red circle"}
        if outcome == "timeout":
            return Reply(chunks=(b"{}",), gate_before_first=release)
        if outcome == "503":
            return Reply(
                status=503,
                headers=MappingProxyType({"retry-after-ms": "1"}),
                body=b'{"error":{"message":"scripted unavailable","type":"server_error"}}',
            )
        return Reply(body=b'{"created":1700000000,"data":[{"b64_json":"aW1n"}]}')

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "num_retries", None)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    with wire_server(respond) as wire:
        router: Final = litellm.Router(
            model_list=[
                {
                    "model_name": "image-retries",
                    "litellm_params": {
                        "model": "openai/gpt-image-1-mini",
                        "api_base": f"{wire.url}/v1",
                        "api_key": "synthetic-image-key",
                        "max_retries": sdk_retries,
                        "timeout": 0.1,
                    },
                }
            ],
            num_retries=router_retries,
            disable_cooldowns=True,
        )
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(proxy_server, "master_key", None)
        monkeypatch.setattr(proxy_server, "prisma_client", None)
        monkeypatch.setattr(proxy_server, "general_settings", {})
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://proxy.test"
            ) as client:
                response: Final = await client.post(
                    "/v1/images/generations",
                    json={"model": "image-retries", "prompt": "a red circle", "max_retries": sdk_retries},
                )
        finally:
            release.set()
            litellm.in_memory_llm_clients_cache.flush_cache()
        requests: Final = wire.drain()
    status: Final = {"503": 503, "timeout": 408, "success": 200}[outcome]
    attempts: Final = 1 if outcome == "success" else router_retries + 1
    assert response.status_code == status, response.text
    assert len(requests) == attempts, tuple(request.headers.get("x-stainless-retry-count") for request in requests)
    if outcome == "success":
        data: Final = TypeAdapter(tuple[dict[str, JsonValue], ...]).validate_python(
            _JSON_OBJECT.validate_json(response.content)["data"]
        )
        assert data[0]["b64_json"] == "aW1n"


def test_image_generation_additional_drop_params_reaches_provider_body(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/images/generations"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert "style" not in body, body
        assert body["model"] == "dall-e-3"
        assert body["prompt"] == "a scripted cat"
        assert body["size"] == "1024x1024"
        return Reply(
            body=json.dumps(
                {
                    "created": 1700000000,
                    "data": [{"b64_json": "aW1n", "revised_prompt": None, "url": None}],
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-3",
            api_base=wire.url,
            api_key="synthetic-image-key",
            additional_drop_params=["style"],
        )
        response: Final = gateway.client.post(
            "/v1/images/generations",
            json={
                "model": model,
                "prompt": "a scripted cat",
                "size": "1024x1024",
                "style": "vivid",
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=30,
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"][0]["b64_json"] == "aW1n"
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/images/generations")]


def _openai_client(gateway: Gateway, model: str, alias: str) -> openai.OpenAI:
    base: Final = str(gateway.client.base_url)
    match alias:
        case "v1":
            return openai.OpenAI(base_url=f"{base}/v1", api_key=gateway.key, max_retries=0)
        case "unversioned":
            return openai.OpenAI(base_url=base, api_key=gateway.key, max_retries=0)
        case "azure_deployment":
            return openai.AzureOpenAI(
                azure_endpoint=base,
                azure_deployment=model,
                api_version="2025-04-01-preview",
                api_key=gateway.key,
                max_retries=0,
            )
    raise AssertionError(alias)


@pytest.mark.parametrize("alias", ("v1", "unversioned", "azure_deployment"))
def test_gpt_image_generation_forwards_every_sdk_field_once_with_integer_n(gateway: Gateway, alias: str) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/images/generations"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": "gpt-image-1",
            "prompt": _PROMPT,
            "n": 2,
            "size": "1024x1536",
            "quality": "high",
            "background": "transparent",
            "output_format": "webp",
            "moderation": "low",
            "partial_images": 0,
        }
        return Reply(
            body=json.dumps(
                {
                    "created": 1700000000,
                    "data": [{"b64_json": "aW1nLTE="}, {"b64_json": "aW1nLTI="}],
                    "background": "transparent",
                    "output_format": "webp",
                    "quality": "high",
                    "size": "1024x1536",
                    "usage": {
                        "input_tokens": 12,
                        "input_tokens_details": {"text_tokens": 12, "image_tokens": 0},
                        "output_tokens": 4160,
                        "total_tokens": 4172,
                    },
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-image-1",
            api_base=wire.url,
            api_key="synthetic-openai-key",
        )
        client: Final = _openai_client(gateway, model, alias)
        result: Final = client.images.generate(
            model=model,
            prompt=_PROMPT,
            n=2,
            size="1024x1536",
            quality="high",
            background="transparent",
            output_format="webp",
            moderation="low",
            extra_body={"partial_images": 0},
        )
        assert [image.b64_json for image in result.data] == ["aW1nLTE=", "aW1nLTI="]
        assert result.usage is not None
        assert (
            result.usage.input_tokens,
            result.usage.output_tokens,
            result.usage.total_tokens,
        ) == (12, 4160, 4172)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/images/generations")]
