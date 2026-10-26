from collections.abc import Mapping
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai_like.model_info import (
    MODEL_INFO_REFRESH_SECONDS,
    get_openai_compatible_model_info,
)


@pytest.mark.parametrize(
    ("card", "expected"),
    (
        ({"max_model_len": 8192}, {"context_window": 8192}),
        (
            {"context_length": 4096, "max_output_tokens": 1024},
            {"context_window": 4096, "max_output_tokens": 1024},
        ),
        (
            {"max_model_len": 4096, "max_input_tokens": 2048, "max_output_tokens": 8192},
            {"context_window": 4096, "max_input_tokens": 2048, "max_output_tokens": 8192},
        ),
        ({"max_input_tokens": 2048}, {"max_input_tokens": 2048}),
        ({"max_output_tokens": 1024}, {"max_output_tokens": 1024}),
        ({"max_model_len": True, "max_output_tokens": -1}, {}),
        ({"max_model_len": "8192", "max_input_tokens": 0, "max_output_tokens": 1.5}, {}),
        ({}, {}),
    ),
)
async def test_discovers_only_valid_advertised_limits(
    card: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tenant/v1/models"
        assert request.headers["authorization"] == "Bearer local-key"
        return httpx.Response(200, json={"data": [{"id": "org/model", **card}]})

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        cache: Final = InMemoryCache()
        result: Final = await get_openai_compatible_model_info(
            model="org/model",
            api_base="https://backend.test/tenant/v1/",
            headers={"Authorization": "Bearer local-key"},
            client=handler,
            cache=cache,
        )
        assert result == expected
        assert (
            await get_openai_compatible_model_info(
                model="missing",
                api_base="https://backend.test/tenant/v1/",
                headers={"Authorization": "Bearer local-key"},
                client=handler,
                cache=cache,
            )
            == {}
        )


async def test_cache_is_scoped_to_endpoint_and_authentication_and_expires() -> None:
    clock: Final = Mock(return_value=0)
    responder: Final = Mock(
        side_effect=(
            httpx.Response(
                200, json={"data": [{"id": "first", "max_model_len": 1024}, {"id": "second", "max_model_len": 2048}]}
            ),
            httpx.Response(200, json={"data": [{"id": "first", "max_model_len": 4096}]}),
            httpx.Response(200, json={"data": [{"id": "first", "max_model_len": 8192}]}),
            httpx.Response(200, json={"data": [{"id": "first", "max_model_len": 16384}]}),
        )
    )
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        handler.client = client
        cache: Final = InMemoryCache(clock=clock)

        async def lookup(model: str = "first", host: str = "one.test", key: str = "one") -> Mapping[str, object]:
            return await get_openai_compatible_model_info(
                model=model, api_base=f"https://{host}", headers={"Authorization": key}, client=handler, cache=cache
            )

        assert (await lookup())["context_window"] == 1024
        assert (await lookup("second"))["context_window"] == 2048
        assert responder.call_count == 1
        assert (await lookup(key="two"))["context_window"] == 4096
        assert (await lookup(host="two.test"))["context_window"] == 8192
        clock.return_value = MODEL_INFO_REFRESH_SECONDS + 1
        assert (await lookup())["context_window"] == 16384
        assert responder.call_count == 4


@pytest.mark.parametrize(
    "response",
    (
        httpx.Response(404),
        httpx.Response(401),
        httpx.Response(302, headers={"location": "https://elsewhere.test"}),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"data": None}),
        httpx.ReadTimeout("backend unavailable"),
    ),
)
async def test_unavailable_metadata_is_best_effort_and_negative_cached(
    response: httpx.Response | Exception,
) -> None:
    responder: Final = Mock(side_effect=response if isinstance(response, Exception) else None, return_value=response)
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder), follow_redirects=True) as client:
        handler.client = client
        cache: Final = InMemoryCache()
        for _ in range(2):
            assert (
                await get_openai_compatible_model_info(
                    model="model", api_base="https://backend.test", headers={}, client=handler, cache=cache
                )
                == {}
            )
        assert responder.call_count == 1


@pytest.mark.parametrize(
    ("provider", "card", "expected"),
    (
        (
            "vercel_ai_gateway",
            {
                "context_window": 1200,
                "max_tokens": 200,
                "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                "supported_parameters": ["tools", "reasoning"],
                "reasoning_options": [{"type": "effort", "values": ["low", "xhigh", "max"]}],
            },
            {
                "context_window": 1200,
                "max_output_tokens": 200,
                "supported_modalities": ["text", "image", "pdf"],
                "supported_output_modalities": ["text"],
                "supports_function_calling": True,
                "supports_reasoning": True,
                "reasoning_effort_levels": ["low", "xhigh", "max"],
            },
        ),
        (
            "openrouter",
            {
                "context_length": 1200,
                "top_provider": {"context_length": 1100, "max_completion_tokens": 200},
                "architecture": {"input_modalities": ["text", "file"], "output_modalities": ["text"]},
                "supported_parameters": ["tools", "reasoning_effort"],
                "reasoning": {"supported_efforts": ["low", "max"], "default_effort": "low"},
            },
            {
                "context_window": 1100,
                "max_output_tokens": 200,
                "supported_modalities": ["text", "file"],
                "supported_output_modalities": ["text"],
                "supports_function_calling": True,
                "supports_reasoning": True,
                "reasoning_effort_levels": ["low", "max"],
                "default_reasoning_effort": "low",
            },
        ),
        (
            "openai",
            {"max_tokens": 200, "supported_parameters": []},
            {"supports_function_calling": False, "supports_reasoning": False},
        ),
    ),
)
async def test_normalizes_supplier_metadata_without_inventing_input_limit(provider, card, expected):
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{"id": "selected", **card}]}))
    ) as client:
        handler.client = client
        result: Final = await get_openai_compatible_model_info(
            model="selected",
            provider=provider,
            api_base="https://supplier.test/v1",
            headers={},
            client=handler,
            cache=InMemoryCache(),
        )
        assert result == expected


async def test_full_inventory_preserves_model_kind_without_fabricating_embedding_output_cap() -> None:
    from litellm.llms.openai_like.model_info import get_openai_compatible_model_inventory
    from litellm.types.proxy.model_inventory import SupplierModelInventory

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "embedding", "type": "embedding", "context_window": 4096, "max_tokens": 4096},
                        {"id": "language", "type": "language", "max_tokens": 1024},
                    ]
                },
            )
        )
    ) as client:
        handler.client = client
        result: Final = await get_openai_compatible_model_inventory(
            api_base="https://supplier.test/v1",
            provider="vercel_ai_gateway",
            headers={},
            client=handler,
            cache=InMemoryCache(),
        )
        assert isinstance(result, SupplierModelInventory)
        assert result.models["embedding"] == {"mode": "embedding", "context_window": 4096}
        assert result.models["language"] == {"mode": "chat", "max_output_tokens": 1024}
