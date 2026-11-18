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
        ({"max_model_len": 8192}, {"max_tokens": 8192, "max_input_tokens": 8192, "max_output_tokens": 8192}),
        (
            {"context_length": 4096, "max_output_tokens": 1024},
            {"max_tokens": 4096, "max_input_tokens": 4096, "max_output_tokens": 1024},
        ),
        (
            {"max_model_len": 4096, "max_input_tokens": 2048, "max_output_tokens": 8192},
            {"max_tokens": 4096, "max_input_tokens": 2048, "max_output_tokens": 4096},
        ),
        ({"max_input_tokens": 2048}, {"max_input_tokens": 2048}),
        ({"max_output_tokens": 1024}, {"max_output_tokens": 1024}),
        ({"max_model_len": True, "max_output_tokens": -1}, {}),
        ({"max_model_len": "8192", "max_input_tokens": 0, "max_output_tokens": 1.5}, {}),
        ({}, {}),
    ),
)
async def test_discovers_only_valid_advertised_limits(card: Mapping[str, object], expected: Mapping[str, int]) -> None:
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

        async def lookup(model: str = "first", host: str = "one.test", key: str = "one") -> Mapping[str, int]:
            return await get_openai_compatible_model_info(
                model=model, api_base=f"https://{host}", headers={"Authorization": key}, client=handler, cache=cache
            )

        assert (await lookup())["max_input_tokens"] == 1024
        assert (await lookup("second"))["max_input_tokens"] == 2048
        assert responder.call_count == 1
        assert (await lookup(key="two"))["max_input_tokens"] == 4096
        assert (await lookup(host="two.test"))["max_input_tokens"] == 8192
        clock.return_value = MODEL_INFO_REFRESH_SECONDS + 1
        assert (await lookup())["max_input_tokens"] == 16384
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
