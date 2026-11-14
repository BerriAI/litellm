from collections.abc import Mapping
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.lm_studio.model_info import get_lm_studio_model_info
from litellm.llms.openai_like.model_info import MODEL_INFO_REFRESH_SECONDS


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        ("org/model", 8192),
        ("instance-a", 8192),
        ("instance-b", 2048),
        ("small-first", 1024),
        ("other", 4096),
        ("model", None),
        ("missing", None),
        ("unloaded", None),
        ("incomplete", None),
    ),
)
async def test_matches_loaded_instances_without_using_theoretical_limits(model: str, expected: int | None) -> None:
    responder: Final = Mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {
                        "key": "org/model",
                        "max_context_length": 131072,
                        "loaded_instances": [
                            {"id": "instance-a", "config": {"context_length": 8192}},
                            {"id": "instance-b", "config": {"context_length": 2048}},
                        ],
                    },
                    {
                        "key": "small-first",
                        "loaded_instances": [
                            {"id": "small-a", "config": {"context_length": 1024}},
                            {"id": "big-b", "config": {"context_length": 4096}},
                        ],
                    },
                    {"key": "other", "loaded_instances": [{"id": "other", "config": {"context_length": 4096}}]},
                    {"key": "unloaded", "max_context_length": 131072, "loaded_instances": []},
                    {
                        "key": "incomplete",
                        "loaded_instances": [
                            {"id": "invalid", "config": {}},
                            {"id": "valid", "config": {"context_length": 4096}},
                        ],
                    },
                ]
            },
        )
    )
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        handler.client = client
        result: Final = await get_lm_studio_model_info(
            model=model, api_base="https://backend.test/v1", headers={}, client=handler, cache=InMemoryCache()
        )
    if model == "incomplete":
        assert result is None
        return
    assert result == (
        {}
        if expected is None
        else {"max_tokens": expected, "max_input_tokens": expected, "max_output_tokens": expected}
    )


@pytest.mark.parametrize("base_path", ("", "/v1/", "/api/v1", "/api/v0"))
@pytest.mark.parametrize("prefix", ("/tenant", "/tenant%2Fid"))
@pytest.mark.parametrize(
    "current_api_response",
    (
        httpx.Response(404),
        httpx.Response(405),
        httpx.Response(200, json={"error": "Unexpected endpoint or method. (GET /api/v1/models)"}),
    ),
)
async def test_legacy_fallback_preserves_origin_and_proxy_prefix(
    base_path: str, prefix: str, current_api_response: httpx.Response
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "backend.test"
        assert request.headers["authorization"] == "Bearer local-key"
        assert request.url.query == b"tenant=one%2Ftwo"
        assert request.url.fragment == ""
        if request.url.raw_path == f"{prefix}/api/v1/models?tenant=one%2Ftwo".encode():
            return current_api_response
        assert request.url.raw_path == f"{prefix}/api/v0/models?tenant=one%2Ftwo".encode()
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "org/model", "state": "loaded", "max_context_length": 32768, "loaded_context_length": 4096},
                    {"id": "unloaded", "state": "not-loaded", "max_context_length": 32768},
                ]
            },
        )

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        cache: Final = InMemoryCache()
        result: Final = await get_lm_studio_model_info(
            model="org/model",
            api_base=f"https://backend.test{prefix}{base_path}?tenant=one%2Ftwo#ignored",
            headers={"authorization": "Bearer local-key"},
            client=handler,
            cache=cache,
        )
        assert result == {"max_tokens": 4096, "max_input_tokens": 4096, "max_output_tokens": 4096}
        assert (
            await get_lm_studio_model_info(
                model="unloaded",
                api_base=f"https://backend.test{prefix}{base_path}?tenant=one%2Ftwo#ignored",
                headers={"authorization": "Bearer local-key"},
                client=handler,
                cache=cache,
            )
            == {}
        )


@pytest.mark.parametrize(
    ("path", "models"),
    (
        (
            "/api/v1/models",
            {
                "models": [
                    {
                        "type": "embedding",
                        "key": "org/embedder",
                        "loaded_instances": [
                            {"id": "org/embedder", "config": {"context_length": 2048}},
                            {"id": "embedder-copy", "config": {"context_length": 2048}},
                        ],
                    },
                    {
                        "type": "llm",
                        "key": "org/chat",
                        "loaded_instances": [{"id": "org/chat", "config": {"context_length": 4096}}],
                    },
                ]
            },
        ),
        (
            "/api/v0/models",
            {
                "data": [
                    {"id": "org/embedder", "type": "embeddings", "state": "loaded", "loaded_context_length": 2048},
                    {"id": "embedder-copy", "type": "embeddings", "state": "loaded", "loaded_context_length": 2048},
                    {"id": "org/chat", "type": "llm", "state": "loaded", "loaded_context_length": 4096},
                ]
            },
        ),
    ),
)
async def test_embedding_models_advertise_no_limits_since_lm_studio_truncates_their_input(
    path: str, models: Mapping[str, object]
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=models) if request.url.path == path else httpx.Response(404)

    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        cache: Final = InMemoryCache()
        limits: Final = {
            model: await get_lm_studio_model_info(
                model=model, api_base="https://backend.test/v1", headers={}, client=handler, cache=cache
            )
            for model in ("org/embedder", "embedder-copy", "org/chat")
        }
    assert limits == {
        "org/embedder": {},
        "embedder-copy": {},
        "org/chat": {"max_tokens": 4096, "max_input_tokens": 4096, "max_output_tokens": 4096},
    }


@pytest.mark.parametrize("context", (True, 0, -1, "8192", 1.5, None))
async def test_invalid_context_is_not_advertised(context: object) -> None:
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "key": "model",
                            "max_context_length": 32768,
                            "loaded_instances": [{"id": "model", "config": {"context_length": context}}],
                        },
                    ]
                },
            )
        )
    ) as client:
        handler.client = client
        assert (
            await get_lm_studio_model_info(
                model="model", api_base="https://backend.test", headers={}, client=handler, cache=InMemoryCache()
            )
            is None
        )


@pytest.mark.parametrize(
    ("response", "requests_per_attempt"),
    (
        (httpx.Response(401), 1),
        (httpx.Response(503), 1),
        (httpx.Response(404), 2),
        (httpx.Response(405), 2),
        (httpx.Response(302, headers={"location": "https://elsewhere.test"}), 1),
        (httpx.Response(200, content=b"not json"), 2),
        (httpx.Response(200, json={"models": None}), 2),
        (httpx.ReadTimeout("offline"), 1),
        (httpx.ConnectError("offline"), 1),
    ),
)
async def test_failures_are_reported_as_unknown_and_negative_cached_without_redirects_or_unbounded_fallbacks(
    response: httpx.Response | Exception,
    requests_per_attempt: int,
) -> None:
    responder: Final = Mock(side_effect=response if isinstance(response, Exception) else None, return_value=response)
    clock: Final = Mock(return_value=0)
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder), follow_redirects=True) as client:
        handler.client = client
        cache: Final = InMemoryCache(clock=clock)

        async def lookup() -> Mapping[str, int] | None:
            return await get_lm_studio_model_info(
                model="model",
                api_base="https://backend.test",
                headers={"authorization": "secret"},
                client=handler,
                cache=cache,
            )

        assert await lookup() is None
        assert await lookup() is None
        assert responder.call_count == requests_per_attempt
        clock.return_value = 61
        assert await lookup() is None
        assert responder.call_count == 2 * requests_per_attempt


async def test_server_without_the_model_loaded_is_known_to_have_no_limits() -> None:
    responder: Final = Mock(return_value=httpx.Response(200, json={"models": []}))
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        handler.client = client
        assert (
            await get_lm_studio_model_info(
                model="model", api_base="https://backend.test", headers={}, client=handler, cache=InMemoryCache()
            )
            == {}
        )
        assert responder.call_count == 1


async def test_cache_is_scoped_to_server_and_credentials_and_refreshes_context() -> None:
    clock: Final = Mock(return_value=0)
    contexts: Final = iter((8192, 4096, 2048, 1024))

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "key": "model",
                        "loaded_instances": [{"id": "instance", "config": {"context_length": next(contexts)}}],
                    }
                ]
            },
        )

    responder: Final = Mock(side_effect=respond)
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        handler.client = client
        cache: Final = InMemoryCache(clock=clock)

        async def lookup(model: str = "model", host: str = "one.test", key: str = "one") -> Mapping[str, int]:
            limits: Final = await get_lm_studio_model_info(
                model=model, api_base=f"https://{host}/v1", headers={"authorization": key}, client=handler, cache=cache
            )
            assert limits is not None
            return limits

        assert (await lookup())["max_input_tokens"] == 8192
        assert (await lookup("instance"))["max_input_tokens"] == 8192
        assert responder.call_count == 1
        assert (await lookup(key="two"))["max_input_tokens"] == 4096
        assert (await lookup(host="two.test"))["max_input_tokens"] == 2048
        clock.return_value = MODEL_INFO_REFRESH_SECONDS // 2
        assert (await lookup())["max_input_tokens"] == 1024
        assert responder.call_count == 4


@pytest.mark.parametrize("api_base", ("", "file:///tmp/server", "https://user:password@backend.test", "http://[:::]"))
async def test_invalid_or_missing_server_never_sends_credentials(api_base: str) -> None:
    responder: Final = Mock(side_effect=AssertionError("no network request expected"))
    handler: Final = AsyncHTTPHandler()
    await handler.client.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        handler.client = client
        assert (
            await get_lm_studio_model_info(
                model="model",
                api_base=api_base,
                headers={"authorization": "secret"},
                client=handler,
                cache=InMemoryCache(),
            )
            is None
        )
        assert responder.call_count == 0
