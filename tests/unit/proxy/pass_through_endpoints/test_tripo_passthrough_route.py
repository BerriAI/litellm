import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as endpoints


@pytest.mark.parametrize(
    ("route", "api_base", "api_key", "expected_target"),
    [
        (
            "/tripo/v3/generation/text-to-model",
            "https://openapi.tripo3d.ai/v3",
            "global-key",
            "https://openapi.tripo3d.ai/v3/generation/text-to-model",
        ),
        (
            "/cn.tripo/v3/generation/text-to-model",
            "https://openapi.tripo3d.com/v3",
            "cn-key",
            "https://openapi.tripo3d.com/v3/generation/text-to-model",
        ),
    ],
)
def test_tripo_route_uses_the_region_deployment_and_preserves_v3(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    api_base: str,
    api_key: str,
    expected_target: str,
) -> None:
    request = MagicMock()
    request.url.path = route
    request.scope = {"root_path": ""}
    request.method = "POST"
    request.headers = {"content-type": "application/json"}
    request.state = SimpleNamespace()
    request.body = AsyncMock(return_value=b'{"prompt":"a chair"}')
    credentials_router = MagicMock()
    credentials_router.get_credentials_with_api_base.return_value = SimpleNamespace(
        api_base=api_base,
        api_key=api_key,
    )
    forwarded = object()
    pass_through = AsyncMock(return_value=forwarded)
    monkeypatch.setattr(endpoints, "pass_through_request", pass_through)

    result = asyncio.run(
        endpoints.tripo_proxy_route(
            endpoint=route.split("/", 2)[2],
            request=request,
            user_api_key_dict=MagicMock(),
            credentials_router=credentials_router,
        )
    )

    assert result is forwarded
    pass_through.assert_awaited_once()
    assert pass_through.await_args.kwargs["target"] == expected_target
    assert pass_through.await_args.kwargs["custom_headers"] == {
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"
    }
    assert pass_through.await_args.kwargs["custom_llm_provider"] == "tripo"


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix,domain", [("tripo", "openapi.tripo3d.ai"), ("cn.tripo", "openapi.tripo3d.com")])
async def test_tripo_json_reaches_upstream_with_content_type_and_original_body(
    monkeypatch: pytest.MonkeyPatch, prefix: str, domain: str
) -> None:
    from typing import Final

    import httpx
    from starlette.requests import Request
    from starlette.types import Message

    import litellm
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import resolve_pass_through_request_timeout
    from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import PassthroughEndpointRouter
    from litellm.types.llms.custom_http import httpxSpecialProvider

    body: Final = b'{ "model": "test-model", "prompt": "a chair" }'
    content_type: Final = "application/json; charset=utf-8"
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    router: Final = litellm.Router(model_list=[{
        "model_name": f"tripo-{region}",
        "litellm_params": {
            "model": "tripo/task", "custom_llm_provider": "tripo", "api_key": f"{region}-key",
            "api_base": f"https://openapi.tripo3d{suffix}/v3",
            "use_in_pass_through": True,
        },
    } for region, suffix in (("cn.tripo", ".com"), ("tripo", ".ai"))])

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.headers["content-type"] == content_type
        assert request.headers["authorization"] == f"Bearer {prefix}-key"
        assert request.content == body
        assert str(request.url) == f"https://{domain}/v3/generation/text-to-model"
        return httpx.Response(200, json={"code": 0, "data": {}}, request=request)

    handler: Final = get_async_httpx_client(
        httpxSpecialProvider.PassThroughEndpoint,
        params={"timeout": resolve_pass_through_request_timeout(None)},
    )
    cache: Final = litellm.in_memory_llm_clients_cache.cache_dict
    cache_key: Final = next(key for key, value in cache.items() if value is handler)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        monkeypatch.setitem(cache, cache_key, SimpleNamespace(client=client))
        request: Final = Request({
            "type": "http", "method": "POST", "path": f"/{prefix}/v3/generation/text-to-model",
            "headers": [(b"content-type", content_type.encode()), (b"authorization", b"Bearer proxy-key")],
            "query_string": b"", "scheme": "http", "server": ("test", 80),
        }, receive=receive)
        result: Final = await endpoints.tripo_proxy_route(
            endpoint="v3/generation/text-to-model", request=request,
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed"),
            credentials_router=PassthroughEndpointRouter(llm_router_getter=lambda: router),
        )
        assert result.status_code == 200
