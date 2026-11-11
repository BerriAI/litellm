import json
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as endpoints


class _RedactPrompt(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Mapping[str, object],
        call_type: CallTypesLiteral,
    ) -> Mapping[str, object]:
        return {**data, "prompt": "[REDACTED]"}


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix,domain", [("tripo", "openapi.tripo3d.ai"), ("cn.tripo", "openapi.tripo3d.com")])
async def test_tripo_forwards_body_returned_by_pre_call_hook(
    monkeypatch: pytest.MonkeyPatch, prefix: str, domain: str
) -> None:
    import httpx
    from starlette.requests import Request
    from starlette.types import Message

    import litellm
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import resolve_pass_through_request_timeout
    from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import PassthroughEndpointRouter
    from litellm.types.llms.custom_http import httpxSpecialProvider

    from litellm.proxy.utils import ProxyLogging

    monkeypatch.setattr(litellm, "callbacks", [_RedactPrompt()])
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=UserApiKeyCache()))
    router: Final = litellm.Router(model_list=[{
        "model_name": "tripo-test",
        "litellm_params": {
            "model": "tripo/task", "custom_llm_provider": "tripo", "api_key": "test-key",
            "api_base": f"https://{domain}/v3", "use_in_pass_through": True,
        },
    }])
    original: Final = b'{"prompt":"secret"}'
    sanitized: Final = {"prompt": "[REDACTED]"}

    async def receive() -> Message:
        return {"type": "http.request", "body": original, "more_body": False}

    def upstream(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == sanitized
        return httpx.Response(200, json={"code": 1}, request=request)

    handler: Final = get_async_httpx_client(
        httpxSpecialProvider.PassThroughEndpoint,
        params={"timeout": resolve_pass_through_request_timeout(None)},
    )
    cache_key: Final = next(key for key, value in litellm.in_memory_llm_clients_cache.cache_dict.items() if value is handler)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        monkeypatch.setitem(litellm.in_memory_llm_clients_cache.cache_dict, cache_key, SimpleNamespace(client=client))
        request: Final = Request({
            "type": "http", "method": "POST", "path": f"/{prefix}/v3/generation/text-to-model",
            "headers": [(b"content-type", b"application/json")], "query_string": b"",
            "scheme": "http", "server": ("test", 80),
        }, receive=receive)
        response: Final = await endpoints.tripo_proxy_route(
            endpoint="v3/generation/text-to-model", request=request,
            user_api_key_dict=UserAPIKeyAuth(models=["tripo-test"]),
            credentials_router=PassthroughEndpointRouter(llm_router_getter=lambda: router),
        )
        assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix,domain", [("tripo", "openapi.tripo3d.ai"), ("cn.tripo", "openapi.tripo3d.com")])
@pytest.mark.parametrize("scope", ["key", "team", "project"])
@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize("body", [
    b'{"prompt":"a chair"}',
    b'{"model":"unrelated","prompt":"a chair"}',
    b'{"prompt":"a chair","stream":false}',
])
async def test_tripo_authorizes_selected_deployment_before_forwarding(
    monkeypatch: pytest.MonkeyPatch, prefix: str, domain: str, scope: str, allowed: bool, body: bytes
) -> None:
    from typing import Final

    import httpx
    from starlette.requests import Request
    from starlette.types import Message

    import litellm
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.proxy import proxy_server
    from litellm.proxy._types import LiteLLM_ProjectTableCachedObj, ModelAccessDeniedProxyException, UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, project_cache_key
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import resolve_pass_through_request_timeout
    from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import PassthroughEndpointRouter
    from litellm.types.llms.custom_http import httpxSpecialProvider

    content_type: Final = "application/json; charset=utf-8"
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    router: Final = litellm.Router(model_list=[{
        "model_name": f"tripo-{region}",
        "litellm_params": {
            "model": "tripo/task", "custom_llm_provider": "tripo", "api_key": f"{region}-key",
            "api_base": f"https://openapi.tripo3d{suffix}/v3",
            "use_in_pass_through": True,
        },
    } for region, suffix in (("cn.tripo", ".com"), ("tripo", ".ai"))] + [{
        "model_name": "unrelated",
        "litellm_params": {
            "model": "tripo/task", "custom_llm_provider": "tripo", "api_key": "other-deployment-key",
            "api_base": f"https://{domain}/v3", "use_in_pass_through": True,
        },
    }])

    granted: Final = [f"tripo-{prefix}"] if allowed else ["unrelated"]
    cache: Final = UserApiKeyCache()
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace())
    if scope == "project":
        await cache.async_set_cache(
            key=project_cache_key("tripo-project"),
            value=LiteLLM_ProjectTableCachedObj(project_id="tripo-project", models=granted),
            model_type=LiteLLM_ProjectTableCachedObj,
        )
    token: Final = UserAPIKeyAuth(
        api_key="hashed",
        models=granted if scope == "key" else [f"tripo-{prefix}"],
        team_id="tripo-team" if scope == "team" else None,
        team_models=granted if scope == "team" else [],
        project_id="tripo-project" if scope == "project" else None,
    )

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    def upstream(request: httpx.Request) -> httpx.Response:
        assert allowed, "Unauthorized request reached upstream"
        assert request.headers["content-type"] == content_type
        assert request.headers["authorization"] == f"Bearer {prefix}-key"
        assert request.content == body
        assert str(request.url) == f"https://{domain}/v3/generation/text-to-model"
        return httpx.Response(200, json={"code": 0, "data": {}}, request=request)

    handler: Final = get_async_httpx_client(
        httpxSpecialProvider.PassThroughEndpoint,
        params={"timeout": resolve_pass_through_request_timeout(None)},
    )
    client_cache: Final = litellm.in_memory_llm_clients_cache.cache_dict
    cache_key: Final = next(key for key, value in client_cache.items() if value is handler)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        monkeypatch.setitem(client_cache, cache_key, SimpleNamespace(client=client))
        request: Final = Request({
            "type": "http", "method": "POST", "path": f"/{prefix}/v3/generation/text-to-model",
            "headers": [(b"content-type", content_type.encode()), (b"authorization", b"Bearer proxy-key")],
            "query_string": b"", "scheme": "http", "server": ("test", 80),
        }, receive=receive)
        if not allowed:
            with pytest.raises(ModelAccessDeniedProxyException):
                await endpoints.tripo_proxy_route(
                    endpoint="v3/generation/text-to-model", request=request,
                    user_api_key_dict=token,
                    credentials_router=PassthroughEndpointRouter(llm_router_getter=lambda: router),
                )
            return
        result: Final = await endpoints.tripo_proxy_route(
            endpoint="v3/generation/text-to-model", request=request,
            user_api_key_dict=token,
            credentials_router=PassthroughEndpointRouter(llm_router_getter=lambda: router),
        )
        assert result.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["tripo", "cn.tripo"])
@pytest.mark.parametrize("stream", [True, 1, "true", "false"])
async def test_tripo_rejects_streaming_before_forwarding(
    monkeypatch: pytest.MonkeyPatch, prefix: str, stream: bool | int | str
) -> None:
    import json
    from typing import Final

    import httpx
    from fastapi import HTTPException
    from starlette.requests import Request
    from starlette.types import Message

    import litellm
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import resolve_pass_through_request_timeout
    from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import PassthroughEndpointRouter
    from litellm.types.llms.custom_http import httpxSpecialProvider

    router: Final = litellm.Router(model_list=[{
        "model_name": "tripo-stream-test",
        "litellm_params": {
            "model": "tripo/task", "custom_llm_provider": "tripo", "api_key": "test-key",
            "api_base": "https://openapi.tripo3d.com/v3" if prefix == "cn.tripo" else "https://openapi.tripo3d.ai/v3",
            "use_in_pass_through": True,
        },
    }])

    async def receive() -> Message:
        return {
            "type": "http.request",
            "body": json.dumps({"prompt": "a chair", "stream": stream}).encode(),
            "more_body": False,
        }

    request: Final = Request({
        "type": "http", "method": "POST", "path": f"/{prefix}/v3/generation/text-to-model",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"", "scheme": "http", "server": ("test", 80),
    }, receive=receive)
    handler: Final = get_async_httpx_client(
        httpxSpecialProvider.PassThroughEndpoint,
        params={"timeout": resolve_pass_through_request_timeout(None)},
    )
    client_cache: Final = litellm.in_memory_llm_clients_cache.cache_dict
    cache_key: Final = next(key for key, value in client_cache.items() if value is handler)

    def upstream(request: httpx.Request) -> httpx.Response:
        pytest.fail("Streaming request reached upstream")

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        monkeypatch.setitem(client_cache, cache_key, SimpleNamespace(client=client))
        with pytest.raises(HTTPException) as rejected:
            await endpoints.tripo_proxy_route(
                endpoint="v3/generation/text-to-model", request=request,
                user_api_key_dict=UserAPIKeyAuth(models=["tripo-stream-test"]),
                credentials_router=PassthroughEndpointRouter(llm_router_getter=lambda: router),
            )
    assert rejected.value.status_code == 400
    assert rejected.value.detail == "Streaming is not supported for Tripo requests"
