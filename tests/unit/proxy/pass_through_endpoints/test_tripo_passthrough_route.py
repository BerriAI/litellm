from types import SimpleNamespace

import pytest

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as endpoints


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix,domain", [("tripo", "openapi.tripo3d.ai"), ("cn.tripo", "openapi.tripo3d.com")])
@pytest.mark.parametrize("scope", ["key", "team", "project"])
@pytest.mark.parametrize("allowed", [False, True])
@pytest.mark.parametrize("body", [b'{"prompt":"a chair"}', b'{"model":"unrelated","prompt":"a chair"}'])
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
