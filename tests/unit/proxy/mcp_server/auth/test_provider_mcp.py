import asyncio
from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LiteLLM_TeamTable, ProxyException, UserAPIKeyAuth
from litellm.proxy.mcp_server import mcp_server_manager
from litellm.proxy.mcp_server.auth.provider_mcp import authorize_provider_mcp
from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
from litellm.responses.mcp.request_context import authorize_provider_mcp_request
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.utils import async_pre_call_deployment_hook


def _openai(tools: object = ("search",), url: str = "https://native.example/mcp") -> dict[str, object]:
    return {"tools": [{"type": "mcp", "server_label": "arbitrary", "server_url": url, "allowed_tools": tools}]}


def _anthropic(legacy: bool = False) -> dict[str, object]:
    return {
        "mcp_servers": [
            {
                "type": "url",
                "name": "docs",
                "url": "https://native.example/mcp",
                **({"tool_configuration": {"allowed_tools": ["search"]}} if legacy else {}),
            }
        ],
        **(
            {}
            if legacy
            else {
                "tools": [
                    {
                        "type": "mcp_toolset",
                        "mcp_server_name": "docs",
                        "default_config": {"enabled": False},
                        "configs": {"search": {"enabled": True}, "delete": {"enabled": False}},
                    }
                ]
            }
        ),
    }


@pytest.fixture
async def native_auth(monkeypatch: pytest.MonkeyPatch) -> UserAPIKeyAuth:
    manager: Final = mcp_server_manager.MCPServerManager()
    cache: Final = DualCache()
    monkeypatch.setattr(mcp_server_manager, "global_mcp_server_manager", manager)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    await manager.load_servers_from_config(
        {
            "docs": {
                "server_id": "native-id",
                "alias": "native-alias",
                "execution_mode": "provider",
                "url": "https://native.example/mcp",
                "allowed_tools": ["search", "fetch"],
            }
        }
    )
    await cache.async_set_cache(
        key="team_id:team",
        value=LiteLLM_TeamTable(
            team_id="team",
            object_permission=LiteLLM_ObjectPermissionTable(
                object_permission_id="team-grant",
                mcp_servers=["docs"],
                mcp_tool_permissions={"docs": ["search"]},
            ),
        ),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace())
    monkeypatch.setattr(proxy_server, "general_settings", {"supported_db_objects": []})
    return UserAPIKeyAuth(
        api_key="test-key",
        team_id="team",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="key-grant",
            mcp_servers=["native-alias"],
            mcp_tool_permissions={"native-alias": ["search", "fetch"]},
        ),
    )


@pytest.mark.parametrize(
    "body", [_openai(), _openai({"tool_names": ["search"], "read_only": True}), _anthropic(), _anthropic(True)]
)
async def test_allowed_native_attachment_is_unchanged(body: Mapping[str, object], native_auth: UserAPIKeyAuth) -> None:
    before: Final = deepcopy(body)
    await authorize_provider_mcp(body, native_auth)
    assert body == before
    assert mcp_server_manager.global_mcp_server_manager.get_registry() == {}


@pytest.mark.parametrize(
    "body",
    [
        _openai(["fetch"]),
        _openai(["delete"]),
        _openai(None),
        _openai({"read_only": True}),
        _openai(["*"]),
        _openai(["search"], "https://unregistered.example/mcp"),
        _openai(["search"], "http://native.example/mcp"),
        _openai(["search"], "https://native.example/mcp?endpoint=other"),
        _openai(["search"], "https://native.example/mcp/"),
        _openai(["search"], "https://native.example@other.example/mcp"),
        {"mcp_servers": [{"type": "url", "name": "docs", "url": "https://native.example/mcp"}]},
        {
            "mcp_servers": [
                {
                    "type": "url",
                    "name": "docs",
                    "url": "https://native.example/mcp",
                    "tool_configuration": {"allowed_tools": []},
                }
            ]
        },
        {
            **_anthropic(),
            "tools": [{"type": "mcp_toolset", "mcp_server_name": "docs", "configs": {"search": {"enabled": True}}}],
        },
        {
            **_anthropic(),
            "tools": [{"type": "mcp_toolset", "mcp_server_name": "missing", "default_config": {"enabled": False}}],
        },
        {"tools": [{"type": "mcp", "connector_id": "connector_example", "allowed_tools": ["search"]}]},
    ],
)
async def test_native_attachment_denies_unbounded_or_unauthorized_scope(
    body: Mapping[str, object], native_auth: UserAPIKeyAuth
) -> None:
    with pytest.raises(ProxyException) as denied:
        await authorize_provider_mcp(body, native_auth)
    assert denied.value.code == "403"


@pytest.mark.parametrize("servers", [["no-mcp-servers"], ["unrelated"]])
async def test_server_denial_cannot_be_bypassed_by_tool_name(servers: list[str], native_auth: UserAPIKeyAuth) -> None:
    auth: Final = native_auth.model_copy(
        update={
            "object_permission": LiteLLM_ObjectPermissionTable(
                object_permission_id="denied",
                mcp_servers=servers,
            )
        }
    )
    with pytest.raises(ProxyException) as denied:
        await authorize_provider_mcp(_openai(), auth)
    assert denied.value.code == "403"


async def test_canonical_host_and_default_port_match(native_auth: UserAPIKeyAuth) -> None:
    body: Final = _openai(["search"], "https://NATIVE.example:443/mcp")
    await authorize_provider_mcp(body, native_auth)
    assert body == _openai(["search"], "https://NATIVE.example:443/mcp")


async def test_extra_body_cannot_replace_authorized_attachment(native_auth: UserAPIKeyAuth) -> None:
    kwargs: Final = {
        **_openai(),
        "extra_body": _openai(["delete"]),
        "litellm_metadata": {"user_api_key_auth": native_auth},
    }
    with pytest.raises(ProxyException) as denied:
        await async_pre_call_deployment_hook(kwargs, "aresponses")
    assert denied.value.code == "403"


@pytest.mark.parametrize("provider_bound", [False, True])
async def test_registration_revocation_is_seen_on_next_admission(
    native_auth: UserAPIKeyAuth, provider_bound: bool
) -> None:
    await authorize_provider_mcp(_openai(), native_auth, provider_bound=provider_bound)
    await mcp_server_manager.global_mcp_server_manager.load_servers_from_config(
        {
            "docs": {
                "server_id": "native-id",
                "execution_mode": "provider",
                "url": "https://native.example/mcp",
                "allowed_tools": [],
            }
        }
    )
    with pytest.raises(ProxyException) as denied:
        await authorize_provider_mcp(_openai(), native_auth, provider_bound=provider_bound)
    assert denied.value.code == "403"


async def test_sdk_without_proxy_identity_preserves_native_passthrough(native_auth: UserAPIKeyAuth) -> None:
    body: Final = _openai(["anything"], "https://unregistered.example/mcp")
    await authorize_provider_mcp_request(body, {})
    assert body == _openai(["anything"], "https://unregistered.example/mcp")


async def test_provider_url_wins_over_gateway_path_shortcut(native_auth: UserAPIKeyAuth) -> None:
    manager: Final = mcp_server_manager.global_mcp_server_manager
    await manager.load_servers_from_config(
        {
            "native": {
                "server_id": "provider-path",
                "execution_mode": "provider",
                "url": "https://native.example/mcp/gateway",
                "allowed_tools": ["search"],
            }
        }
    )

    async def gateway_names(names: object) -> frozenset[str]:
        return frozenset({"gateway"})

    tools: Final = [{"type": "mcp", "server_url": "https://native.example/mcp/gateway", "allowed_tools": ["search"]}]
    assert await LiteLLM_Proxy_MCP_Handler.routes_through_gateway(tools, gateway_names) == (False,)
    assert await LiteLLM_Proxy_MCP_Handler.split_mcp_tools(tools, gateway_names) == ([], tools)


async def test_explicit_gateway_references_retain_existing_path(native_auth: UserAPIKeyAuth) -> None:
    tools: Final = [{"type": "mcp", "server_url": "litellm_proxy/mcp/gateway"}]
    await authorize_provider_mcp({"tools": tools}, native_auth)
    assert await LiteLLM_Proxy_MCP_Handler.split_mcp_tools(tools) == (tools, [])


@pytest.mark.parametrize("api", ["aresponses", "anthropic_messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("configured", [False, True])
async def test_denied_native_request_never_reaches_provider(
    native_auth: UserAPIKeyAuth,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    api: str,
    stream: bool,
    configured: bool,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post(url__regex=r"https://native-provider.example/.*").respond(500)
    attachment: Final = (
        _openai(["delete"])
        if api == "aresponses"
        else {
            "mcp_servers": [
                {
                    "type": "url",
                    "name": "docs",
                    "url": "https://unregistered.example/mcp",
                    "tool_configuration": {"allowed_tools": ["search"]},
                }
            ],
        }
    )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "native-model",
                "litellm_params": {
                    "model": "openai/gpt-5.6-luna" if api == "aresponses" else "anthropic/claude-haiku-5-5",
                    "api_key": "test-provider-key",
                    "api_base": "https://native-provider.example/v1",
                    **(attachment if configured else {}),
                },
            }
        ],
        num_retries=0,
    )
    request: Final = {
        "model": "native-model",
        "stream": stream,
        "litellm_metadata": {"user_api_key_auth": native_auth},
        **({} if configured else attachment),
    }
    with pytest.raises(Exception) as denied:
        if api == "aresponses":
            await router.aresponses(input="Use the tool", **request)
        else:
            await router.anthropic_messages(
                messages=[{"role": "user", "content": "Use the tool"}], max_tokens=32, **request
            )
    assert str(getattr(denied.value, "status_code", getattr(denied.value, "code", ""))) == "403"
    assert upstream.call_count == 0


@pytest.mark.parametrize("api", ["aresponses", "anthropic_messages"])
async def test_allowed_native_request_reaches_provider_unchanged(
    native_auth: UserAPIKeyAuth,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    api: str,
) -> None:
    import json

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    attachment: Final = _openai(["search"]) if api == "aresponses" else _anthropic()
    response: Final = (
        {
            "id": "resp_native",
            "object": "response",
            "created_at": 1,
            "model": "gpt-5.6-luna",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        if api == "aresponses"
        else {
            "id": "msg_native",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    upstream: Final = respx_mock.post(url__regex=r"https://native-provider.example/.*").respond(200, json=response)
    request: Final = {
        "api_key": "test-provider-key",
        "api_base": "https://native-provider.example/v1",
        "litellm_metadata": {"user_api_key_auth": native_auth},
        **attachment,
    }
    if api == "aresponses":
        await litellm.aresponses(model="openai/gpt-5.6-luna", input="Use the tool", **request)
    else:
        await litellm.anthropic_messages(
            model="anthropic/claude-haiku-5-5",
            messages=[{"role": "user", "content": "Use the tool"}],
            max_tokens=32,
            **request,
        )
    sent: Final = json.loads(upstream.calls.last.request.content)
    assert {key: sent[key] for key in attachment} == attachment
    assert upstream.call_count == 1


async def test_cached_response_does_not_bypass_revoked_mcp_permission(
    native_auth: UserAPIKeyAuth,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type="local"))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    upstream: Final = respx_mock.post(url__regex=r"https://native-provider.example/.*").respond(
        200,
        json={
            "id": "resp_cached",
            "object": "response",
            "created_at": 1,
            "model": "gpt-5.6-luna",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_cached",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "tool result", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    )
    request: Final = {
        "model": "openai/gpt-5.6-luna",
        "input": "Use the tool",
        "caching": True,
        "api_key": "test-provider-key",
        "api_base": "https://native-provider.example/v1",
        **_openai(["search"]),
    }
    await litellm.aresponses(**request, litellm_metadata={"user_api_key_auth": native_auth})
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    await litellm.aresponses(**request, litellm_metadata={"user_api_key_auth": native_auth})
    assert upstream.call_count == 1
    denied_auth: Final = native_auth.model_copy(
        update={
            "object_permission": LiteLLM_ObjectPermissionTable(
                object_permission_id="revoked",
                mcp_servers=["no-mcp-servers"],
            )
        }
    )
    with pytest.raises(ProxyException) as denied:
        await litellm.aresponses(**request, litellm_metadata={"user_api_key_auth": denied_auth})
    assert denied.value.code == "403"
    assert upstream.call_count == 1


class _LateMCPAttachment(CustomLogger):
    async def async_pre_request_hook(self, model: str, messages: list[object], kwargs: dict[str, object]) -> dict:
        return {
            **kwargs,
            "mcp_servers": [
                {
                    "type": "url",
                    "name": "docs",
                    "url": "https://unregistered.example/mcp",
                    "tool_configuration": {"allowed_tools": ["search"]},
                }
            ],
            "tools": [],
        }


async def test_final_body_check_blocks_attachment_added_after_deployment_hook(
    native_auth: UserAPIKeyAuth,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_LateMCPAttachment()])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    upstream: Final = respx_mock.post(url__regex=r"https://native-provider.example/.*").respond(500)
    with pytest.raises(Exception) as denied:
        await litellm.anthropic_messages(
            model="anthropic/claude-haiku-5-5",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=32,
            api_key="test-provider-key",
            api_base="https://native-provider.example/v1",
            litellm_metadata={"user_api_key_auth": native_auth},
        )
    assert str(getattr(denied.value, "status_code", getattr(denied.value, "code", ""))) == "403"
    assert upstream.call_count == 0


async def test_gateway_registration_does_not_authorize_provider_execution(native_auth: UserAPIKeyAuth) -> None:
    manager: Final = mcp_server_manager.global_mcp_server_manager
    manager.catalog.publish_config_registrations({})
    manager.config_mcp_servers = {
        "gateway": MCPServer(
            server_id="gateway",
            name="gateway",
            server_name="gateway",
            url="https://native.example/mcp",
            transport=MCPTransport.http,
            allowed_tools=["search"],
        )
    }
    auth: Final = native_auth.model_copy(
        update={
            "team_id": None,
            "object_permission": LiteLLM_ObjectPermissionTable(
                object_permission_id="gateway-grant",
                mcp_servers=["gateway"],
            ),
        }
    )
    with pytest.raises(ProxyException) as denied:
        await authorize_provider_mcp(_openai(), auth)
    assert denied.value.code == "403"


async def test_connector_id_cannot_override_a_registered_url(native_auth: UserAPIKeyAuth) -> None:
    with pytest.raises(ProxyException) as denied:
        await authorize_provider_mcp(
            {
                "tools": [
                    {
                        "type": "mcp",
                        "server_url": "https://native.example/mcp",
                        "connector_id": "connector_other",
                        "allowed_tools": ["search"],
                    }
                ]
            },
            native_auth,
        )
    assert denied.value.code == "403"
