import asyncio
from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

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
    team: Final = await cache.async_get_cache(key="team_id:team")
    database: Final = SimpleNamespace(litellm_teamtable=SimpleNamespace(find_unique=AsyncMock(return_value=team)))
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=database, writer_db=database))
    monkeypatch.setattr(
        proxy_server, "general_settings", {"supported_db_objects": [], "enforce_provider_mcp_permissions": True}
    )
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
        {
            **_anthropic(),
            "tools": [{"type": "mcp_toolset", "mcp_server_name": "docs", "default_config": {"enabled": False}}] * 2,
        },
        {
            **_anthropic(),
            "mcp_servers": [{"type": "url", "name": "docs", "url": "https://native.example/mcp"}] * 2,
        },
        {**_anthropic(True), "tools": _anthropic()["tools"]},
        {
            **_anthropic(),
            "tools": [
                {
                    "type": "mcp_toolset",
                    "mcp_server_name": "docs",
                    "default_config": {"enabled": False},
                    "configs": {"search": {"enabled": "true"}},
                }
            ],
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


async def test_duplicate_url_registration_neither_shadows_nor_widens_the_granted_one(
    native_auth: UserAPIKeyAuth, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two database rows share one URL, as a database without the partial index can hold. The ungranted
    duplicate is listed last, where a one-registration-per-URL map would have picked it."""
    from litellm.models.mcp_server import LiteLLM_MCPServerTable

    granted: Final = LiteLLM_MCPServerTable(
        server_id="native-id", server_name="docs", alias="native-alias", execution_mode="provider",
        transport=None, url="https://native.example/mcp", allowed_tools=["search", "fetch"],
    )
    duplicate: Final = LiteLLM_MCPServerTable(
        server_id="duplicate-id", server_name="duplicate", execution_mode="provider",
        transport=None, url="https://native.example/mcp", allowed_tools=["delete"],
    )
    team: Final = await proxy_server.user_api_key_cache.async_get_cache(key="team_id:team")
    tables: Final = SimpleNamespace(
        litellm_mcpservertable=SimpleNamespace(find_many=AsyncMock(return_value=[granted, duplicate])),
        litellm_config=SimpleNamespace(find_unique=AsyncMock(return_value=SimpleNamespace(reload_revision=1))),
        litellm_teamtable=SimpleNamespace(find_unique=AsyncMock(return_value=team)),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=tables, writer_db=tables))
    monkeypatch.setattr(proxy_server, "general_settings", {"enforce_provider_mcp_permissions": True})
    monkeypatch.setattr(mcp_server_manager, "global_mcp_server_manager", mcp_server_manager.MCPServerManager())

    await authorize_provider_mcp(_openai(["search"]), native_auth)
    with pytest.raises(ProxyException) as denied:
        await authorize_provider_mcp(_openai(["delete"]), native_auth)
    assert denied.value.code == "403"


async def test_empty_openai_tool_name_filter_is_denied_because_openai_reads_it_as_every_tool(
    native_auth: UserAPIKeyAuth,
) -> None:
    with pytest.raises(ProxyException) as denied:
        await authorize_provider_mcp(_openai({"tool_names": []}), native_auth)
    assert (denied.value.code, denied.value.message) == (
        "403",
        "Provider MCP tool filters require a nonempty tool_names list",
    )


async def test_failed_tool_grant_lookup_denies_instead_of_allowing_every_registered_tool(
    native_auth: UserAPIKeyAuth, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key reaches the server directly, but its toolset grant cannot be read: the tool lookup fails
    while server admission still succeeds, and that failure must not widen to every registered tool."""
    team: Final = await proxy_server.user_api_key_cache.async_get_cache(key="team_id:team")
    toolset_store_down: Final = SimpleNamespace(
        litellm_teamtable=SimpleNamespace(find_unique=AsyncMock(return_value=team)),
        litellm_mcptoolsettable=SimpleNamespace(find_many=AsyncMock(side_effect=RuntimeError("toolset store down"))),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=toolset_store_down, writer_db=toolset_store_down))
    auth: Final = native_auth.model_copy(
        update={
            "requires_fresh_policy": True,
            "object_permission": LiteLLM_ObjectPermissionTable(
                object_permission_id="key-grant", mcp_servers=["native-alias"], mcp_toolsets=["research"]
            ),
        }
    )
    with pytest.raises(ProxyException) as denied:
        await authorize_provider_mcp(_openai(["search"]), auth)
    assert denied.value.code == "403"


@pytest.mark.parametrize("tools,admitted", [(["search"], True), (["delete"], False)])
async def test_mcp_tools_carried_in_responses_input_items_are_admitted_like_top_level_tools(
    native_auth: UserAPIKeyAuth, tools: list[str], admitted: bool
) -> None:
    body: Final = {"input": [{"type": "tool_search_output", "tools": _openai(tools)["tools"]}]}
    context: Final = {"litellm_metadata": {"user_api_key_auth": native_auth}}
    if admitted:
        await authorize_provider_mcp_request(body, context)
        return
    with pytest.raises(ProxyException) as denied:
        await authorize_provider_mcp_request(body, context)
    assert denied.value.code == "403"


@pytest.mark.parametrize(
    "body",
    [
        _openai(["search"], "https://unregistered.example/mcp"),
        _openai(["delete"]),
        _openai({"tool_names": []}),
        {"tools": [{"type": "mcp", "server_label": "docs", "server_url": "https://native.example/mcp"}]},
    ],
    ids=["unregistered_url", "ungranted_tool", "empty_tool_name_filter", "unbounded_attachment"],
)
async def test_provider_mcp_checks_are_off_unless_the_proxy_opts_in(
    native_auth: UserAPIKeyAuth, monkeypatch: pytest.MonkeyPatch, body: dict[str, object]
) -> None:
    """Without enforce_provider_mcp_permissions, provider-run MCP passes through as it did before."""
    context: Final = {"litellm_metadata": {"user_api_key_auth": native_auth}}
    monkeypatch.setattr(proxy_server, "general_settings", {"supported_db_objects": []})
    assert await authorize_provider_mcp_request(body, context) is None
    monkeypatch.setattr(
        proxy_server, "general_settings", {"supported_db_objects": [], "enforce_provider_mcp_permissions": "true"}
    )
    with pytest.raises(ProxyException) as denied:
        await authorize_provider_mcp_request(body, context)
    assert denied.value.code == "403"


async def test_ungranted_provider_tool_reaches_provider_unchanged_when_checks_are_off(
    native_auth: UserAPIKeyAuth, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    import json

    monkeypatch.setattr(proxy_server, "general_settings", {"supported_db_objects": []})
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post(url__regex=r"https://native-provider.example/.*").respond(
        200,
        json={
            "id": "resp_native", "object": "response", "created_at": 1, "model": "gpt-5.6-luna",
            "status": "completed", "output": [], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    )
    attachment: Final = _openai(["delete"], "https://unregistered.example/mcp")
    await litellm.aresponses(
        model="openai/gpt-5.6-luna",
        input="Use the tool",
        api_key="test-provider-key",
        api_base="https://native-provider.example/v1",
        litellm_metadata={"user_api_key_auth": native_auth},
        **attachment,
    )
    assert json.loads(upstream.calls.last.request.content)["tools"] == attachment["tools"]


async def test_canonical_host_and_default_port_match(native_auth: UserAPIKeyAuth) -> None:
    body: Final = _openai(["search"], "https://NATIVE.example:443/mcp")
    await authorize_provider_mcp(body, native_auth)
    assert body == _openai(["search"], "https://NATIVE.example:443/mcp")


async def test_toolsets_match_only_their_named_server_in_a_mixed_configuration(native_auth: UserAPIKeyAuth) -> None:
    body: Final = {
        "mcp_servers": [
            {
                "type": "url",
                "name": "legacy",
                "url": "https://native.example/mcp",
                "tool_configuration": {"allowed_tools": ["search"]},
            },
            {"type": "url", "name": "modern", "url": "https://native.example/mcp"},
        ],
        "tools": [
            {
                "type": "mcp_toolset",
                "mcp_server_name": "modern",
                "default_config": {"enabled": False},
                "configs": {"search": {"enabled": True}},
            }
        ],
    }
    original: Final = deepcopy(body)
    await authorize_provider_mcp(body, native_auth)
    assert body == original


@pytest.mark.parametrize(
    "call_type,model",
    [("aresponses", "openai/gpt-5.6-luna"), ("acompletion", "anthropic/claude-haiku-5-5")],
)
async def test_extra_body_cannot_replace_authorized_attachment(
    native_auth: UserAPIKeyAuth, call_type: str, model: str
) -> None:
    kwargs: Final = {
        "model": model,
        **_openai(),
        "extra_body": _openai(["delete"]),
        "litellm_metadata": {"user_api_key_auth": native_auth},
    }
    with pytest.raises(ProxyException) as denied:
        await async_pre_call_deployment_hook(kwargs, call_type)
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


@pytest.mark.parametrize("api", ["aresponses", "anthropic_messages", "acompletion"])
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
        "metadata" if api == "acompletion" else "litellm_metadata": {"user_api_key_auth": native_auth},
        **({} if configured else attachment),
    }
    call: Final = (
        router.aresponses(input="Use the tool", **request)
        if api == "aresponses"
        else router.anthropic_messages(messages=[{"role": "user", "content": "Use the tool"}], max_tokens=32, **request)
        if api == "anthropic_messages"
        else router.acompletion(messages=[{"role": "user", "content": "Use the tool"}], max_tokens=32, **request)
    )
    with pytest.raises(ProxyException) as denied:
        await call
    assert str(getattr(denied.value, "status_code", getattr(denied.value, "code", ""))) == "403"
    assert upstream.call_count == 0


CLAUDE_PLATFORM_PARAMS: Final = {
    "model": "bedrock/claude_platform/claude-sonnet-4-6",
    "api_key": "test-provider-key",
    "api_base": "https://native-provider.example/v1/messages",
    "workspace_id": "wrkspc_test",
    "aws_region_name": "us-west-2",
}
CLAUDE_PLATFORM_REPLY: Final = {
    "id": "msg_native",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "done"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("routed", [False, True])
async def test_claude_platform_chat_with_ungranted_provider_tools_never_reaches_bedrock(
    native_auth: UserAPIKeyAuth, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, stream: bool, routed: bool
) -> None:
    """Claude Platform on Bedrock runs MCP servers named in a chat request, so chat must be checked
    like the Anthropic provider, both before dispatch (routed) and on the final body (direct)."""
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post(url__regex=r"https://native-provider.example/.*").respond(
        200, json=CLAUDE_PLATFORM_REPLY
    )
    request: Final = {
        "messages": [{"role": "user", "content": "Use the tool"}],
        "max_tokens": 32,
        "stream": stream,
        "metadata": {"user_api_key_auth": native_auth},
        **_openai(["delete"]),
    }
    call: Final = (
        litellm.Router(
            model_list=[{"model_name": "platform", "litellm_params": dict(CLAUDE_PLATFORM_PARAMS)}], num_retries=0
        ).acompletion(model="platform", **request)
        if routed
        else litellm.acompletion(**CLAUDE_PLATFORM_PARAMS, **request)
    )
    with pytest.raises((ProxyException, litellm.PermissionDeniedError)) as denied:
        await call
    assert str(getattr(denied.value, "status_code", getattr(denied.value, "code", ""))) == "403"
    assert upstream.call_count == 0


async def test_claude_platform_chat_with_granted_provider_tools_reaches_bedrock_with_its_mcp_server(
    native_auth: UserAPIKeyAuth, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    import json

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    upstream: Final = respx_mock.post(url__regex=r"https://native-provider.example/.*").respond(
        200, json=CLAUDE_PLATFORM_REPLY
    )
    await litellm.acompletion(
        **CLAUDE_PLATFORM_PARAMS,
        messages=[{"role": "user", "content": "Use the tool"}],
        max_tokens=32,
        metadata={"user_api_key_auth": native_auth},
        **_openai(["search"]),
    )
    sent: Final = json.loads(upstream.calls.last.request.content)
    assert sent["mcp_servers"] == [
        {
            "type": "url",
            "url": "https://native.example/mcp",
            "name": "arbitrary",
            "tool_configuration": {"allowed_tools": ["search"]},
        }
    ]


@pytest.mark.parametrize("api", ["aresponses", "anthropic_messages", "acompletion"])
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
    attachment: Final = _openai(["search"]) if api == "aresponses" else _anthropic(legacy=api == "acompletion")
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
    elif api == "anthropic_messages":
        await litellm.anthropic_messages(
            model="anthropic/claude-haiku-5-5",
            messages=[{"role": "user", "content": "Use the tool"}],
            max_tokens=32,
            **request,
        )
    else:
        await litellm.acompletion(
            model="anthropic/claude-haiku-5-5",
            messages=[{"role": "user", "content": "Use the tool"}],
            max_tokens=32,
            **request,
        )
    sent: Final = json.loads(upstream.calls.last.request.content)
    assert {key: sent[key] for key in attachment} == attachment
    assert upstream.call_count == 1


@pytest.mark.parametrize("api", ["aresponses", "acompletion"])
async def test_cached_response_does_not_bypass_revoked_mcp_permission(
    native_auth: UserAPIKeyAuth,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    api: str,
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
        }
        if api == "aresponses"
        else {
            "id": "msg_cached",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": "tool result"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    request: Final = {
        "caching": True,
        "api_key": "test-provider-key",
        "api_base": "https://native-provider.example/v1",
        **(
            {"model": "openai/gpt-5.6-luna", "input": "Use the tool", **_openai(["search"])}
            if api == "aresponses"
            else {
                "model": "anthropic/claude-haiku-5-5",
                "messages": [{"role": "user", "content": "Use the tool"}],
                "tools": [
                    {
                        "type": "url",
                        "name": "docs",
                        "url": "https://native.example/mcp",
                        "tool_configuration": {"allowed_tools": ["search"]},
                    }
                ],
            }
        ),
    }
    call: Final = litellm.aresponses if api == "aresponses" else litellm.acompletion
    await call(**request, litellm_metadata={"user_api_key_auth": native_auth})
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    await call(**request, litellm_metadata={"user_api_key_auth": native_auth})
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
        await call(**request, litellm_metadata={"user_api_key_auth": denied_auth})
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


class _LateChatMCPAttachment(CustomLogger):
    def log_pre_api_call(self, model: str, messages: list, kwargs: dict) -> None:
        body: Final = kwargs["additional_args"]["complete_input_dict"]  # mutable-ok: exercise payload mutation by a callback
        body.update(
            {
                "mcp_servers": [
                    {
                        "type": "url",
                        "name": "docs",
                        "url": "https://unregistered.example/mcp",
                        "tool_configuration": {"allowed_tools": ["search"]},
                    }
                ]
            }
        )


@pytest.mark.parametrize("api", ["anthropic_messages", "acompletion", "claude_platform_chat"])
@pytest.mark.parametrize("stream", [False, True])
async def test_final_body_check_blocks_attachment_added_after_deployment_hook(
    native_auth: UserAPIKeyAuth,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    api: str,
    stream: bool,
) -> None:
    callback: Final = _LateChatMCPAttachment() if api in ("acompletion", "claude_platform_chat") else _LateMCPAttachment()
    monkeypatch.setattr(litellm, "callbacks", [callback])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    upstream: Final = respx_mock.post(url__regex=r"https://native-provider.example/.*").respond(500)
    call: Final = litellm.anthropic_messages if api == "anthropic_messages" else litellm.acompletion
    deployment: Final = (
        CLAUDE_PLATFORM_PARAMS
        if api == "claude_platform_chat"
        else {
            "model": "anthropic/claude-haiku-5-5",
            "api_key": "test-provider-key",
            "api_base": "https://native-provider.example/v1",
        }
    )
    with pytest.raises((ProxyException, litellm.PermissionDeniedError)) as denied:
        await call(
            **deployment,
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=32,
            stream=stream,
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
