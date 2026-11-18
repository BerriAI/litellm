from typing import Final

import httpx2
import pytest
from mcp.types import CallToolResult, JSONRPCResponse
from pydantic import SecretStr

from litellm.proxy.mcp_server.auth_profile_selection import (
    AmbiguousProfile,
    AuthProfileGrantSource,
    AuthProfileOwnerSource,
    UnloadedProfile,
    effective_server_for,
    select_auth_profile,
    with_auth_profile,
    without_profile_credential_headers,
)
from litellm.proxy.mcp_server.outbound_credentials.adapter import to_server_spec
from litellm.proxy.mcp_server.outbound_credentials.types import ClientCredentialsConfig, ServerSpec
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPAuthProfile, MCPServer
from tests.unit.proxy.mcp_server.auth_profile_fakes import AnalyticsUpstream, granting

BOT_PROFILE: Final = MCPAuthProfile(
    id="ana-bots",
    auth_type=MCPAuth.oauth2,
    oauth2_flow="client_credentials",
    client_id="bot-app",
    client_secret="bot-secret",
    token_url="https://idp.example/bot-token",
    scopes=["read:news"],
)
STATIC_PROFILE: Final = MCPAuthProfile(
    id="ana-static", auth_type=MCPAuth.bearer_token, authentication_token="static-bot-token"
)


def _interactive_server(**overrides: object) -> MCPServer:
    return MCPServer.model_validate(
        {
            "server_id": "analytics",
            "name": "analytics",
            "url": "https://mcp.analytics.example/v1/mcp",
            "transport": MCPTransport.http,
            "auth_type": MCPAuth.oauth2,
            "oauth2_flow": "authorization_code",
            "client_id": "person-app",
            "authorization_url": "https://idp.example/authorize",
            "token_url": "https://idp.example/person-token",
            "registration_url": "https://idp.example/register",
            "configured_authorization_url": "https://idp.example/authorize",
            "configured_token_url": "https://idp.example/person-token",
            "configured_registration_url": "https://idp.example/register",
            "scopes": ["offline_access"],
            "audience": "person-audience",
            "additional_auth": (BOT_PROFILE, STATIC_PROFILE),
            **overrides,
        }
    )


LOADED: Final = frozenset({"ana-bots", "ana-static", "gh-bots"})


@pytest.mark.parametrize(
    ("carriers", "expected"),
    [
        pytest.param((), None, id="no-grant"),
        pytest.param((("gh-bots",),), None, id="another-servers-grant"),
        pytest.param((("ana-bots",),), BOT_PROFILE, id="granted"),
        pytest.param((("ANA-Bots",),), BOT_PROFILE, id="case-insensitive"),
        pytest.param(((), ("ana-bots",)), BOT_PROFILE, id="empty-carrier-falls-through"),
        pytest.param((("ana-static",), ("ana-bots",)), STATIC_PROFILE, id="most-specific-carrier-wins"),
        pytest.param((("gh-bots",), ("ana-bots",)), BOT_PROFILE, id="carrier-without-this-server-falls-through"),
        pytest.param(
            (("ana-static", "ana-bots"),),
            AmbiguousProfile("analytics", ("ana-static", "ana-bots")),
            id="two-ids-of-one-server",
        ),
        pytest.param((("new-bots",), ("ana-bots",)), UnloadedProfile(("new-bots",)), id="unloaded-id-stops"),
        pytest.param((("ana-bots", "ana-bots"),), BOT_PROFILE, id="one-id-listed-twice"),
        pytest.param((("ana-bots", "ANA-Bots"),), BOT_PROFILE, id="one-id-in-two-spellings"),
    ],
)
def test_the_most_specific_carrier_naming_this_server_selects_its_profile(
    carriers: tuple[tuple[str, ...], ...], expected: object
) -> None:
    assert select_auth_profile(_interactive_server(), carriers, LOADED) == expected


def test_a_server_without_profiles_selects_the_default_for_any_grant() -> None:
    assert select_auth_profile(_interactive_server(additional_auth=None), (("ana-bots",),), LOADED) is None


def test_a_profiled_copy_sends_none_of_the_default_static_or_forwarded_headers() -> None:
    server: Final = _interactive_server(
        static_headers={"Authorization": "Bearer static", "X-API-Key": "default-key"}, extra_headers=["X-API-Key"]
    )
    copy: Final = with_auth_profile(server, STATIC_PROFILE)

    assert (copy.static_headers, copy.extra_headers) == (None, None)
    assert without_profile_credential_headers(copy, {"authorization": "Bearer caller", "X-Trace": "t"}) == {
        "X-Trace": "t"
    }
    assert without_profile_credential_headers(server, {"authorization": "Bearer caller"}) == {
        "authorization": "Bearer caller"
    }


def test_the_default_profile_leaves_the_server_object_untouched() -> None:
    server: Final = _interactive_server()
    assert with_auth_profile(server, None) is server


@pytest.mark.parametrize(
    "default_overrides",
    [
        {},
        {"delegate_auth_to_upstream": True},
        {"per_server_oauth_discovery": True},
        {"oauth2_flow": "client_credentials", "client_id": "default-app", "client_secret": "default-secret"},
    ],
)
def test_a_client_credentials_profile_mints_with_its_own_app_whatever_the_default_flow(
    default_overrides: dict[str, object],
) -> None:
    effective: Final = with_auth_profile(_interactive_server(**default_overrides), BOT_PROFILE)

    assert to_server_spec(effective) == ServerSpec(
        server_id="analytics",
        resource="https://mcp.analytics.example/v1/mcp",
        config=ClientCredentialsConfig(
            client_id="bot-app",
            client_secret=SecretStr("bot-secret"),
            token_url="https://idp.example/bot-token",
            scopes=("read:news",),
        ),
    )
    assert (
        effective.needs_user_oauth_token,
        effective.requires_per_user_auth,
        effective.reauthorizes_in_client,
        effective.is_gateway_managed_oauth2,
        effective.effective_authorization_url,
        effective.effective_registration_url,
    ) == (False, False, False, True, None, None), "a profile must not look like the default's sign-in flow"
    assert effective.auth_profile_id == "ana-bots"


def test_the_selected_profile_stays_out_of_the_server_dump() -> None:
    server: Final = _interactive_server()
    assert with_auth_profile(server, BOT_PROFILE).model_dump().keys() == server.model_dump().keys()


def test_a_profile_without_a_token_url_never_borrows_the_default_one() -> None:
    profile: Final = BOT_PROFILE.model_copy(update={"token_url": None})
    effective: Final = with_auth_profile(_interactive_server(), profile)

    spec: Final = to_server_spec(effective)
    assert spec is not None and isinstance(spec.config, ClientCredentialsConfig)
    assert spec.config.token_url is None


class _GrantlessContext:
    @property
    def auth_profile_grants(self) -> AuthProfileGrantSource:
        async def no_grants(user_api_key_auth: object) -> tuple[tuple[str, ...], ...]:
            return ((),)

        return no_grants

    @property
    def auth_profile_owners(self) -> AuthProfileOwnerSource:
        raise AssertionError("a caller without grants looked up profile owners")

    def loaded_auth_profile_ids(self) -> frozenset[str]:
        raise AssertionError("a caller without grants scanned the loaded profiles")


@pytest.mark.asyncio
async def test_a_caller_without_grants_never_scans_the_loaded_profiles() -> None:
    server: Final = _interactive_server()
    assert await effective_server_for(server, None, _GrantlessContext()) is server


def test_a_profiled_copy_forwards_no_caller_header_the_default_would_forward() -> None:
    from litellm.proxy.mcp_server.operations import _prepare_mcp_server_headers

    server: Final = _interactive_server(auth_type=MCPAuth.none, oauth2_flow=None, extra_headers=["X-API-Key"])
    caller_headers: Final = {"x-api-key": "caller-key"}

    assert _prepare_mcp_server_headers(server, None, None, None, caller_headers)[1] == {"X-API-Key": "caller-key"}
    assert (
        _prepare_mcp_server_headers(with_auth_profile(server, STATIC_PROFILE), None, None, None, caller_headers)[1]
        is None
    )


class _InstructingUpstream(AnalyticsUpstream):
    async def respond(self, request: httpx2.Request) -> httpx2.Response:
        response: Final = await super().respond(request)
        if not response.content:
            return response
        message: Final = JSONRPCResponse.model_validate_json(response.content)
        if "protocolVersion" not in message.result:
            return response
        notes: Final = f"notes for {request.headers.get('authorization') or 'every caller'}"
        return httpx2.Response(
            200, json={"jsonrpc": "2.0", "id": message.id, "result": {**message.result, "instructions": notes}}
        )


@pytest.mark.asyncio
async def test_a_profiled_call_never_fills_the_initialize_instructions_every_caller_shares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.mcp_server import operations
    from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager

    upstream: Final = _InstructingUpstream()
    server: Final = _interactive_server(auth_type=MCPAuth.none, oauth2_flow=None)
    manager: Final = MCPServerManager(
        upstream_transport=httpx2.MockTransport(upstream.respond),
        auth_profile_grants=granting("ana-static"),
    )
    manager.registry[server.server_id] = server
    manager.tool_name_to_mcp_server_name_mapping["analytics-get_issue"] = server.server_id
    monkeypatch.setattr(operations, "global_mcp_server_manager", manager)

    await manager.ensure_upstream_initialize_instructions_cached(with_auth_profile(server, STATIC_PROFILE))
    called: Final = await manager.call_tool(
        server_name="analytics", name="get_issue", arguments={}, user_api_key_auth=UserAPIKeyAuth(api_key="bot-key")
    )
    await manager.ensure_upstream_initialize_instructions_cached(server)

    assert isinstance(called, CallToolResult) and not called.is_error, called
    assert upstream.calls("initialize") == ("Bearer static-bot-token", "")
    assert operations._merge_gateway_initialize_instructions([server]) == "notes for every caller"
