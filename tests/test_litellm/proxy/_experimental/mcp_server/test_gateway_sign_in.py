from collections.abc import Iterator, Mapping
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._experimental.mcp_server.gateway_sign_in import (
    GatewaySignInProvider,
    gateway_authorization_servers,
    gateway_scopes_supported,
    gateway_sign_in_required,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class _SignInGuardrail(CustomGuardrail):
    def __init__(self, issuer: str, scope: str, requires_sign_in: bool) -> None:
        super().__init__(guardrail_name=f"sign-in-{issuer}")
        self.issuer: Final = issuer
        self.scope: Final = scope
        self.requires_sign_in: Final = requires_sign_in
        self.seen: list[tuple[str, str | None, Mapping[str, str] | None]] = []  # mutable-ok: call recorder

    def gateway_authorization_servers(
        self, server: MCPServer, user_api_key_auth: UserAPIKeyAuth | None
    ) -> tuple[str, ...]:
        return (self.issuer,)

    def gateway_scopes_supported(self, server: MCPServer, user_api_key_auth: UserAPIKeyAuth | None) -> tuple[str, ...]:
        return (self.scope,)

    async def gateway_sign_in_required(
        self, server: MCPServer, user_api_key_auth: UserAPIKeyAuth | None, oauth2_headers: Mapping[str, str] | None
    ) -> bool:
        self.seen.append((server.name, user_api_key_auth.user_id if user_api_key_auth else None, oauth2_headers))
        return self.requires_sign_in


class _PlainGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(guardrail_name="plain")


def _server(scopes: list[str] | None = None) -> MCPServer:
    return MCPServer(
        server_id="tools-id",
        name="tools",
        server_name="tools",
        transport=MCPTransport.http,
        url="https://tools.test/mcp",
        scopes=scopes,
    )


@pytest.fixture
def registered() -> Iterator[tuple[_SignInGuardrail, _SignInGuardrail]]:
    first: Final = _SignInGuardrail("https://idp-a.test", "scope-a", requires_sign_in=False)
    second: Final = _SignInGuardrail("https://idp-b.test", "scope-b", requires_sign_in=True)
    plain: Final = _PlainGuardrail()
    for callback in (first, plain, second):
        litellm.logging_callback_manager.add_litellm_callback(callback)
    try:
        yield first, second
    finally:
        for callback in (first, plain, second):
            litellm.logging_callback_manager.remove_callback_from_list_by_object(
                litellm.callbacks, callback, require_self=False
            )


def test_protocol_matches_only_guardrails_implementing_every_hook():
    assert isinstance(_SignInGuardrail("i", "s", requires_sign_in=False), GatewaySignInProvider)
    assert not isinstance(_PlainGuardrail(), GatewaySignInProvider)


def test_no_registered_provider_advertises_nothing_and_never_challenges():
    assert gateway_authorization_servers(_server(), None) == ()
    assert gateway_scopes_supported(_server(), None) == ()


@pytest.mark.asyncio
async def test_no_registered_provider_does_not_require_sign_in():
    assert await gateway_sign_in_required(_server(), None, {"Authorization": "Bearer sk-key"}) is False


def test_issuers_and_scopes_come_from_every_provider_in_registration_order(registered):
    assert gateway_authorization_servers(_server(), None) == ("https://idp-a.test", "https://idp-b.test")
    assert gateway_scopes_supported(_server(), None) == ("scope-a", "scope-b")


def test_admin_scopes_on_the_server_replace_provider_scopes(registered):
    assert gateway_scopes_supported(_server(scopes=["admin-scope"]), None) == ("admin-scope",)


def test_duplicate_issuers_are_advertised_once(registered):
    twin: Final = _SignInGuardrail("https://idp-a.test", "scope-a", requires_sign_in=False)
    litellm.logging_callback_manager.add_litellm_callback(twin)
    try:
        assert gateway_authorization_servers(_server(), None) == ("https://idp-a.test", "https://idp-b.test")
    finally:
        litellm.logging_callback_manager.remove_callback_from_list_by_object(
            litellm.callbacks, twin, require_self=False
        )


@pytest.mark.asyncio
async def test_any_provider_requiring_sign_in_challenges_the_connect(registered):
    first, second = registered
    key: Final = UserAPIKeyAuth(api_key="sk-key", user_id="u-1")
    headers: Final = {"Authorization": "Bearer a.b.c"}

    assert await gateway_sign_in_required(_server(), key, headers) is True
    assert first.seen == [("tools", "u-1", headers)]
    assert second.seen == [("tools", "u-1", headers)]


@pytest.mark.asyncio
async def test_no_provider_requiring_sign_in_admits_the_connect():
    lenient: Final = _SignInGuardrail("https://idp-a.test", "scope-a", requires_sign_in=False)
    litellm.logging_callback_manager.add_litellm_callback(lenient)
    try:
        assert await gateway_sign_in_required(_server(), None, None) is False
    finally:
        litellm.logging_callback_manager.remove_callback_from_list_by_object(
            litellm.callbacks, lenient, require_self=False
        )
