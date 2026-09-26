from collections.abc import Iterator, Mapping
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._experimental.mcp_server.caller_sign_in import (
    CallerSignIn,
    CallerSignInProvider,
    caller_sign_in_for,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class _SignInGuardrail(CustomGuardrail):
    def __init__(self, issuer: str, scope: str, gated: bool = True) -> None:
        super().__init__(guardrail_name=f"sign-in-{issuer}")
        self.issuer: Final = issuer
        self.scope: Final = scope
        self.gated: Final = gated
        self.seen: list[tuple[str, str | None]] = []  # mutable-ok: call recorder

    def caller_sign_in(self, server: MCPServer, user_api_key_auth: UserAPIKeyAuth | None) -> CallerSignIn | None:
        self.seen.append((server.name, user_api_key_auth.user_id if user_api_key_auth else None))
        if not self.gated:
            return None
        return CallerSignIn(issuers=(self.issuer,), scopes=(self.scope,))


class _PlainGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(guardrail_name="plain")


def _server(auth_type: MCPAuth | None = None, scopes: list[str] | None = None) -> MCPServer:
    return MCPServer(
        server_id="tools-id",
        name="tools",
        server_name="tools",
        transport=MCPTransport.http,
        url="https://tools.test/mcp",
        auth_type=auth_type,
        scopes=scopes,
    )


@pytest.fixture
def registered() -> Iterator[tuple[_SignInGuardrail, _SignInGuardrail]]:
    first: Final = _SignInGuardrail("https://idp-a.test", "scope-a")
    second: Final = _SignInGuardrail("https://idp-b.test", "scope-b")
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


def test_protocol_matches_only_guardrails_implementing_the_hook():
    assert isinstance(_SignInGuardrail("i", "s"), CallerSignInProvider)
    assert not isinstance(_PlainGuardrail(), CallerSignInProvider)


def test_no_registered_provider_and_non_obo_advertises_nothing():
    assert caller_sign_in_for(_server(), None) is None


def test_registered_providers_merge_in_order_and_dedupe(registered):
    first, second = registered
    sign_in: Final = caller_sign_in_for(_server(), None)

    assert sign_in is not None
    assert sign_in.issuers == ("https://idp-a.test", "https://idp-b.test")
    assert sign_in.scopes == ("scope-a", "scope-b")
    assert first.seen == [("tools", None)]
    assert second.seen == [("tools", None)]


def test_provider_returning_none_contributes_nothing(registered):
    ungated = _SignInGuardrail("https://idp-c.test", "scope-c", gated=False)
    litellm.logging_callback_manager.add_litellm_callback(ungated)
    try:
        sign_in: Final = caller_sign_in_for(_server(), None)
        assert sign_in is not None
        assert "https://idp-c.test" not in sign_in.issuers
    finally:
        litellm.logging_callback_manager.remove_callback_from_list_by_object(
            litellm.callbacks, ungated, require_self=False
        )


def test_obo_server_contributes_jwt_issuers_and_own_scopes(monkeypatch):
    monkeypatch.setenv("JWT_ISSUER", "https://jwt-idp.test")
    sign_in: Final = caller_sign_in_for(
        _server(auth_type=MCPAuth.oauth2_token_exchange, scopes=["read"]), None
    )
    assert sign_in == CallerSignIn(issuers=("https://jwt-idp.test",), scopes=("read",))


def test_obo_server_and_provider_merge_and_dedupe(monkeypatch, registered):
    monkeypatch.setenv("JWT_ISSUER", "https://idp-a.test")
    sign_in: Final = caller_sign_in_for(
        _server(auth_type=MCPAuth.oauth2_token_exchange, scopes=["read", "scope-a"]), None
    )
    assert sign_in is not None
    assert sign_in.issuers == ("https://idp-a.test", "https://idp-b.test")
    assert sign_in.scopes == ("read", "scope-a", "scope-b")


def test_obo_server_without_jwt_issuer_still_signs_in_when_a_provider_gates(registered):
    sign_in: Final = caller_sign_in_for(_server(auth_type=MCPAuth.oauth2_token_exchange), None)
    assert sign_in is not None
    assert sign_in.issuers == ("https://idp-a.test", "https://idp-b.test")


def test_oauth_utils_strips_the_route_relative_root_path():
    """Regression: Starlette sets ``app_root_path`` to ``""`` on an unmounted app, so the strip must
    fall back to ``root_path`` (which is where ``/mcp`` lands when the MCP app is mounted)."""
    from litellm.proxy._experimental.mcp_server.oauth_utils import get_route_relative_request_path

    scope: Final[Mapping[str, object]] = {
        "type": "http",
        "path": "/mcp/catalog",
        "root_path": "/mcp",
        "app_root_path": "",
    }
    assert get_route_relative_request_path(scope) == "/catalog"  # pyright: ignore[reportArgumentType]
