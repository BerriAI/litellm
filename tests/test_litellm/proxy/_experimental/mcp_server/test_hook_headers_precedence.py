"""
Unit tests for MCP pre_mcp_call hook-header precedence (issue #31977).

``mcp_jwt_signer`` injects a signed JWT as ``Authorization`` in the
``pre_mcp_call`` hook. On the ``tools/call`` path it must NOT overwrite a
per-user OAuth ``Authorization`` credential (it already doesn't on
``tools/list``) — otherwise OAuth-backed MCP servers reject the call. It should
still override an admin-configured *static* ``Authorization``, and on a
non-OAuth server a caller-forwarded ``Authorization`` must not suppress it.
"""

from types import SimpleNamespace

from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    _merge_hook_extra_headers_with_precedence,
)
from litellm.types.mcp import MCPAuth

_JWT = "Bearer litellm-signed-jwt"
_OAUTH = "Bearer user-oauth-token"
_STATIC = "Bearer static-token"


def _server(static_headers=None, auth_type=None):
    return SimpleNamespace(server_name="test-mcp", name="test-mcp", static_headers=static_headers, auth_type=auth_type)


def _merge(extra=None, hook=None, server_auth=None, mcp_auth=None, static_headers=None, auth_type=None):
    return _merge_hook_extra_headers_with_precedence(
        extra_headers=extra,
        hook_extra_headers=hook or {},
        server_auth_header=server_auth,
        mcp_auth_header=mcp_auth,
        mcp_server=_server(static_headers=static_headers, auth_type=auth_type),
    )


def test_caller_oauth_authorization_is_preserved_on_oauth_server():
    # OAuth token resolved into extra_headers on an OAuth-backed server.
    result = _merge(
        extra={"Authorization": _OAUTH},
        hook={"Authorization": _JWT, "X-Extra": "keep"},
        auth_type=MCPAuth.oauth2,
    )
    assert result["Authorization"] == _OAUTH  # OAuth wins, JWT dropped
    assert result["X-Extra"] == "keep"  # non-auth hook headers still applied


def test_server_auth_header_preserves_credential():
    result = _merge(hook={"Authorization": _JWT}, server_auth=_OAUTH)
    assert "Authorization" not in result  # JWT dropped; server auth_value carries it


def test_mcp_auth_header_preserves_credential():
    result = _merge(hook={"Authorization": _JWT}, mcp_auth=_OAUTH)
    assert "Authorization" not in result


def test_jwt_applied_when_no_existing_credential():
    result = _merge(hook={"Authorization": _JWT})
    assert result["Authorization"] == _JWT  # signer JWT is the sole credential


def test_jwt_still_overrides_static_authorization():
    # The signer supersedes an admin-configured static Authorization.
    result = _merge(
        extra={"Authorization": _STATIC},
        hook={"Authorization": _JWT},
        static_headers={"Authorization": _STATIC},
    )
    assert result["Authorization"] == _JWT


def test_non_oauth_server_caller_authorization_does_not_suppress_jwt():
    # On a non-OAuth server, a caller-forwarded Authorization must NOT suppress
    # the signer JWT — the JWT stays authoritative (issue #31977 review).
    result = _merge(
        extra={"Authorization": "Bearer caller-supplied"},
        hook={"Authorization": _JWT},
        auth_type=None,
    )
    assert result["Authorization"] == _JWT


def test_non_authorization_hook_headers_always_applied():
    result = _merge(extra={"Authorization": _OAUTH}, hook={"X-Signed": "1"}, auth_type=MCPAuth.oauth2)
    assert result["Authorization"] == _OAUTH
    assert result["X-Signed"] == "1"


def test_authorization_match_is_case_insensitive():
    result = _merge(
        extra={"authorization": _OAUTH},  # lowercase caller OAuth
        hook={"Authorization": _JWT},  # uppercase hook
        auth_type=MCPAuth.oauth2,
    )
    assert result["authorization"] == _OAUTH
    assert "Authorization" not in result  # hook's uppercase variant not added
