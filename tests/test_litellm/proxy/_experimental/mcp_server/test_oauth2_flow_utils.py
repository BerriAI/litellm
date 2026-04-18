"""Tests for MCP OAuth2 flow persistence helpers."""

import pytest

from litellm.proxy._experimental.mcp_server.oauth2_flow_utils import (
    infer_oauth2_flow_for_storage,
    resolve_oauth2_flow_for_runtime,
)
from litellm.types.mcp import MCPAuth


def test_infer_explicit_client_credentials():
    assert (
        infer_oauth2_flow_for_storage(
            auth_type=MCPAuth.oauth2,
            oauth2_flow="client_credentials",
            token_url="https://idp.example.com/token",
            credentials_plain={"client_id": "a", "client_secret": "b"},
        )
        == "client_credentials"
    )


def test_infer_explicit_authorization_code():
    assert (
        infer_oauth2_flow_for_storage(
            auth_type=MCPAuth.oauth2,
            oauth2_flow="authorization_code",
            token_url="https://github.example.com/token",
            credentials_plain={"client_id": "a", "client_secret": "b"},
        )
        == "authorization_code"
    )


def test_infer_from_credentials_and_token_url_ui_pattern():
    """UI omits oauth2_flow but sends M2M-style fields (same as _execute_with_mcp_client)."""
    assert (
        infer_oauth2_flow_for_storage(
            auth_type=MCPAuth.oauth2,
            oauth2_flow=None,
            token_url="http://127.0.0.1:18901/oauth/token",
            credentials_plain={
                "client_id": "cid",
                "client_secret": "sec",
            },
        )
        == "client_credentials"
    )


def test_infer_not_oauth2():
    assert (
        infer_oauth2_flow_for_storage(
            auth_type=MCPAuth.api_key,
            oauth2_flow=None,
            token_url="https://x/token",
            credentials_plain={"client_id": "a", "client_secret": "b"},
        )
        is None
    )


def test_infer_partial_credentials():
    assert (
        infer_oauth2_flow_for_storage(
            auth_type=MCPAuth.oauth2,
            oauth2_flow=None,
            token_url="https://x/token",
            credentials_plain={"client_id": "only-id"},
        )
        is None
    )


def test_resolve_runtime_uses_stored():
    assert (
        resolve_oauth2_flow_for_runtime(
            auth_type=MCPAuth.oauth2,
            stored_oauth2_flow="authorization_code",
            token_url="https://gh/token",
            client_id="a",
            client_secret="b",
        )
        == "authorization_code"
    )


def test_resolve_runtime_infers_when_stored_null():
    assert (
        resolve_oauth2_flow_for_runtime(
            auth_type=MCPAuth.oauth2,
            stored_oauth2_flow=None,
            token_url="https://idp/token",
            client_id="x",
            client_secret="y",
        )
        == "client_credentials"
    )
