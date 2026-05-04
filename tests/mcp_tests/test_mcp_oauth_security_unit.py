"""Unit tests for MCP OAuth broker security helpers (discoverable / UI flow).

``_validate_token_response`` rules are covered in ``tests/mcp_tests/test_per_user_oauth_cache.py``.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.discoverable_endpoints import (
    _get_validated_client_redirect_uri,
    decode_state_hash,
    encode_state_with_base_url,
)
from litellm.proxy._experimental.mcp_server.oauth_utils import (
    validate_loopback_redirect_uri,
)


@pytest.mark.parametrize(
    "uri",
    [
        "https://evil.com/callback",
        "http://192.168.1.1/callback",
        "http://10.0.0.1/callback",
        "http://example.com/callback",
    ],
)
def test_validate_loopback_redirect_uri_rejects_non_loopback(uri: str) -> None:
    with pytest.raises(HTTPException) as exc:
        validate_loopback_redirect_uri(uri)
    assert exc.value.status_code == 400


@pytest.mark.parametrize(
    "uri",
    [
        "http://127.0.0.1:9/callback",
        "http://127.0.0.2:4000/ui/mcp/oauth/callback",
        "http://localhost:3000/cb",
        "http://[::1]:8080/oauth/callback",
    ],
)
def test_validate_loopback_redirect_uri_accepts_loopback(uri: str) -> None:
    validate_loopback_redirect_uri(uri)


def test_encode_state_with_base_url_decode_state_hash_roundtrip(monkeypatch) -> None:
    """State must survive encrypt → decrypt with a stable salt (CI-safe)."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "unit-test-salt-key-32chars!!!")

    enc = encode_state_with_base_url(
        base_url="http://127.0.0.1:60108/callback",
        original_state="client-state-xyz",
        code_challenge="cc",
        code_challenge_method="S256",
        client_redirect_uri="http://127.0.0.1:60108/callback",
    )
    assert enc != ""
    data = decode_state_hash(enc)
    assert data["base_url"] == "http://127.0.0.1:60108/callback"
    assert data["original_state"] == "client-state-xyz"
    assert data["code_challenge"] == "cc"
    assert data["code_challenge_method"] == "S256"
    assert data["client_redirect_uri"] == "http://127.0.0.1:60108/callback"


def test_get_validated_client_redirect_uri_accepts_loopback_from_state() -> None:
    uri = _get_validated_client_redirect_uri(
        {
            "client_redirect_uri": "http://127.0.0.1:55/x",
            "base_url": "ignored-when-client-set",
        }
    )
    assert uri == "http://127.0.0.1:55/x"


def test_get_validated_client_redirect_uri_falls_back_to_base_url_loopback() -> None:
    uri = _get_validated_client_redirect_uri(
        {
            "original_state": "s",
            "base_url": "http://localhost:9/oauth",
        }
    )
    assert uri == "http://localhost:9/oauth"


def test_get_validated_client_redirect_uri_rejects_public_client_redirect() -> None:
    with pytest.raises(HTTPException) as exc:
        _get_validated_client_redirect_uri(
            {
                "client_redirect_uri": "https://evil.com/steal",
                "base_url": "http://127.0.0.1:1/x",
            }
        )
    assert exc.value.status_code == 400


def test_get_validated_client_redirect_uri_rejects_public_base_url_fallback() -> None:
    with pytest.raises(HTTPException) as exc:
        _get_validated_client_redirect_uri({"base_url": "https://evil.com/noloop"})
    assert exc.value.status_code == 400


def test_get_validated_client_redirect_uri_empty_client_uses_loopback_base_url() -> (
    None
):
    """When client_redirect_uri is absent/empty, base_url must still be loopback-validated."""
    uri = _get_validated_client_redirect_uri(
        {"client_redirect_uri": "", "base_url": "http://127.0.0.1:1/x"}
    )
    assert uri == "http://127.0.0.1:1/x"


def test_get_validated_client_redirect_uri_rejects_missing_uri() -> None:
    with pytest.raises(HTTPException) as exc:
        _get_validated_client_redirect_uri({"original_state": "x"})
    assert exc.value.status_code == 400
