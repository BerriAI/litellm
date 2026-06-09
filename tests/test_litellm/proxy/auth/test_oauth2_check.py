import base64
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.auth.oauth2_check import Oauth2Handler


def test_introspection_request_confidential_client_uses_basic_auth():
    headers, body = Oauth2Handler._prepare_introspection_request(
        token="opaque-token",
        oauth_client_id="client-id",
        oauth_client_secret="client-secret",
    )

    assert headers["Content-Type"] == "application/x-www-form-urlencoded"
    expected = base64.b64encode(b"client-id:client-secret").decode()
    assert headers["Authorization"] == f"Basic {expected}"

    parsed = parse_qs(body)
    assert parsed["token"] == ["opaque-token"]
    assert "client_secret" not in parsed


def test_introspection_request_public_client_puts_client_id_in_body():
    headers, body = Oauth2Handler._prepare_introspection_request(
        token="opaque-token",
        oauth_client_id="client-id",
        oauth_client_secret=None,
    )

    assert "Authorization" not in headers
    parsed = parse_qs(body)
    assert parsed["token"] == ["opaque-token"]
    assert parsed["client_id"] == ["client-id"]


def test_introspection_request_without_credentials_only_sends_token():
    headers, body = Oauth2Handler._prepare_introspection_request(
        token="opaque-token",
        oauth_client_id=None,
        oauth_client_secret=None,
    )

    assert "Authorization" not in headers
    parsed = parse_qs(body)
    assert parsed == {"token": ["opaque-token"]}


@pytest.mark.asyncio
async def test_check_oauth2_token_sends_introspection_body_as_raw_content(monkeypatch):
    """The introspection POST must send the pre-encoded form body via httpx
    ``content=``. ``_prepare_introspection_request`` already form-encodes the
    body (and Authlib may append client auth), so passing it through ``data=``
    would re-encode it and is deprecated for strings in httpx.
    """
    monkeypatch.setenv(
        "OAUTH_TOKEN_INFO_ENDPOINT", "https://idp.example.com/introspect"
    )
    monkeypatch.setenv("OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "client-secret")

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(
        return_value={"active": True, "sub": "u1", "role": "internal_user"}
    )

    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    with (
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch(
            "litellm.proxy.auth.oauth2_check.get_async_httpx_client",
            return_value=client,
        ),
    ):
        await Oauth2Handler.check_oauth2_token("opaque-token")

    assert client.post.await_count == 1
    kwargs = client.post.await_args.kwargs
    assert "data" not in kwargs
    assert isinstance(kwargs["content"], str)
    assert "token=opaque-token" in kwargs["content"]
