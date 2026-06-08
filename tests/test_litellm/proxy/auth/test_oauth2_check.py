import base64
import os
import sys
from urllib.parse import parse_qs

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
