import base64
import hashlib
import json
import os
import threading
import time
from urllib.parse import parse_qs, urlparse
from unittest.mock import MagicMock

import httpx
import litellm
import pytest
from click.testing import CliRunner

import litellm.llms.xai.oauth as xai_oauth_module
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.xai.oauth import (
    XAI_OAUTH_CLIENT_ID,
    XAI_OAUTH_SCOPE,
    XAIOAuthError,
    XAIOAuthAuthenticator,
    XAIOAuthLoginRequiredError,
)
from litellm.llms.xai.chat.transformation import XAIChatConfig
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import get_optional_params, validate_environment


def _write_auth_file(tmp_path, payload):
    token_dir = tmp_path / "xai_oauth"
    token_dir.mkdir()
    auth_file = token_dir / "auth.json"
    auth_file.write_text(json.dumps(payload))
    return token_dir, auth_file


def test_get_access_token_uses_fresh_local_token(tmp_path, monkeypatch):
    token_dir, _ = _write_auth_file(
        tmp_path,
        {
            "access_token": "fresh-token",
            "refresh_token": "refresh-token",
            "expires_at": time.time() + 3600,
        },
    )
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))

    assert XAIOAuthAuthenticator().get_access_token() == "fresh-token"


def test_get_access_token_refreshes_and_preserves_refresh_token(tmp_path, monkeypatch):
    token_dir, auth_file = _write_auth_file(
        tmp_path,
        {
            "access_token": "expired-token",
            "refresh_token": "refresh-token",
            "token_endpoint": "https://auth.x.ai/oauth/token",
            "expires_at": time.time() - 1,
        },
    )
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))

    def handler(request: httpx.Request) -> httpx.Response:
        body = dict(item.split("=") for item in request.content.decode().split("&"))
        assert body["grant_type"] == "refresh_token"
        assert body["refresh_token"] == "refresh-token"
        assert body["client_id"] == XAI_OAUTH_CLIENT_ID
        return httpx.Response(
            200,
            json={
                "access_token": "new-token",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert XAIOAuthAuthenticator(http_client=client).get_access_token() == "new-token"
    stored = json.loads(auth_file.read_text())
    assert stored["access_token"] == "new-token"
    assert stored["refresh_token"] == "refresh-token"


def test_get_access_token_reuses_token_refreshed_by_parallel_request():
    expired_auth_data = {
        "access_token": "expired-token",
        "refresh_token": "refresh-token",
        "token_endpoint": "https://auth.x.ai/oauth/token",
        "expires_at": time.time() - 1,
    }
    refreshed_auth_data = {
        "access_token": "already-refreshed-token",
        "refresh_token": "rotated-refresh-token",
        "token_endpoint": "https://auth.x.ai/oauth/token",
        "expires_at": time.time() + 3600,
    }
    authenticator = XAIOAuthAuthenticator()
    authenticator._read_auth_file = MagicMock(side_effect=[expired_auth_data, refreshed_auth_data])
    authenticator._refresh_tokens = MagicMock()

    assert authenticator.get_access_token() == "already-refreshed-token"
    authenticator._refresh_tokens.assert_not_called()


def test_get_access_token_requires_login_without_auth_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(tmp_path / "missing"))

    with pytest.raises(XAIOAuthLoginRequiredError):
        XAIOAuthAuthenticator().get_access_token()


def test_get_access_token_ignores_invalid_auth_file(tmp_path, monkeypatch):
    token_dir = tmp_path / "xai_oauth"
    token_dir.mkdir()
    (token_dir / "auth.json").write_text("{not-json")
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))

    with pytest.raises(XAIOAuthLoginRequiredError):
        XAIOAuthAuthenticator().get_access_token()


def test_refresh_failure_surfaces_oauth_error(tmp_path, monkeypatch):
    token_dir, _ = _write_auth_file(
        tmp_path,
        {
            "access_token": "expired-token",
            "refresh_token": "refresh-token",
            "token_endpoint": "https://auth.x.ai/oauth/token",
            "expires_at": time.time() - 1,
        },
    )
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))

    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(401, text="invalid_grant", request=request))
    )

    with pytest.raises(XAIOAuthError) as exc_info:
        XAIOAuthAuthenticator(http_client=client).get_access_token()

    assert "401 invalid_grant" in str(exc_info.value)


def test_build_auth_record_requires_access_and_refresh_tokens():
    authenticator = XAIOAuthAuthenticator()

    with pytest.raises(XAIOAuthError, match="access_token"):
        authenticator._build_auth_record(
            {"refresh_token": "refresh-token"},
            "https://auth.x.ai/oauth/token",
        )

    with pytest.raises(XAIOAuthError, match="refresh_token"):
        authenticator._build_auth_record(
            {"access_token": "access-token"},
            "https://auth.x.ai/oauth/token",
        )


def test_build_auth_record_defaults_expiry_and_token_type():
    authenticator = XAIOAuthAuthenticator()

    auth_data = authenticator._build_auth_record(
        {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": "not-a-number",
        },
        "https://auth.x.ai/oauth/token",
    )

    assert auth_data["token_type"] == "Bearer"
    assert auth_data["expires_at"] > time.time()


def test_is_expired_treats_missing_or_invalid_expiry_as_expired():
    authenticator = XAIOAuthAuthenticator()

    assert authenticator._is_expired({}) is True
    assert authenticator._is_expired({"expires_at": "not-a-number"}) is True


def test_write_auth_file_creates_private_file(tmp_path, monkeypatch):
    token_dir = tmp_path / "xai_oauth"
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))
    authenticator = XAIOAuthAuthenticator()
    old_umask = os.umask(0o022)
    replace_calls = []
    real_replace = os.replace

    def assert_private_temp_file(src, dst):
        replace_calls.append((src, dst))
        assert oct(os.stat(src).st_mode & 0o777) == "0o600"
        with open(src) as f:
            assert json.load(f)["refresh_token"] == "refresh-token"
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", assert_private_temp_file)

    try:
        authenticator._write_auth_file(
            {
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_at": time.time() + 3600,
            }
        )
    finally:
        os.umask(old_umask)

    stored = json.loads((token_dir / "auth.json").read_text())
    assert stored["access_token"] == "access-token"
    assert replace_calls
    assert oct(os.stat(token_dir).st_mode & 0o777) == "0o700"
    assert oct(os.stat(token_dir / "auth.json").st_mode & 0o777) == "0o600"


def test_discovery_rejects_unexpected_endpoint():
    authenticator = XAIOAuthAuthenticator()

    with pytest.raises(XAIOAuthError, match="unexpected endpoint"):
        authenticator._validate_xai_endpoint("https://evil.example.com/oauth/token")

    with pytest.raises(XAIOAuthError, match="unexpected endpoint"):
        authenticator._validate_xai_endpoint("http://auth.x.ai/oauth/token")


def test_discover_returns_validated_xai_endpoints():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://auth.x.ai/.well-known/openid-configuration"
        return httpx.Response(
            200,
            json={
                "authorization_endpoint": "https://auth.x.ai/oauth/authorize",
                "token_endpoint": "https://auth.x.ai/oauth/token",
            },
        )

    authenticator = XAIOAuthAuthenticator(http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    assert authenticator._discover() == {
        "authorization_endpoint": "https://auth.x.ai/oauth/authorize",
        "token_endpoint": "https://auth.x.ai/oauth/token",
    }


def test_discover_requires_authorization_and_token_endpoints():
    authenticator = XAIOAuthAuthenticator(
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    )

    with pytest.raises(XAIOAuthError, match="missing endpoints"):
        authenticator._discover()


def test_discover_wraps_http_errors():
    authenticator = XAIOAuthAuthenticator(
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(500, text="discovery failed", request=request))
        )
    )

    with pytest.raises(XAIOAuthError) as exc_info:
        authenticator._discover()

    assert "xAI OAuth discovery request failed: 500 discovery failed" in str(exc_info.value)


def test_discover_wraps_invalid_json_response():
    authenticator = XAIOAuthAuthenticator(
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html>not-json</html>"))
        )
    )

    with pytest.raises(XAIOAuthError, match="discovery response was not valid JSON"):
        authenticator._discover()


def test_refresh_discovers_token_endpoint_when_auth_file_is_legacy(tmp_path, monkeypatch):
    token_dir, auth_file = _write_auth_file(
        tmp_path,
        {
            "access_token": "expired-token",
            "refresh_token": "refresh-token",
            "expires_at": time.time() - 1,
        },
    )
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "authorization_endpoint": "https://auth.x.ai/oauth/authorize",
                    "token_endpoint": "https://auth.x.ai/oauth/token",
                },
            )
        return httpx.Response(
            200,
            json={
                "access_token": "discovered-token",
                "refresh_token": "new-refresh-token",
                "expires_in": 3600,
            },
        )

    authenticator = XAIOAuthAuthenticator(http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    assert authenticator.get_access_token() == "discovered-token"
    stored = json.loads(auth_file.read_text())
    assert stored["token_endpoint"] == "https://auth.x.ai/oauth/token"


def test_exchange_token_rejects_non_object_response():
    authenticator = XAIOAuthAuthenticator(
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=["not", "an", "object"]))
        )
    )

    with pytest.raises(XAIOAuthError, match="was not an object"):
        authenticator._exchange_token("https://auth.x.ai/oauth/token", {})


def test_exchange_token_wraps_invalid_json_response():
    authenticator = XAIOAuthAuthenticator(
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html>not-json</html>"))
        )
    )

    with pytest.raises(XAIOAuthError, match="token response was not valid JSON"):
        authenticator._exchange_token("https://auth.x.ai/oauth/token", {})


def test_start_callback_server_falls_back_to_ephemeral_port(monkeypatch):
    calls = []
    real_server = xai_oauth_module._CallbackServer

    class FirstPortFailsCallbackServer(real_server):
        def __init__(self, server_address, handler_class):
            calls.append(server_address[1])
            if server_address[1] == xai_oauth_module.XAI_OAUTH_REDIRECT_PORT:
                raise OSError("port unavailable")
            super().__init__(server_address, handler_class)

    monkeypatch.setattr(xai_oauth_module, "_CallbackServer", FirstPortFailsCallbackServer)

    server, redirect_uri = XAIOAuthAuthenticator()._start_callback_server("state-value")
    try:
        assert calls == [xai_oauth_module.XAI_OAUTH_REDIRECT_PORT, 0]
        assert redirect_uri.startswith("http://127.0.0.1:")
        assert redirect_uri.endswith("/callback")
    finally:
        server.server_close()


def test_wait_for_callback_times_out_and_closes_server(monkeypatch):
    server, _ = XAIOAuthAuthenticator()._start_callback_server("state-value")
    monkeypatch.setattr(xai_oauth_module, "XAI_OAUTH_CALLBACK_TIMEOUT_SECONDS", 0)

    with pytest.raises(XAIOAuthError, match="Timed out"):
        XAIOAuthAuthenticator()._wait_for_callback(server)


def test_callback_handler_records_success_and_rejects_state_mismatch():
    authenticator = XAIOAuthAuthenticator()
    server, redirect_uri = authenticator._start_callback_server("expected-state")
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    response = httpx.get(f"{redirect_uri}?code=auth-code&state=expected-state")
    thread.join(timeout=5)

    assert response.status_code == 200
    assert server.callback_result == {
        "code": "auth-code",
        "state": "expected-state",
        "error": None,
        "error_description": None,
    }

    server, redirect_uri = authenticator._start_callback_server("expected-state")
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    response = httpx.get(f"{redirect_uri}?code=auth-code&state=wrong-state")
    thread.join(timeout=5)

    assert response.status_code == 400
    assert server.callback_result["state"] == "wrong-state"


def test_login_exchanges_authorization_code_and_persists_auth_record(monkeypatch):
    authenticator = XAIOAuthAuthenticator()
    fake_server = MagicMock()
    written_records = []

    class FakeUUID:
        def __init__(self, value):
            self.hex = value

    monkeypatch.setattr(
        xai_oauth_module.uuid,
        "uuid4",
        MagicMock(side_effect=[FakeUUID("state-value"), FakeUUID("nonce-value")]),
    )
    authenticator._read_auth_file = MagicMock(return_value=None)
    authenticator._discover = MagicMock(
        return_value={
            "authorization_endpoint": "https://auth.x.ai/oauth/authorize",
            "token_endpoint": "https://auth.x.ai/oauth/token",
        }
    )
    authenticator._pkce_pair = MagicMock(return_value=("verifier", "challenge"))
    authenticator._start_callback_server = MagicMock(return_value=(fake_server, "http://127.0.0.1:56121/callback"))
    authenticator._wait_for_callback = MagicMock(return_value={"state": "state-value", "code": "auth-code"})
    authenticator._exchange_token = MagicMock(
        return_value={
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
        }
    )
    authenticator._write_auth_file = MagicMock(side_effect=written_records.append)

    auth_data = authenticator.login(no_browser=True)

    authenticator._exchange_token.assert_called_once_with(
        "https://auth.x.ai/oauth/token",
        {
            "grant_type": "authorization_code",
            "code": "auth-code",
            "redirect_uri": "http://127.0.0.1:56121/callback",
            "client_id": XAI_OAUTH_CLIENT_ID,
            "code_verifier": "verifier",
        },
    )
    assert auth_data["access_token"] == "access-token"
    assert written_records == [auth_data]


def test_login_raises_on_callback_error_or_missing_code(monkeypatch):
    authenticator = XAIOAuthAuthenticator()

    class FakeUUID:
        hex = "state-value"

    monkeypatch.setattr(xai_oauth_module.uuid, "uuid4", MagicMock(return_value=FakeUUID()))
    authenticator._read_auth_file = MagicMock(return_value=None)
    authenticator._discover = MagicMock(
        return_value={
            "authorization_endpoint": "https://auth.x.ai/oauth/authorize",
            "token_endpoint": "https://auth.x.ai/oauth/token",
        }
    )
    authenticator._pkce_pair = MagicMock(return_value=("verifier", "challenge"))
    authenticator._start_callback_server = MagicMock(return_value=(MagicMock(), "http://127.0.0.1:56121/callback"))
    authenticator._wait_for_callback = MagicMock(
        return_value={
            "state": "state-value",
            "error": "access_denied",
            "error_description": "denied",
        }
    )

    with pytest.raises(XAIOAuthError, match="denied"):
        authenticator.login(no_browser=True)

    authenticator._wait_for_callback = MagicMock(return_value={"state": "state-value"})

    with pytest.raises(XAIOAuthError, match="no code returned"):
        authenticator.login(no_browser=True)


def test_pkce_pair_generates_s256_challenge():
    verifier, challenge = XAIOAuthAuthenticator()._pkce_pair()
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()

    assert challenge == expected
    assert "=" not in verifier
    assert "=" not in challenge


def test_build_authorize_url_contains_xai_oauth_parameters():
    authorize_url = XAIOAuthAuthenticator()._build_authorize_url(
        authorization_endpoint="https://auth.x.ai/oauth/authorize",
        redirect_uri="http://127.0.0.1:56121/callback",
        challenge="pkce-challenge",
        state="state-value",
        nonce="nonce-value",
    )
    parsed = urlparse(authorize_url)
    params = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "auth.x.ai"
    assert params["response_type"] == ["code"]
    assert params["client_id"] == [XAI_OAUTH_CLIENT_ID]
    assert params["scope"] == [XAI_OAUTH_SCOPE]
    assert params["code_challenge"] == ["pkce-challenge"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["state"] == ["state-value"]
    assert params["nonce"] == ["nonce-value"]


def test_get_llm_provider_uses_single_xai_provider(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "api-key")

    model, provider, api_key, api_base = get_llm_provider("xai/grok-4")

    assert model == "grok-4"
    assert provider == "xai"
    assert api_key == "api-key"
    assert api_base == "https://api.x.ai/v1"


def test_xai_oauth_alias_is_not_a_provider():
    with pytest.raises(Exception):
        get_llm_provider("xai_oauth/grok-4")


def test_chat_config_wraps_flagged_oauth_errors_as_authentication_error(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(tmp_path / "missing"))

    with pytest.raises(litellm.AuthenticationError) as exc_info:
        XAIChatConfig().validate_environment(
            headers={},
            model="grok-4",
            messages=[],
            optional_params={},
            litellm_params={"use_xai_oauth": True},
            api_key=None,
        )

    assert exc_info.value.llm_provider == "xai"
    assert "litellm xai-oauth login" in str(exc_info.value)


def test_chat_config_injects_flagged_oauth_token(tmp_path, monkeypatch):
    token_dir, _ = _write_auth_file(
        tmp_path,
        {
            "access_token": "chat-token",
            "refresh_token": "refresh-token",
            "expires_at": time.time() + 3600,
        },
    )
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))

    headers = XAIChatConfig().validate_environment(
        headers={},
        model="grok-4",
        messages=[],
        optional_params={},
        litellm_params={"use_xai_oauth": True},
        api_key=None,
    )

    assert headers["Authorization"] == "Bearer chat-token"


def test_chat_config_ignores_api_base_override_for_flagged_oauth(monkeypatch):
    monkeypatch.setenv("XAI_OAUTH_API_BASE", "https://api.x.ai/v1")

    url = XAIChatConfig().get_complete_url(
        api_base="https://attacker.example.com/v1",
        api_key=None,
        model="grok-4",
        optional_params={},
        litellm_params={"use_xai_oauth": True},
    )

    assert url == "https://api.x.ai/v1/chat/completions"


def test_chat_config_treats_blank_api_key_as_absent_for_flagged_oauth(tmp_path, monkeypatch):
    token_dir, _ = _write_auth_file(
        tmp_path,
        {
            "access_token": "stored-oauth-token",
            "refresh_token": "refresh-token",
            "expires_at": time.time() + 3600,
        },
    )
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))

    headers = XAIChatConfig().validate_environment(
        headers={},
        model="grok-4",
        messages=[],
        optional_params={},
        litellm_params={"use_xai_oauth": True},
        api_key="",
    )

    assert headers["Authorization"] == "Bearer stored-oauth-token"


def test_chat_config_allows_api_base_override_with_caller_api_key():
    headers = XAIChatConfig().validate_environment(
        headers={},
        model="grok-4",
        messages=[],
        optional_params={},
        litellm_params={"use_xai_oauth": True},
        api_key="caller-api-key",
    )
    url = XAIChatConfig().get_complete_url(
        api_base="https://custom.example.com/v1",
        api_key="caller-api-key",
        model="grok-4",
        optional_params={},
        litellm_params={"use_xai_oauth": True},
    )

    assert headers["Authorization"] == "Bearer caller-api-key"
    assert url == "https://custom.example.com/v1/chat/completions"


def test_chat_config_prioritizes_env_api_key_over_oauth_flag(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "env-api-key")

    headers = XAIChatConfig().validate_environment(
        headers={},
        model="grok-4",
        messages=[],
        optional_params={},
        litellm_params={"use_xai_oauth": True},
        api_key=None,
    )
    url = XAIChatConfig().get_complete_url(
        api_base="https://custom.example.com/v1",
        api_key=None,
        model="grok-4",
        optional_params={},
        litellm_params={"use_xai_oauth": True},
    )

    assert headers["Authorization"] == "Bearer env-api-key"
    assert url == "https://custom.example.com/v1/chat/completions"


def test_validate_environment_still_reports_xai_api_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "env-api-key")

    assert validate_environment("xai/grok-4") == {
        "keys_in_environment": True,
        "missing_keys": [],
    }


def test_xai_oauth_flag_uses_xai_optional_param_mapping():
    litellm_params = GenericLiteLLMParams(use_xai_oauth=True)
    optional_params = get_optional_params(
        model="grok-4",
        custom_llm_provider="xai",
        temperature=0.2,
        max_tokens=8,
    )

    assert optional_params["temperature"] == 0.2
    assert optional_params["max_tokens"] == 8
    assert litellm_params.use_xai_oauth is True
    assert "use_xai_oauth" not in optional_params


def test_responses_config_injects_flagged_oauth_bearer_token(tmp_path, monkeypatch):
    token_dir, _ = _write_auth_file(
        tmp_path,
        {
            "access_token": "responses-token",
            "refresh_token": "refresh-token",
            "expires_at": time.time() + 3600,
        },
    )
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))

    headers = XAIResponsesAPIConfig().validate_environment(
        headers={},
        model="grok-4",
        litellm_params=GenericLiteLLMParams(use_xai_oauth=True),
    )

    assert headers["Authorization"] == "Bearer responses-token"


def test_responses_config_endpoint_url_uses_oauth_authenticator(monkeypatch):
    monkeypatch.setenv("XAI_OAUTH_API_BASE", "https://xai.example.com/v1/")
    config = XAIResponsesAPIConfig()

    assert config.get_complete_url(api_base=None, litellm_params={"use_xai_oauth": True}) == (
        "https://xai.example.com/v1/responses"
    )
    assert (
        config.get_complete_url(
            api_base="https://custom.example.com/v1/",
            litellm_params={"use_xai_oauth": True},
        )
        == "https://xai.example.com/v1/responses"
    )
    assert (
        config.get_complete_url(
            api_base="https://custom.example.com/v1/",
            litellm_params={"api_key": "", "use_xai_oauth": True},
        )
        == "https://xai.example.com/v1/responses"
    )
    assert (
        config.get_complete_url(
            api_base="https://custom.example.com/v1/",
            litellm_params={"api_key": "caller-api-key"},
        )
        == "https://custom.example.com/v1/responses"
    )


def test_responses_config_wraps_flagged_oauth_errors_as_authentication_error(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(tmp_path / "missing"))

    with pytest.raises(litellm.AuthenticationError) as exc_info:
        XAIResponsesAPIConfig().validate_environment(
            headers={},
            model="grok-4",
            litellm_params=GenericLiteLLMParams(use_xai_oauth=True),
        )

    assert XAIResponsesAPIConfig().custom_llm_provider.value == "xai"
    assert exc_info.value.llm_provider == "xai"


def test_proxy_cli_xai_oauth_login_uses_single_authenticator(monkeypatch):
    from litellm.proxy.proxy_cli import run_server

    instances = []

    class FakeAuthenticator:
        auth_file = "/tmp/xai-oauth-auth.json"

        def __init__(self):
            instances.append(self)

        def login(self):
            return {"expires_at": 1234567890}

    monkeypatch.setattr("litellm.llms.xai.oauth.XAIOAuthAuthenticator", FakeAuthenticator)

    result = CliRunner().invoke(run_server, ["xai-oauth", "login"])

    assert result.exit_code == 0
    assert len(instances) == 1
    assert "Credentials saved to /tmp/xai-oauth-auth.json" in result.output
    assert "Access token expires at 1234567890" in result.output


# ---------------------------------------------------------------------------
# Multi-account support: per-deployment xai_oauth_token_file
# ---------------------------------------------------------------------------


def test_authenticator_accepts_explicit_absolute_auth_file(tmp_path):
    """__init__(auth_file=<abs path>) reads/writes that exact path."""
    custom = tmp_path / "custom-dir" / "alice.json"
    custom.parent.mkdir(parents=True)
    custom.write_text(
        json.dumps(
            {
                "access_token": "alice-token",
                "refresh_token": "refresh-token",
                "expires_at": time.time() + 3600,
            }
        )
    )

    auth = XAIOAuthAuthenticator(auth_file=str(custom))

    assert auth.auth_file == str(custom)
    assert auth.get_access_token() == "alice-token"


def test_authenticator_relative_auth_file_joins_token_dir(tmp_path, monkeypatch):
    """A relative auth_file is joined with the token_dir."""
    token_dir = tmp_path / "xai_oauth"
    token_dir.mkdir()
    (token_dir / "auth-bob.json").write_text(
        json.dumps(
            {
                "access_token": "bob-token",
                "refresh_token": "refresh-token",
                "expires_at": time.time() + 3600,
            }
        )
    )
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))

    auth = XAIOAuthAuthenticator(auth_file="auth-bob.json")

    assert auth.auth_file == str(token_dir / "auth-bob.json")
    assert auth.get_access_token() == "bob-token"


def test_authenticator_explicit_auth_file_overrides_env(tmp_path, monkeypatch):
    """Explicit auth_file wins over XAI_OAUTH_AUTH_FILE env var."""
    env_file = tmp_path / "env.json"
    env_file.write_text(
        json.dumps(
            {
                "access_token": "env-token",
                "refresh_token": "r",
                "expires_at": time.time() + 3600,
            }
        )
    )
    explicit_file = tmp_path / "explicit.json"
    explicit_file.write_text(
        json.dumps(
            {
                "access_token": "explicit-token",
                "refresh_token": "r",
                "expires_at": time.time() + 3600,
            }
        )
    )
    monkeypatch.setenv("XAI_OAUTH_AUTH_FILE", str(env_file))

    auth = XAIOAuthAuthenticator(auth_file=str(explicit_file))

    assert auth.get_access_token() == "explicit-token"


def test_chat_config_uses_per_deployment_token_file(tmp_path):
    """chat validate_environment picks the token from xai_oauth_token_file."""
    alice_file = tmp_path / "auth-alice.json"
    alice_file.write_text(
        json.dumps(
            {
                "access_token": "alice-chat-token",
                "refresh_token": "refresh-token",
                "expires_at": time.time() + 3600,
            }
        )
    )

    headers = XAIChatConfig().validate_environment(
        headers={},
        model="grok-4",
        messages=[],
        optional_params={},
        litellm_params={
            "use_xai_oauth": True,
            "xai_oauth_token_file": str(alice_file),
        },
        api_key=None,
    )

    assert headers["Authorization"] == "Bearer alice-chat-token"


def test_chat_config_multi_deployment_token_isolation(tmp_path):
    """Two deployments with different xai_oauth_token_file read different tokens."""
    alice = tmp_path / "auth-alice.json"
    alice.write_text(
        json.dumps(
            {
                "access_token": "alice-token",
                "refresh_token": "r",
                "expires_at": time.time() + 3600,
            }
        )
    )
    bob = tmp_path / "auth-bob.json"
    bob.write_text(
        json.dumps(
            {
                "access_token": "bob-token",
                "refresh_token": "r",
                "expires_at": time.time() + 3600,
            }
        )
    )

    headers_alice = XAIChatConfig().validate_environment(
        headers={},
        model="grok-4",
        messages=[],
        optional_params={},
        litellm_params={
            "use_xai_oauth": True,
            "xai_oauth_token_file": str(alice),
        },
        api_key=None,
    )
    headers_bob = XAIChatConfig().validate_environment(
        headers={},
        model="grok-4",
        messages=[],
        optional_params={},
        litellm_params={
            "use_xai_oauth": True,
            "xai_oauth_token_file": str(bob),
        },
        api_key=None,
    )

    assert headers_alice["Authorization"] == "Bearer alice-token"
    assert headers_bob["Authorization"] == "Bearer bob-token"


def test_responses_config_uses_per_deployment_token_file(tmp_path):
    """responses validate_environment picks the token from xai_oauth_token_file."""
    bob = tmp_path / "auth-bob.json"
    bob.write_text(
        json.dumps(
            {
                "access_token": "bob-responses-token",
                "refresh_token": "r",
                "expires_at": time.time() + 3600,
            }
        )
    )

    headers = XAIResponsesAPIConfig().validate_environment(
        headers={},
        model="grok-4",
        litellm_params=GenericLiteLLMParams(
            use_xai_oauth=True,
            xai_oauth_token_file=str(bob),
        ),
    )

    assert headers["Authorization"] == "Bearer bob-responses-token"


def test_proxy_cli_xai_oauth_login_with_account(monkeypatch, tmp_path):
    """`litellm xai-oauth login <account>` writes auth-<account>.json."""
    from litellm.proxy.proxy_cli import run_server

    captured = {}

    class FakeAuthenticator:
        def __init__(self, http_client=None, auth_file=None, token_dir=None):
            captured["auth_file"] = auth_file
            self.auth_file = auth_file or "/tmp/xai-oauth-auth.json"

        def login(self):
            return {"expires_at": 1234567890}

    monkeypatch.setattr("litellm.llms.xai.oauth.XAIOAuthAuthenticator", FakeAuthenticator)
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(tmp_path))

    result = CliRunner().invoke(run_server, ["xai-oauth", "login", "alice"])

    assert result.exit_code == 0
    expected = os.path.join(str(tmp_path), "auth-alice.json")
    assert captured["auth_file"] == expected
    assert expected in result.output
