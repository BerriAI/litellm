import json
import os
import time

import pytest
from click.testing import CliRunner

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import _map_openai_exception
from litellm.llms.xai.chat.transformation import XAIChatConfig
from litellm.llms.xai.oauth import XAIOAuthAuthenticator
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams


def test_authenticator_accepts_explicit_absolute_auth_file(tmp_path):
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
    from litellm.proxy.proxy_cli import run_server

    captured: dict[str, str | None] = {}

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


def test_spending_limit_403_maps_to_rate_limit_error():
    class FakeHTTPError(Exception):
        def __init__(self):
            self.status_code = 403
            self.message = "personal-team-blocked:spending-limit"
            self.response = None
            super().__init__(self.message)

    with pytest.raises(litellm.RateLimitError):
        _map_openai_exception(
            model="grok-4",
            original_exception=FakeHTTPError(),
            custom_llm_provider="xai",
            error_str="personal-team-blocked:spending-limit",
            exception_type="APIStatusError",
            exception_provider="XaiException",
            extra_information="",
        )
