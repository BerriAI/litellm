import json
import os
import time

import pytest
from click.testing import CliRunner

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import _map_openai_exception
from litellm.llms.xai.chat.transformation import XAIChatConfig
from litellm.llms.xai.oauth import XAIOAuthAuthenticator, XAIOAuthError
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

    auth = XAIOAuthAuthenticator(auth_file=str(custom), token_dir=str(custom.parent))

    assert auth.auth_file == os.path.realpath(str(custom))
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

    assert auth.auth_file == os.path.realpath(str(token_dir / "auth-bob.json"))
    assert auth.get_access_token() == "bob-token"


def test_authenticator_rejects_relative_path_escaping_token_dir(tmp_path, monkeypatch):
    token_dir = tmp_path / "xai_oauth"
    token_dir.mkdir()
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))

    with pytest.raises(XAIOAuthError, match="token directory"):
        XAIOAuthAuthenticator(auth_file="../secret.json")


def test_authenticator_rejects_absolute_path_outside_token_dir(tmp_path, monkeypatch):
    token_dir = tmp_path / "xai_oauth"
    token_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))

    with pytest.raises(XAIOAuthError, match="token directory"):
        XAIOAuthAuthenticator(auth_file=str(outside))


def test_authenticator_rejects_dotdot_absolute_path_outside_token_dir(tmp_path, monkeypatch):
    token_dir = tmp_path / "xai_oauth"
    token_dir.mkdir()
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(token_dir))
    escaped = str((token_dir / ".." / "outside.json").resolve())

    with pytest.raises(XAIOAuthError, match="token directory"):
        XAIOAuthAuthenticator(auth_file=escaped)


def test_oauth_auth_file_for_account_builds_path_inside_token_dir(tmp_path):
    from litellm.llms.xai.oauth import oauth_auth_file_for_account

    assert oauth_auth_file_for_account("alice", str(tmp_path)) == os.path.realpath(
        os.path.join(str(tmp_path), "auth-alice.json")
    )
    assert oauth_auth_file_for_account("Bob_1-2", str(tmp_path)) == os.path.realpath(
        os.path.join(str(tmp_path), "auth-Bob_1-2.json")
    )


@pytest.mark.parametrize(
    "account",
    ["", ".", "..", "foo/bar", "../alice", "alice.json", "foo\\bar", "alice/../bob"],
)
def test_oauth_auth_file_for_account_rejects_unsafe_names(account, tmp_path):
    from litellm.llms.xai.oauth import oauth_auth_file_for_account

    with pytest.raises(ValueError, match="account"):
        oauth_auth_file_for_account(account, str(tmp_path))


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

    auth = XAIOAuthAuthenticator(auth_file=str(explicit_file), token_dir=str(tmp_path))

    assert auth.get_access_token() == "explicit-token"


def test_chat_config_uses_per_deployment_token_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(tmp_path))
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


def test_chat_config_multi_deployment_token_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(tmp_path))
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


def test_responses_config_uses_per_deployment_token_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(tmp_path))
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


def test_proxy_cli_xai_oauth_login_rejects_unsafe_account(monkeypatch, tmp_path):
    from litellm.proxy.proxy_cli import run_server

    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(tmp_path))
    result = CliRunner().invoke(run_server, ["xai-oauth", "login", "../alice"])

    assert result.exit_code != 0
    assert result.exception is not None


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


def test_spending_limit_403_on_non_xai_does_not_map_to_rate_limit_error():
    class FakeHTTPError(Exception):
        def __init__(self):
            self.status_code = 403
            self.message = "personal-team-blocked:spending-limit"
            self.response = None
            super().__init__(self.message)

    with pytest.raises(litellm.APIError) as exc_info:
        _map_openai_exception(
            model="gpt-4",
            original_exception=FakeHTTPError(),
            custom_llm_provider="openai",
            error_str="personal-team-blocked:spending-limit",
            exception_type="APIStatusError",
            exception_provider="OpenAIException",
            extra_information="",
        )
    assert not isinstance(exc_info.value, litellm.RateLimitError)
    assert exc_info.value.status_code == 403
