import json
import os
from pathlib import Path

import httpx
import pytest
import respx
from click.testing import CliRunner

import litellm
from litellm import Router
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.xai.chat.transformation import XAIChatConfig
from litellm.llms.xai.oauth import XAIOAuthAuthenticator, XAIOAuthError, oauth_auth_file_for_account
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams

FAR_FUTURE_EXPIRY = 4_102_444_800
SPENDING_LIMIT_BODY = '{"code":"personal-team-blocked:spending-limit","error":"spending limit reached"}'


@pytest.fixture(autouse=True)
def _oauth_only(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_OAUTH_AUTH_FILE", raising=False)
    monkeypatch.setattr(litellm, "xai_key", None)
    monkeypatch.setattr(litellm, "api_key", None)


@pytest.fixture
def token_dir(tmp_path, monkeypatch):
    directory = tmp_path / "xai_oauth"
    directory.mkdir()
    monkeypatch.setenv("XAI_OAUTH_TOKEN_DIR", str(directory))
    return directory


def _write_token(path: Path, access_token: str) -> Path:
    path.write_text(
        json.dumps({"access_token": access_token, "refresh_token": "refresh", "expires_at": FAR_FUTURE_EXPIRY})
    )
    return path


def _chat_headers(token_file: str) -> dict:
    return XAIChatConfig().validate_environment(
        headers={},
        model="grok-4",
        messages=[],
        optional_params={},
        litellm_params={"use_xai_oauth": True, "xai_oauth_token_file": token_file},
        api_key=None,
    )


def test_absolute_auth_file_inside_token_dir_is_read(token_dir):
    alice = _write_token(token_dir / "alice.json", "alice-token")

    auth = XAIOAuthAuthenticator(auth_file=str(alice))

    assert auth.auth_file == os.path.realpath(alice)
    assert auth.get_access_token() == "alice-token"


def test_relative_auth_file_resolves_inside_token_dir(token_dir):
    _write_token(token_dir / "auth-bob.json", "bob-token")

    assert XAIOAuthAuthenticator(auth_file="auth-bob.json").get_access_token() == "bob-token"


@pytest.mark.parametrize("auth_file", ["../outside.json", ".", "sub/../../outside.json"])
def test_relative_auth_file_escaping_token_dir_is_rejected(token_dir, auth_file):
    _write_token(token_dir.parent / "outside.json", "outside-token")

    with pytest.raises(XAIOAuthError, match="token directory"):
        XAIOAuthAuthenticator(auth_file=auth_file)


def test_absolute_auth_file_outside_token_dir_is_rejected(token_dir):
    outside = _write_token(token_dir.parent / "outside.json", "outside-token")

    with pytest.raises(XAIOAuthError, match="token directory"):
        XAIOAuthAuthenticator(auth_file=str(outside))


def test_symlink_inside_token_dir_pointing_outside_is_rejected(token_dir):
    outside = _write_token(token_dir.parent / "outside.json", "outside-token")
    (token_dir / "link.json").symlink_to(outside)

    with pytest.raises(XAIOAuthError, match="token directory"):
        XAIOAuthAuthenticator(auth_file="link.json")


def test_operator_env_auth_file_outside_token_dir_keeps_working(token_dir, monkeypatch):
    env_file = _write_token(token_dir.parent / "env.json", "env-token")
    monkeypatch.setenv("XAI_OAUTH_AUTH_FILE", str(env_file))

    assert XAIOAuthAuthenticator().get_access_token() == "env-token"


def test_explicit_auth_file_wins_over_env_auth_file(token_dir, monkeypatch):
    monkeypatch.setenv("XAI_OAUTH_AUTH_FILE", str(_write_token(token_dir / "env.json", "env-token")))
    explicit = _write_token(token_dir / "explicit.json", "explicit-token")

    assert XAIOAuthAuthenticator(auth_file=str(explicit)).get_access_token() == "explicit-token"


@pytest.mark.parametrize("account", ["alice", "Bob_1-2"])
def test_account_login_file_lands_in_token_dir(token_dir, account):
    auth = XAIOAuthAuthenticator(auth_file=oauth_auth_file_for_account(account))

    assert auth.auth_file == os.path.realpath(token_dir / f"auth-{account}.json")


@pytest.mark.parametrize("account", ["", ".", "..", "foo/bar", "../alice", "alice.json", "foo\\bar", "alice/../bob"])
def test_unsafe_account_names_are_rejected(account):
    with pytest.raises(ValueError, match="account"):
        oauth_auth_file_for_account(account)


def test_cli_login_rejects_unsafe_account_before_starting_oauth(token_dir):
    from litellm.proxy.proxy_cli import run_server

    result = CliRunner().invoke(run_server, ["xai-oauth", "login", "../alice"])

    assert result.exit_code == 2
    assert "xAI OAuth account must match" in result.output
    assert list(token_dir.iterdir()) == []


def test_cli_login_rejects_extra_arguments(token_dir):
    from litellm.proxy.proxy_cli import run_server

    result = CliRunner().invoke(run_server, ["xai-oauth", "login", "alice", "bob"])

    assert result.exit_code == 2
    assert "Unknown command" in result.output


def test_chat_deployments_authenticate_with_their_own_token_files(token_dir):
    alice = _write_token(token_dir / "auth-alice.json", "alice-token")
    bob = _write_token(token_dir / "auth-bob.json", "bob-token")

    assert _chat_headers(str(alice))["Authorization"] == "Bearer alice-token"
    assert _chat_headers(str(bob))["Authorization"] == "Bearer bob-token"


def test_chat_token_file_outside_token_dir_raises_authentication_error(token_dir):
    outside = _write_token(token_dir.parent / "outside.json", "outside-token")

    with pytest.raises(litellm.AuthenticationError, match="token directory"):
        _chat_headers(str(outside))


def test_responses_deployment_authenticates_with_its_token_file(token_dir):
    bob = _write_token(token_dir / "auth-bob.json", "bob-responses-token")

    headers = XAIResponsesAPIConfig().validate_environment(
        headers={},
        model="grok-4",
        litellm_params=GenericLiteLLMParams(use_xai_oauth=True, xai_oauth_token_file=str(bob)),
    )

    assert headers["Authorization"] == "Bearer bob-responses-token"


@pytest.mark.parametrize("config", [XAIChatConfig(), XAIResponsesAPIConfig()])
@pytest.mark.parametrize(
    ("status_code", "message", "expected_status"),
    [
        (403, SPENDING_LIMIT_BODY, 429),
        (403, '{"error":"model access denied"}', 403),
        (400, SPENDING_LIMIT_BODY, 400),
    ],
)
def test_error_class_reports_spending_limit_403_as_rate_limit(config, status_code, message, expected_status):
    try:
        error = config.get_error_class(error_message=message, status_code=status_code, headers={})
    except BaseLLMException as raised:
        error = raised

    assert error.status_code == expected_status
    assert error.message == message


def _chat_completion_body(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "grok-4",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _xai_by_account(request: httpx.Request) -> httpx.Response:
    if request.headers["Authorization"] == "Bearer alice-token":
        return httpx.Response(403, text=SPENDING_LIMIT_BODY)
    return httpx.Response(200, json=_chat_completion_body(f"served with {request.headers['Authorization']}"))


def test_spending_limit_403_surfaces_as_rate_limit_error(token_dir):
    alice = _write_token(token_dir / "auth-alice.json", "alice-token")

    with respx.mock(assert_all_called=True) as router:
        router.post("https://api.x.ai/v1/chat/completions").mock(side_effect=_xai_by_account)
        with pytest.raises(litellm.RateLimitError, match="spending-limit"):
            litellm.completion(
                model="xai/grok-4",
                messages=[{"role": "user", "content": "hi"}],
                use_xai_oauth=True,
                xai_oauth_token_file=str(alice),
            )


def test_router_fails_over_to_next_account_when_one_hits_its_spending_limit(token_dir):
    alice = _write_token(token_dir / "auth-alice.json", "alice-token")
    bob = _write_token(token_dir / "auth-bob.json", "bob-token")
    llm_router = Router(
        model_list=[
            {
                "model_name": "grok",
                "litellm_params": {
                    "model": "xai/grok-4",
                    "use_xai_oauth": True,
                    "xai_oauth_token_file": str(account_file),
                    "order": order,
                },
            }
            for order, account_file in ((1, alice), (2, bob))
        ],
        num_retries=1,
        retry_after=0,
    )

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://api.x.ai/v1/chat/completions").mock(side_effect=_xai_by_account)
        response = llm_router.completion(model="grok", messages=[{"role": "user", "content": "hi"}])

    assert response.choices[0].message.content == "served with Bearer bob-token"
    assert [call.request.headers["Authorization"] for call in route.calls] == [
        "Bearer alice-token",
        "Bearer bob-token",
    ]
