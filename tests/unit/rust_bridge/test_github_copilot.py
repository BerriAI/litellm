from typing import Final
from unittest.mock import patch

import pytest

import litellm
from litellm.exceptions import AuthenticationError, CallerCredentialAuthenticationError
from litellm.llms.github_copilot.common_utils import DEFAULT_GITHUB_COPILOT_API_BASE, GetAPIKeyError
from litellm.llms.github_copilot.per_user_auth import GITHUB_COPILOT_USER_SESSION_KWARG_KEY, GithubCopilotUserSession
from litellm.rust_bridge import github_copilot
from litellm.types.utils import CredentialItem


class SharedLogin:
    def __init__(self, token: str = "shared-token", api_base: str | None = "https://shared.githubcopilot.com/") -> None:
        self.token: Final = token
        self.api_base: Final = api_base

    def get_api_key(self) -> str:
        return self.token

    def get_api_base(self) -> str | None:
        return self.api_base


class UnusedSharedLogin:
    def get_api_key(self) -> str:
        raise AssertionError("per-user calls must never read the shared device login")

    def get_api_base(self) -> str | None:
        raise AssertionError("per-user calls must never read the shared device login")


class RejectedSharedLogin(SharedLogin):
    def get_api_key(self) -> str:
        raise GetAPIKeyError(message="Failed to refresh API key", status_code=401)


def _per_user_credentials() -> list[CredentialItem]:
    return [
        CredentialItem(
            credential_name="copilot-cred",
            credential_values={"github_copilot_auth_type": "per_user_oauth"},
            credential_info={},
        )
    ]


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        ("https://shared.githubcopilot.com/", "https://shared.githubcopilot.com"),
        (None, DEFAULT_GITHUB_COPILOT_API_BASE),
    ],
)
def test_shared_mode_hands_rust_the_device_login_session(api_base: str | None, expected: str) -> None:
    assert github_copilot.resolve({}, SharedLogin(api_base=api_base)) == ("shared-token", expected)


def test_attached_per_user_session_outranks_the_shared_login() -> None:
    arguments: Final = {
        "litellm_credential_name": "copilot-cred",
        GITHUB_COPILOT_USER_SESSION_KWARG_KEY: GithubCopilotUserSession(
            token="user-token", api_base="https://tenant.githubcopilot.com"
        ),
    }

    with patch.object(litellm, "credential_list", _per_user_credentials()):
        resolved: Final = github_copilot.resolve(arguments, UnusedSharedLogin())

    assert resolved == ("user-token", "https://tenant.githubcopilot.com")


def test_per_user_call_without_a_session_never_falls_back_to_the_shared_login() -> None:
    with (
        patch.object(litellm, "credential_list", _per_user_credentials()),
        pytest.raises(CallerCredentialAuthenticationError),
    ):
        github_copilot.resolve({"litellm_credential_name": "copilot-cred"}, UnusedSharedLogin())


def test_session_shaped_request_json_is_not_a_session() -> None:
    forged: Final = {
        GITHUB_COPILOT_USER_SESSION_KWARG_KEY: {"token": "forged", "api_base": "https://attacker.example"},
    }

    assert github_copilot.resolve(forged, SharedLogin()) == ("shared-token", "https://shared.githubcopilot.com")


def test_rejected_shared_login_is_an_authentication_error() -> None:
    with pytest.raises(AuthenticationError):
        github_copilot.resolve({}, RejectedSharedLogin())


def test_other_providers_get_no_copilot_session() -> None:
    assert github_copilot.session("anthropic/claude-sonnet-4-5", None, {}) is None


def test_session_exchanges_the_callers_github_connection_before_resolving() -> None:
    exchanged: Final = GithubCopilotUserSession(token="user-token", api_base="https://tenant.githubcopilot.com")
    arguments: Final[dict[str, object]] = {
        "litellm_credential_name": "copilot-cred",
        "secret_fields": {
            "user_provider_credentials_user_id": "user-1",
            "user_provider_credentials": {"copilot-cred": "gho_user"},
        },
    }

    with (
        patch.object(litellm, "credential_list", _per_user_credentials()),
        patch(
            "litellm.llms.github_copilot.per_user_auth.exchange_github_token", return_value=exchanged
        ) as exchange,
    ):
        resolved: Final = github_copilot.session("github_copilot/claude-sonnet-4-5", None, arguments)

    assert resolved == ("user-token", "https://tenant.githubcopilot.com")
    exchange.assert_called_once_with(user_id="user-1", github_token="gho_user", credential_name="copilot-cred")
