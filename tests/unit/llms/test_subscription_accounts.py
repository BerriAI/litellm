import base64
import json
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.constants import SUBSCRIPTION_BACKED_PROVIDERS
from litellm.llms.chatgpt.common_utils import OPENAI_AUTH_CLAIM_KEY
from litellm.llms.subscription_accounts import SUBSCRIPTION_ACCOUNT_ID_RESOLVERS, resolve_subscription_account_id
from litellm.types.utils import CredentialItem


def _write_login(path: Path, account_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"account_id": account_id}))
    return path


def _access_token_for(account_id: str) -> str:
    claims: Final = json.dumps({OPENAI_AUTH_CLAIM_KEY: {"chatgpt_account_id": account_id}}).encode()
    payload: Final = base64.urlsafe_b64encode(claims).decode().rstrip("=")
    return f"header.{payload}.signature"


@pytest.fixture
def signed_in_chatgpt_token_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
    _write_login(tmp_path / "auth.json", "acct-from-token-file")
    return tmp_path


def test_every_subscription_backed_provider_has_an_account_id_resolver() -> None:
    assert frozenset(SUBSCRIPTION_ACCOUNT_ID_RESOLVERS) == SUBSCRIPTION_BACKED_PROVIDERS


def test_chatgpt_resolves_to_the_account_in_the_signed_in_token_file(signed_in_chatgpt_token_dir: Path) -> None:
    assert resolve_subscription_account_id("chatgpt", {"model": "chatgpt/gpt-5.6-terra"}) == "acct-from-token-file"


def test_each_deployment_takes_the_account_of_its_own_auth_file(signed_in_chatgpt_token_dir: Path) -> None:
    first: Final = _write_login(signed_in_chatgpt_token_dir / "team-a" / "auth.json", "acct-team-a")
    second: Final = _write_login(signed_in_chatgpt_token_dir / "team-b" / "auth.json", "acct-team-b")

    assert resolve_subscription_account_id("chatgpt", {"chatgpt_auth_file": str(first)}) == "acct-team-a"
    assert resolve_subscription_account_id("chatgpt", {"chatgpt_auth_file": str(second)}) == "acct-team-b"


def test_a_configured_access_token_resolves_to_its_own_account(signed_in_chatgpt_token_dir: Path) -> None:
    litellm_params: Final = {"model": "chatgpt/gpt-5.6-terra", "api_key": _access_token_for("acct-from-api-key")}

    assert resolve_subscription_account_id("chatgpt", litellm_params) == "acct-from-api-key"


def test_a_metered_provider_resolves_to_no_account_even_with_a_chatgpt_login(
    signed_in_chatgpt_token_dir: Path,
) -> None:
    assert resolve_subscription_account_id("openai", {"model": "openai/gpt-5.6-terra"}) is None


def test_chatgpt_without_a_login_resolves_to_no_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty_dir: Final = tmp_path / "no-login"
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(empty_dir))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
    assert resolve_subscription_account_id("chatgpt", {"model": "chatgpt/gpt-5.6-terra"}) is None


def test_resolving_an_account_leaves_the_login_file_untouched(signed_in_chatgpt_token_dir: Path) -> None:
    login: Final = signed_in_chatgpt_token_dir / "no-saved-account" / "auth.json"
    login.parent.mkdir(parents=True)
    saved: Final = json.dumps({"access_token": _access_token_for("acct-derived"), "refresh_token": "rt-current"})
    login.write_text(saved)

    assert resolve_subscription_account_id("chatgpt", {"chatgpt_auth_file": str(login)}) == "acct-derived"
    assert login.read_text() == saved


@pytest.mark.parametrize("credential_field", ["chatgpt_auth_file", "api_key"])
def test_a_named_credential_resolves_to_the_account_it_signs_in_as(
    credential_field: str, signed_in_chatgpt_token_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    login: Final = _write_login(signed_in_chatgpt_token_dir / "named" / "auth.json", "acct-named-credential")
    values: Final = {
        "chatgpt_auth_file": {"chatgpt_auth_file": str(login)},
        "api_key": {"api_key": _access_token_for("acct-named-credential")},
    }[credential_field]
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [CredentialItem(credential_name="chatgpt-team", credential_values=values, credential_info={})],
    )
    litellm_params: Final = {"model": "chatgpt/gpt-5.6-terra", "litellm_credential_name": "chatgpt-team"}

    assert resolve_subscription_account_id("chatgpt", litellm_params) == "acct-named-credential"


@pytest.mark.parametrize(
    ("api_key", "expected_account"),
    [("oidc/env/CHATGPT_TEAM_TOKEN", "acct-from-oidc-env"), ("oidc/env/CHATGPT_UNSET_TOKEN", None)],
)
def test_an_oidc_env_api_key_resolves_to_the_account_of_the_token_it_points_to(
    api_key: str, expected_account: str | None, signed_in_chatgpt_token_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHATGPT_TEAM_TOKEN", _access_token_for("acct-from-oidc-env"))
    monkeypatch.delenv("CHATGPT_UNSET_TOKEN", raising=False)
    litellm_params: Final = {"model": "chatgpt/gpt-5.6-terra", "api_key": api_key}

    assert resolve_subscription_account_id("chatgpt", litellm_params) == expected_account
