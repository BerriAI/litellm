import json
from pathlib import Path
from typing import Final

import pytest

from litellm.constants import SUBSCRIPTION_BACKED_PROVIDERS
from litellm.llms.subscription_accounts import SUBSCRIPTION_ACCOUNT_ID_RESOLVERS, resolve_subscription_account_id


@pytest.fixture
def signed_in_chatgpt_token_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
    (tmp_path / "auth.json").write_text(json.dumps({"account_id": "acct-from-token-file"}))
    return tmp_path


def test_every_subscription_backed_provider_has_an_account_id_resolver() -> None:
    assert frozenset(SUBSCRIPTION_ACCOUNT_ID_RESOLVERS) == SUBSCRIPTION_BACKED_PROVIDERS


def test_chatgpt_resolves_to_the_account_in_the_signed_in_token_file(signed_in_chatgpt_token_dir: Path) -> None:
    assert resolve_subscription_account_id("chatgpt") == "acct-from-token-file"


def test_a_metered_provider_resolves_to_no_account_even_with_a_chatgpt_login(
    signed_in_chatgpt_token_dir: Path,
) -> None:
    assert resolve_subscription_account_id("openai") is None


def test_chatgpt_without_a_login_resolves_to_no_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty_dir: Final = tmp_path / "no-login"
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(empty_dir))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)
    assert resolve_subscription_account_id("chatgpt") is None
