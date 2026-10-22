import base64
from typing import Final

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error, Ok
from litellm.proxy._experimental.mcp_server.state_tokens import StateTokenError, open_state, seal_state


@pytest.fixture(autouse=True)
def state_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "synthetic-shared-pagination-key")


def test_state_is_portable_repeatable_and_preserves_json() -> None:
    value: Final = {"caller": "user-a", "upstream": {"cursor": "opaque+/=", "offset": 3}, "revision": "r1"}
    sealed: Final = seal_state(value, purpose="pagination", expires_at=200, now=100)
    assert isinstance(sealed, Ok)
    assert open_state(sealed.ok, purpose="pagination", now=150) == Ok(value)
    assert open_state(sealed.ok, purpose="pagination", now=199) == Ok(value)
    assert "user-a" not in sealed.ok


def test_sealing_the_same_state_uses_distinct_nonces() -> None:
    first: Final = seal_state("same", purpose="pagination", expires_at=200, now=100)
    second: Final = seal_state("same", purpose="pagination", expires_at=200, now=100)
    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert first.ok != second.ok
    assert open_state(first.ok, purpose="pagination", now=101) == Ok("same")
    assert open_state(second.ok, purpose="pagination", now=101) == Ok("same")


@pytest.mark.parametrize("purpose", ("continuation", "pagination-other", ""))
def test_state_cannot_be_opened_for_another_purpose(purpose: str) -> None:
    sealed: Final = seal_state("private", purpose="pagination", expires_at=200, now=100)
    assert isinstance(sealed, Ok)
    assert open_state(sealed.ok, purpose=purpose, now=100) == Error(StateTokenError.INVALID)


def test_rotating_the_key_invalidates_existing_state(monkeypatch: pytest.MonkeyPatch) -> None:
    sealed: Final = seal_state("private", purpose="pagination", expires_at=200, now=100)
    assert isinstance(sealed, Ok)
    monkeypatch.setenv("LITELLM_SALT_KEY", "different-synthetic-key")
    assert open_state(sealed.ok, purpose="pagination", now=100) == Error(StateTokenError.INVALID)


@pytest.mark.parametrize("missing", (True, False))
def test_master_key_cannot_replace_missing_or_empty_salt(monkeypatch: pytest.MonkeyPatch, missing: bool) -> None:
    monkeypatch.setenv("LITELLM_MASTER_KEY", "synthetic-master-key")
    if missing:
        monkeypatch.delenv("LITELLM_SALT_KEY")
    else:
        monkeypatch.setenv("LITELLM_SALT_KEY", "")
    assert seal_state("private", purpose="pagination", expires_at=200, now=100) == Error(StateTokenError.MISSING_KEY)
    assert open_state("forged", purpose="pagination", now=100) == Error(StateTokenError.MISSING_KEY)


@pytest.mark.parametrize("now", (200, 201))
def test_state_expires_at_the_deadline(now: int) -> None:
    sealed: Final = seal_state("private", purpose="pagination", expires_at=200, now=100)
    assert isinstance(sealed, Ok)
    assert open_state(sealed.ok, purpose="pagination", now=now) == Error(StateTokenError.EXPIRED)
    assert seal_state("private", purpose="pagination", expires_at=200, now=now) == Error(StateTokenError.EXPIRED)


def test_altered_ciphertext_is_rejected() -> None:
    sealed: Final = seal_state("private", purpose="pagination", expires_at=200, now=100)
    assert isinstance(sealed, Ok)
    prefix, encoded = sealed.ok.split(".", 1)
    raw: Final = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    altered: Final = bytes((raw[0] ^ 1,)) + raw[1:]
    token: Final = prefix + "." + base64.urlsafe_b64encode(altered).decode("ascii").rstrip("=")
    assert open_state(token, purpose="pagination", now=100) == Error(StateTokenError.INVALID)


@pytest.mark.parametrize(
    "token", ("", "forged", "mcp_state_v2.abc", "mcp_state_v1.!", "mcp_state_v1.YQ", "mcp_state_v1.é")
)
def test_malformed_state_is_rejected(token: str) -> None:
    assert open_state(token, purpose="pagination", now=100) == Error(StateTokenError.INVALID)


def test_excessive_state_is_rejected() -> None:
    assert seal_state("x" * 65536, purpose="pagination", expires_at=200, now=100) == Error(StateTokenError.TOO_LARGE)
    assert seal_state("x" * 50000, purpose="pagination", expires_at=200, now=100) == Error(StateTokenError.TOO_LARGE)
    assert open_state("x" * 65537, purpose="pagination", now=100) == Error(StateTokenError.TOO_LARGE)


def test_empty_purpose_cannot_mint_state() -> None:
    assert seal_state("private", purpose="", expires_at=200, now=100) == Error(StateTokenError.INVALID)
