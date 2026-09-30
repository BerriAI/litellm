import time

import pytest

import litellm
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.types.utils import CredentialItem


def _credential(name: str, api_key: str) -> CredentialItem:
    return CredentialItem(credential_name=name, credential_values={"api_key": api_key}, credential_info={})


def _names_and_keys() -> list[tuple[str, str]]:
    return [(cred.credential_name, cred.credential_values["api_key"]) for cred in litellm.credential_list]


def test_upsert_credentials_replaces_by_name_in_place_and_appends_new(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm, "credential_list", [_credential("a", "old-a"), _credential("b", "old-b"), _credential("c", "old-c")]
    )

    CredentialAccessor.upsert_credentials(
        [_credential("b", "new-b"), _credential("d", "new-d"), _credential("a", "new-a")]
    )

    assert _names_and_keys() == [("a", "new-a"), ("b", "new-b"), ("c", "old-c"), ("d", "new-d")]


def test_upsert_credentials_replaces_the_first_duplicate_that_find_credential_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "credential_list", [_credential("a", "first"), _credential("a", "second")])

    CredentialAccessor.upsert_credentials([_credential("a", "from-db")])

    assert _names_and_keys() == [("a", "from-db"), ("a", "second")]
    assert CredentialAccessor.get_credential_values("a") == {"api_key": "from-db"}


def test_upsert_credentials_scales_linearly_with_credential_count(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 20_000
    monkeypatch.setattr(litellm, "credential_list", [_credential(f"cred-{i}", f"k{i}") for i in range(n)])
    refreshed = [_credential(f"cred-{i}", f"k{i}-v2") for i in range(n)]

    started = time.perf_counter()
    CredentialAccessor.upsert_credentials(refreshed)
    elapsed = time.perf_counter() - started

    assert litellm.credential_list == refreshed
    assert elapsed < 2.0, f"upsert of {n} credentials took {elapsed:.1f}s; the per-reload sync must stay linear"
