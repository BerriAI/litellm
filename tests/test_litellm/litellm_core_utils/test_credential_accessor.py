"""CredentialAccessor.upsert_credentials: the per-reload sync of DB credentials into litellm.credential_list."""

import time

import pytest

import litellm
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.types.utils import CredentialItem


def _cred(name: str, api_key: str) -> CredentialItem:
    return CredentialItem(credential_name=name, credential_values={"api_key": api_key}, credential_info={})


def test_upsert_credentials_replaces_by_name_in_place_and_appends_new(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "credential_list", [_cred("a", "old-a"), _cred("b", "old-b"), _cred("c", "old-c")])

    CredentialAccessor.upsert_credentials([_cred("b", "new-b"), _cred("d", "new-d"), _cred("a", "new-a")])

    assert [(c.credential_name, c.credential_values["api_key"]) for c in litellm.credential_list] == [
        ("a", "new-a"),
        ("b", "new-b"),
        ("c", "old-c"),
        ("d", "new-d"),
    ]


def test_upsert_credentials_scales_linearly_with_credential_count(monkeypatch: pytest.MonkeyPatch) -> None:
    # The proxy re-upserts every DB credential on each config reload (default every 30s) on the serving
    # event loop. 20k credentials took ~40s with the previous nested scan; linear takes milliseconds.
    n = 20_000
    monkeypatch.setattr(litellm, "credential_list", [_cred(f"cred-{i}", f"k{i}") for i in range(n)])
    refreshed = [_cred(f"cred-{i}", f"k{i}-v2") for i in range(n)]

    started = time.perf_counter()
    CredentialAccessor.upsert_credentials(refreshed)
    elapsed = time.perf_counter() - started

    assert len(litellm.credential_list) == n
    assert litellm.credential_list[n - 1].credential_values["api_key"] == f"k{n - 1}-v2"
    assert elapsed < 2.0, f"upsert of {n} credentials took {elapsed:.1f}s; expected linear-time behaviour"
