"""Unit tests for Azure Entra ID database token minting.

These tests inject a fake credential, so azure-identity does NOT need to be
installed to run them.

Run:
    uv run pytest tests/test_litellm/proxy/db/test_azure_entra_db_token.py -v
"""

import urllib.parse
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from litellm.proxy.auth import azure_entra_db_token
from litellm.proxy.auth.azure_entra_db_token import (
    AZURE_DB_SCOPE,
    generate_azure_entra_db_token,
)


def _fake_credential(token_value: str) -> MagicMock:
    cred = MagicMock()
    cred.get_token.return_value = SimpleNamespace(token=token_value)
    return cred


def test_uses_injected_credential_and_correct_scope():
    cred = _fake_credential("header.payload.sig")
    token = generate_azure_entra_db_token(credential=cred)
    cred.get_token.assert_called_once_with(AZURE_DB_SCOPE)
    # A JWT uses the url-safe base64 alphabet, so quoting is a no-op.
    assert token == "header.payload.sig"


def test_url_quotes_reserved_characters():
    # A token containing URL-reserved characters must be quoted so it cannot
    # corrupt the assembled postgresql:// URL.
    cred = _fake_credential("ab/cd=ef&gh")
    token = generate_azure_entra_db_token(credential=cred)
    assert token == urllib.parse.quote("ab/cd=ef&gh", safe="")
    assert "/" not in token and "&" not in token and "=" not in token


def test_parity_args_do_not_affect_token():
    cred = _fake_credential("tok")
    token = generate_azure_entra_db_token(
        db_host="h.postgres.database.azure.com",
        db_port="5432",
        db_user="mi-name",
        credential=cred,
    )
    assert token == "tok"


def test_missing_azure_identity_raises_helpful_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "azure.identity":
            raise ImportError("simulated: azure-identity not installed")
        return real_import(name, *args, **kwargs)

    # Reset the module-level credential cache so the lazy import is attempted.
    monkeypatch.setattr(azure_entra_db_token, "_cached_credential", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="azure-identity is required"):
        generate_azure_entra_db_token()


def test_get_default_credential_builds_once_and_caches(monkeypatch):
    """The real (non-injected) path lazily builds DefaultAzureCredential exactly
    once and reuses it (covers the credential cache + double-checked lock)."""
    azure_identity = pytest.importorskip("azure.identity")

    constructed = []

    class _FakeCred:
        def __init__(self):
            constructed.append(1)

    monkeypatch.setattr(azure_identity, "DefaultAzureCredential", _FakeCred)
    monkeypatch.setattr(azure_entra_db_token, "_cached_credential", None)

    first = azure_entra_db_token._get_default_credential()
    second = azure_entra_db_token._get_default_credential()

    assert first is second  # cached
    assert len(constructed) == 1  # built only once
