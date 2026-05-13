"""
Tests for cascading credential renames into model rows.

When a credential is renamed via PATCH /credentials/{old_name}, every model
row whose `litellm_params.litellm_credential_name` references the old name
must be updated in lockstep — otherwise those models will fail at request
time when the router tries to resolve a credential that no longer exists.
"""

import json
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.proxy_server as ps
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.credential_endpoints.endpoints import (
    _cascade_rename_credential_in_models,
)


@pytest.fixture
def salt_key(monkeypatch):
    """Encrypt/decrypt helpers require a signing key — set one for the test."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-cascade-rename")
    yield


def _model_row(model_id: str, credential_name_plain: str | None):
    """
    Build a fake LiteLLM_ProxyModelTable row whose `litellm_params` mirrors the
    on-disk shape (encrypted values, dict-typed JSON column).
    """
    params: dict = {"model": "gpt-4o"}
    if credential_name_plain is not None:
        params["litellm_credential_name"] = encrypt_value_helper(
            value=credential_name_plain
        )
    row = MagicMock()
    row.model_id = model_id
    row.litellm_params = params
    return row


@pytest.mark.asyncio
async def test_cascade_renames_only_matching_models(salt_key, monkeypatch):
    """Only models referencing the old name are updated; others are left alone."""
    monkeypatch.setattr(ps, "llm_router", None)

    matching = _model_row("model-1", "old-cred")
    other = _model_row("model-2", "different-cred")
    no_credential = _model_row("model-3", None)

    tx = types.SimpleNamespace(
        litellm_proxymodeltable=types.SimpleNamespace(
            find_many=AsyncMock(return_value=[matching, other, no_credential]),
            update=AsyncMock(),
        )
    )

    updated = await _cascade_rename_credential_in_models(
        tx=tx,
        old_credential_name="old-cred",
        new_credential_name="new-cred",
    )

    assert updated == 1
    tx.litellm_proxymodeltable.update.assert_awaited_once()
    call = tx.litellm_proxymodeltable.update.await_args
    assert call.kwargs["where"] == {"model_id": "model-1"}

    # Prisma's JSON column expects a serialized string, not a dict.
    raw_params = call.kwargs["data"]["litellm_params"]
    assert isinstance(raw_params, str)
    written_params = json.loads(raw_params)
    assert written_params["model"] == "gpt-4o"
    # The new credential name is stored encrypted, not plain text.
    assert written_params["litellm_credential_name"] != "new-cred"
    # And it must round-trip back to the new name.
    from litellm.proxy.common_utils.encrypt_decrypt_utils import (
        decrypt_value_helper,
    )

    assert (
        decrypt_value_helper(
            value=written_params["litellm_credential_name"],
            key="litellm_credential_name",
            return_original_value=True,
        )
        == "new-cred"
    )


@pytest.mark.asyncio
async def test_cascade_updates_in_memory_router(salt_key, monkeypatch):
    """
    The router's in-memory model_list holds decrypted credential names. The
    cascade must rewrite them so live traffic doesn't try to resolve a name
    that no longer exists in litellm.credential_list.
    """
    fake_router = MagicMock()
    fake_router.model_list = [
        {"litellm_params": {"litellm_credential_name": "old-cred", "model": "gpt-4o"}},
        {
            "litellm_params": {
                "litellm_credential_name": "other-cred",
                "model": "claude",
            }
        },
        {"litellm_params": {"model": "no-creds"}},
    ]
    monkeypatch.setattr(ps, "llm_router", fake_router)

    matching = _model_row("model-1", "old-cred")
    tx = types.SimpleNamespace(
        litellm_proxymodeltable=types.SimpleNamespace(
            find_many=AsyncMock(return_value=[matching]),
            update=AsyncMock(),
        )
    )

    await _cascade_rename_credential_in_models(
        tx=tx,
        old_credential_name="old-cred",
        new_credential_name="new-cred",
    )

    assert (
        fake_router.model_list[0]["litellm_params"]["litellm_credential_name"]
        == "new-cred"
    )
    assert (
        fake_router.model_list[1]["litellm_params"]["litellm_credential_name"]
        == "other-cred"
    )
    assert "litellm_credential_name" not in fake_router.model_list[2]["litellm_params"]


@pytest.mark.asyncio
async def test_cascade_noop_when_no_models_match(salt_key, monkeypatch):
    """No matching rows → no updates issued, no in-memory mutation."""
    fake_router = MagicMock()
    untouched = {
        "litellm_params": {"litellm_credential_name": "other-cred", "model": "gpt-4o"}
    }
    fake_router.model_list = [untouched]
    monkeypatch.setattr(ps, "llm_router", fake_router)

    tx = types.SimpleNamespace(
        litellm_proxymodeltable=types.SimpleNamespace(
            find_many=AsyncMock(return_value=[_model_row("model-2", "other-cred")]),
            update=AsyncMock(),
        )
    )

    updated = await _cascade_rename_credential_in_models(
        tx=tx,
        old_credential_name="old-cred",
        new_credential_name="new-cred",
    )

    assert updated == 0
    tx.litellm_proxymodeltable.update.assert_not_awaited()
    assert untouched["litellm_params"]["litellm_credential_name"] == "other-cred"
