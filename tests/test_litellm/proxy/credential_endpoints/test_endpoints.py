"""Tests for the credential management endpoints."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.credential_endpoints.endpoints import update_db_credential
from litellm.proxy.proxy_server import app
from litellm.types.utils import CredentialItem

client = TestClient(app)


def _as_admin():
    return UserAPIKeyAuth(api_key="test-key", user_role="proxy_admin")


def test_update_db_credential_replaces_credential_info_wholesale():
    """The stored row has always been replaced, not merged: the old guard asked whether
    the stored info contained a key literally named ``credential_info``, which nothing
    stores, so it emptied the dict and repopulated it from the patch every time. This
    pins that behaviour so the simplification cannot drift into a real merge."""
    stored = CredentialItem(
        credential_name="c1",
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai", "description": "keep me"},
    )
    patch_item = CredentialItem(
        credential_name="c1",
        credential_values={},
        credential_info={"description": "patched"},
    )

    merged = update_db_credential(stored, patch_item)

    assert merged.credential_info == {"description": "patched"}
    assert "custom_llm_provider" not in merged.credential_info


def test_update_db_credential_keeps_credential_info_when_the_patch_carries_none():
    """An empty ``credential_info`` on the patch must leave the stored one alone; the
    master-key rotation path relies on the untouched branch."""
    stored = CredentialItem(
        credential_name="c1",
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai"},
    )
    patch_item = CredentialItem(credential_name="c1", credential_values={"api_key": "sk-new"}, credential_info={})

    with patch("litellm.proxy.proxy_server.master_key", "sk-test-master"):
        merged = update_db_credential(stored, patch_item)

    assert merged.credential_info == {"custom_llm_provider": "openai"}


def test_update_db_credential_is_unchanged_for_a_full_patch():
    """Master-key rotation passes the whole decrypted row as the patch
    (``key_management_endpoints.py`` rotate loop), so replace and merge agree there."""
    info = {"custom_llm_provider": "openai", "description": "prod"}
    stored = CredentialItem(credential_name="c1", credential_values={"api_key": "sk-old"}, credential_info=dict(info))
    full_patch = CredentialItem(
        credential_name="c1", credential_values={"api_key": "sk-old"}, credential_info=dict(info)
    )

    with patch("litellm.proxy.proxy_server.master_key", "sk-test-master"):
        merged = update_db_credential(stored, full_patch)

    assert merged.credential_info == info


def test_partial_patch_leaves_no_stale_fields_in_the_in_memory_credential():
    """Regression: the in-memory copy merged ``credential_info`` while the DB row replaced
    it, so after a partial patch ``litellm.credential_list`` still advertised fields the
    row no longer had. The dashboard refetches from that list, so it showed values that
    were already gone, until the next config reload (30s) silently dropped them."""
    stored = CredentialItem(
        credential_name="c1",
        credential_values={"api_key": "sk-old"},
        credential_info={"custom_llm_provider": "openai", "description": "keep me"},
    )
    original_list = litellm.credential_list
    litellm.credential_list = [
        CredentialItem(
            credential_name="c1",
            credential_values={"api_key": "sk-old"},
            credential_info={"custom_llm_provider": "openai", "description": "keep me"},
        )
    ]
    app.dependency_overrides[user_api_key_auth] = _as_admin
    try:
        with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()), patch(
            "litellm.proxy.proxy_server.master_key", "sk-test-master"
        ), patch("litellm.proxy.credential_endpoints.endpoints.CredentialsRepository") as repository:
            repository.return_value.find_by_name = AsyncMock(return_value=stored)
            repository.return_value.update_by_name = AsyncMock(return_value=None)

            response = client.patch(
                "/credentials/c1",
                json={
                    "credential_name": "c1",
                    "credential_values": {"api_key": "sk-new"},
                    "credential_info": {"description": "patched"},
                },
                headers={"Authorization": "Bearer test-key"},
            )

        assert response.status_code == 200, response.text
        in_memory = next(c for c in litellm.credential_list if c.credential_name == "c1")
        assert in_memory.credential_info == {"description": "patched"}
        assert "custom_llm_provider" not in in_memory.credential_info
    finally:
        app.dependency_overrides.clear()
        litellm.credential_list = original_list
