from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.proxy.common_utils.credential_hydration import (
    effective_server_owned_wif_fields,
    hydrate_named_credential_authoritative,
    named_credential_wif_fields,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.types.utils import CredentialItem


@pytest.mark.asyncio
async def test_authoritative_hydrate_returns_an_encrypted_empty_value_as_empty(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-hydration-test-salt")
    row = {
        "credential_name": "openai-wif",
        "credential_values": {
            "api_base": encrypt_value_helper(""),
            "openai_service_account_id": encrypt_value_helper("user-1"),
        },
        "credential_info": {"custom_llm_provider": "openai"},
    }
    prisma = MagicMock()
    prisma.db.litellm_credentialstable.find_unique = AsyncMock(return_value=row)

    with patch.object(litellm, "credential_list", []):  # test-quality-ok: the row under test must win over memory
        resolved = await hydrate_named_credential_authoritative("openai-wif", prisma)

    assert resolved is not None
    assert resolved.credential_values == {"api_base": "", "openai_service_account_id": "user-1"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credential_api_key, expected",
    [
        ("oidc/file//var/run/secrets/kubernetes.io/serviceaccount/token", ("api_key",)),
        ("sk-static", ()),
    ],
    ids=["oidc_reference", "static_key"],
)
async def test_attaching_a_credential_is_server_owned_only_when_its_api_key_is_an_oidc_reference(
    credential_api_key: str, expected: tuple[str, ...]
):
    credential: Final = CredentialItem(
        credential_name="admin-oidc", credential_values={"api_key": credential_api_key}, credential_info={}
    )

    with patch.object(litellm, "credential_list", [credential]):  # test-quality-ok: the in-memory list is the input
        fields = await effective_server_owned_wif_fields(
            {"model": "mistral/devstral-latest"}, {"litellm_credential_name": "admin-oidc"}, None
        )

    assert fields == expected


@pytest.mark.asyncio
async def test_a_stored_credential_whose_encrypted_api_key_is_an_oidc_reference_is_server_owned(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-hydration-test-salt")
    row = {
        "credential_name": "admin-oidc",
        "credential_values": {
            "api_key": encrypt_value_helper("oidc/file//var/run/secrets/kubernetes.io/serviceaccount/token")
        },
        "credential_info": {"custom_llm_provider": "mistral"},
    }
    prisma = MagicMock()
    prisma.db.litellm_credentialstable.find_unique = AsyncMock(return_value=row)

    with patch.object(litellm, "credential_list", []):  # test-quality-ok: a pod that has not loaded the row yet
        fields = await named_credential_wif_fields("admin-oidc", prisma)

    assert fields == ("api_key",)
