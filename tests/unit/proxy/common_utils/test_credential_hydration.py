from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.proxy.common_utils.credential_hydration import (
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
async def test_wif_gate_sees_the_padded_credential_a_trimmed_name_resolves_to():
    """Resolution maps ``corp-wif`` onto a credential stored as ``corp-wif ``, so the federation
    gate must see that credential's server-owned fields too, or a team admin could swap them out."""
    padded = CredentialItem(
        credential_name="corp-wif ",
        credential_values={"anthropic_keycloak_token_url": "https://keycloak.internal/token"},
        credential_info={},
        source="config",
    )
    with patch.object(
        litellm, "credential_list", [padded]
    ):  # test-quality-ok: the in-memory config credential under test
        resolved = CredentialAccessor.find_credential("corp-wif")
        gated = await named_credential_wif_fields("corp-wif", None)

    assert resolved is padded
    assert "anthropic_keycloak_token_url" in gated
