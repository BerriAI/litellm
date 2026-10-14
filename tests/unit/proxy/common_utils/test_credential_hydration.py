from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.proxy.common_utils.credential_hydration import hydrate_named_credential_authoritative
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper


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
