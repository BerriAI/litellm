import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm._logging import verbose_proxy_logger
from litellm.proxy import proxy_server
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with
from litellm.proxy.vector_store_endpoints.litellm_params_encryption import (
    decrypt_vector_store_litellm_params,
    encrypt_vector_store_litellm_params,
    holds_undecrypted_secret,
    reencrypt_vector_store_litellm_params,
)
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry

_SALT_KEY = "sk-vector-store-test-salt"
_ENCRYPTED_PREFIX = "litellm_enc::"


@pytest.fixture
def salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", _SALT_KEY)
    monkeypatch.setattr(proxy_server, "general_settings", {})


def _decrypted_under(value: str, key: str):
    assert value.startswith(_ENCRYPTED_PREFIX)
    return decrypt_if_encrypted_with(value.removeprefix(_ENCRYPTED_PREFIX), key)


def test_encrypt_litellm_params_encrypts_only_secret_values(salt_key):
    params = {
        "api_key": "sk-vs-secret",
        "api_base": "https://vector.example/v1",
        "aws_secret_access_key": "aws-secret",
        "aws_region_name": "us-east-1",
        "valkey_password": "valkey-secret",
        "max_retries": 2,
        "litellm_embedding_config": {"api_key": "sk-embed-secret", "api_base": "https://embed.example"},
        "vertex_credentials": {"client_email": "svc@example.iam", "private_key": "pem-secret"},
    }

    encrypted = encrypt_vector_store_litellm_params(params)

    for key in ("api_key", "aws_secret_access_key", "valkey_password"):
        assert _decrypted_under(encrypted[key], _SALT_KEY) == params[key]
    assert _decrypted_under(encrypted["litellm_embedding_config"]["api_key"], _SALT_KEY) == "sk-embed-secret"
    assert _decrypted_under(encrypted["vertex_credentials"]["client_email"], _SALT_KEY) == "svc@example.iam"
    assert _decrypted_under(encrypted["vertex_credentials"]["private_key"], _SALT_KEY) == "pem-secret"
    assert encrypted["api_base"] == "https://vector.example/v1"
    assert encrypted["aws_region_name"] == "us-east-1"
    assert encrypted["max_retries"] == 2
    assert encrypted["litellm_embedding_config"]["api_base"] == "https://embed.example"
    for secret in ("sk-vs-secret", "aws-secret", "valkey-secret", "sk-embed-secret", "pem-secret", "svc@example"):
        assert secret not in json.dumps(encrypted)


def test_ciphertext_supplied_by_a_caller_is_never_decrypted_back(salt_key):
    ciphertext = encrypt_vector_store_litellm_params({"api_key": "someone-elses-secret"})["api_key"]
    supplied = {"api_key": ciphertext, "api_base": ciphertext, "nested": {"url": ciphertext}}

    stored = encrypt_vector_store_litellm_params(supplied)
    read_back = decrypt_vector_store_litellm_params(
        LiteLLM_ManagedVectorStore(vector_store_id="vs", litellm_params=stored)
    )

    assert read_back["litellm_params"] == supplied
    assert "someone-elses-secret" not in json.dumps(read_back)


def test_encrypt_litellm_params_without_a_master_or_salt_key_stores_values_as_given(monkeypatch):
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    monkeypatch.setattr(proxy_server, "master_key", None)

    with patch.object(verbose_proxy_logger, "warning") as warning:
        assert encrypt_vector_store_litellm_params({"api_key": "sk-vs-secret"}) == {"api_key": "sk-vs-secret"}

    warning.assert_called_once()
    assert "sk-vs-secret" not in str(warning.call_args)


def test_decrypt_litellm_params_round_trips_and_keeps_legacy_plaintext(salt_key):
    params = {
        "api_key": "sk-vs-secret",
        "api_base": "https://vector.example/v1",
        "litellm_embedding_config": {"api_key": "sk-embed-secret"},
        "vertex_credentials": {"client_email": "svc@example.iam", "private_key": "pem-secret"},
    }
    encrypted_store = LiteLLM_ManagedVectorStore(
        vector_store_id="vs_new",
        custom_llm_provider="openai",
        litellm_params=encrypt_vector_store_litellm_params(params),
    )
    legacy_store = LiteLLM_ManagedVectorStore(
        vector_store_id="vs_legacy", custom_llm_provider="openai", litellm_params={"api_key": "sk-legacy-plaintext"}
    )

    assert decrypt_vector_store_litellm_params(encrypted_store)["litellm_params"] == params
    assert decrypt_vector_store_litellm_params(legacy_store) == legacy_store
    assert decrypt_vector_store_litellm_params(LiteLLM_ManagedVectorStore(vector_store_id="vs_none")) == {
        "vector_store_id": "vs_none"
    }


def test_decrypt_litellm_params_keeps_a_value_it_cannot_decrypt(salt_key, monkeypatch):
    encrypted = encrypt_vector_store_litellm_params({"api_key": "sk-vs-secret"})
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-some-other-salt")

    store = LiteLLM_ManagedVectorStore(vector_store_id="vs", litellm_params=encrypted)
    assert decrypt_vector_store_litellm_params(store)["litellm_params"] == encrypted


@pytest.mark.asyncio
async def test_vector_stores_loaded_from_db_have_decrypted_litellm_params(salt_key):
    new_row = {
        "vector_store_id": "vs_new",
        "custom_llm_provider": "openai",
        "litellm_params": encrypt_vector_store_litellm_params({"api_key": "sk-new-secret", "api_base": "https://b"}),
    }
    legacy_row = {
        "vector_store_id": "vs_legacy",
        "custom_llm_provider": "openai",
        "litellm_params": {"api_key": "sk-legacy-plaintext"},
    }
    prisma_client = MagicMock()
    prisma_client.db.litellm_managedvectorstorestable.find_many = AsyncMock(return_value=[new_row, legacy_row])

    loaded = await VectorStoreRegistry._get_vector_stores_from_db(prisma_client=prisma_client)

    assert [vs["litellm_params"] for vs in loaded] == [
        {"api_key": "sk-new-secret", "api_base": "https://b"},
        {"api_key": "sk-legacy-plaintext"},
    ]


def _rotation_rows(encrypted_params):
    rows = [
        {"vector_store_id": "vs_new", "litellm_params": encrypted_params},
        {"vector_store_id": "vs_legacy", "litellm_params": {"api_key": "sk-legacy-plaintext"}},
    ]
    prisma_client = MagicMock()
    prisma_client.db.litellm_managedvectorstorestable.find_many = AsyncMock(return_value=rows)
    prisma_client.db.litellm_managedvectorstorestable.update = AsyncMock()
    return prisma_client


@pytest.mark.asyncio
async def test_reencrypt_moves_encrypted_rows_to_the_new_master_key_and_leaves_plaintext_rows(monkeypatch):
    old_key, new_key = "sk-current-master-key", "sk-rotated-master-key"
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "master_key", old_key)
    prisma_client = _rotation_rows(
        encrypt_vector_store_litellm_params({"api_key": "sk-new-secret", "api_base": "https://b"})
    )

    rewritten = await reencrypt_vector_store_litellm_params(prisma_client=prisma_client, new_master_key=new_key)

    assert rewritten == 1
    prisma_client.db.litellm_managedvectorstorestable.update.assert_awaited_once()
    call = prisma_client.db.litellm_managedvectorstorestable.update.await_args
    assert call.kwargs["where"] == {"vector_store_id": "vs_new"}
    stored = json.loads(call.kwargs["data"]["litellm_params"])
    assert _decrypted_under(stored["api_key"], new_key) == "sk-new-secret"
    assert _decrypted_under(stored["api_key"], old_key) is None
    assert stored["api_base"] == "https://b"


@pytest.mark.asyncio
async def test_reencrypt_keeps_values_under_the_salt_key_when_one_is_set(salt_key):
    prisma_client = _rotation_rows(encrypt_vector_store_litellm_params({"api_key": "sk-new-secret"}))

    await reencrypt_vector_store_litellm_params(prisma_client=prisma_client, new_master_key="sk-rotated-master-key")

    stored = json.loads(
        prisma_client.db.litellm_managedvectorstorestable.update.await_args.kwargs["data"]["litellm_params"]
    )
    assert _decrypted_under(stored["api_key"], _SALT_KEY) == "sk-new-secret"


@pytest.mark.asyncio
async def test_reencrypt_warns_about_values_it_cannot_decrypt(salt_key, monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-removed-salt")
    unreadable = encrypt_vector_store_litellm_params({"api_key": "sk-lost-secret"})
    monkeypatch.setenv("LITELLM_SALT_KEY", _SALT_KEY)
    prisma_client = _rotation_rows(unreadable)

    with patch.object(verbose_proxy_logger, "warning") as warning:
        rewritten = await reencrypt_vector_store_litellm_params(prisma_client=prisma_client, new_master_key="sk-new")

    assert rewritten == 0
    prisma_client.db.litellm_managedvectorstorestable.update.assert_not_awaited()
    warning.assert_called_once()
    assert "vs_new" in warning.call_args.args
    assert "sk-lost-secret" not in str(warning.call_args)


@pytest.mark.asyncio
async def test_reencrypt_treats_an_empty_salt_key_as_set(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "")
    monkeypatch.setattr(proxy_server, "general_settings", {})
    prisma_client = _rotation_rows(encrypt_vector_store_litellm_params({"api_key": "sk-new-secret"}))

    await reencrypt_vector_store_litellm_params(prisma_client=prisma_client, new_master_key="sk-rotated-master-key")

    stored = json.loads(
        prisma_client.db.litellm_managedvectorstorestable.update.await_args.kwargs["data"]["litellm_params"]
    )
    assert decrypt_vector_store_litellm_params(LiteLLM_ManagedVectorStore(vector_store_id="vs", litellm_params=stored))[
        "litellm_params"
    ] == {"api_key": "sk-new-secret"}


def test_holds_undecrypted_secret(salt_key, monkeypatch):
    readable = encrypt_vector_store_litellm_params({"api_key": "sk-a", "nested": {"api_key": "sk-b"}})
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-other-salt")
    unreadable_nested = {"api_key": "sk-plain", "nested": encrypt_vector_store_litellm_params({"api_key": "sk-b"})}
    monkeypatch.setenv("LITELLM_SALT_KEY", _SALT_KEY)

    def holds(params):
        return holds_undecrypted_secret(
            decrypt_vector_store_litellm_params(LiteLLM_ManagedVectorStore(vector_store_id="vs", litellm_params=params))
        )

    assert holds(readable) is False
    assert holds({"api_key": "sk-plain", "api_base": "litellm_enc::not-a-secret-key"}) is False
    assert holds(unreadable_nested) is True
