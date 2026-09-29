"""Encryption at rest for the secret values in a managed vector store's ``litellm_params``."""

import os
from collections.abc import Callable, Iterator, Mapping
from datetime import timedelta
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from typing_extensions import TypeIs

from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy.common_utils.callback_utils import CALLBACK_VAR_ENCRYPTED_PREFIX
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.repositories.table_repositories import ManagedVectorStoresRepository
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

VECTOR_STORE_SECRET_PARAM_MASKER: Final = SensitiveDataMasker(extra_sensitive_patterns=frozenset(("connection",)))


def _is_json_object(value: object) -> TypeIs[dict[str, object]]:  # guard-ok: trivial isinstance; JSON keys are str
    return isinstance(value, dict)


def _is_json_array(value: object) -> TypeIs[list[object]]:  # guard-ok: trivial isinstance narrowing
    return isinstance(value, list)


def _map_litellm_param_strings(
    litellm_params: Mapping[str, object],
    transform: Callable[[str, str, bool], str],
    parent_is_secret: bool = False,
    depth: int = 0,
) -> dict[str, object]:
    """Apply ``transform(key, value, is_secret)`` to every string in a nested ``litellm_params`` dict.

    ``is_secret`` is true for a secret-named key and for every string nested under one. A list item is passed
    with the key of the list that holds it.
    """
    return {
        key: _map_param_value(
            key,
            value,
            transform,
            parent_is_secret or VECTOR_STORE_SECRET_PARAM_MASKER.is_sensitive_key(key),
            depth,
        )
        for key, value in litellm_params.items()
    }


def _map_param_value(
    key: str, value: object, transform: Callable[[str, str, bool], str], is_secret: bool, depth: int
) -> object:
    if isinstance(value, str):
        return transform(key, value, is_secret)
    if depth >= DEFAULT_MAX_RECURSE_DEPTH:
        return value
    if _is_json_object(value):
        return _map_litellm_param_strings(value, transform, is_secret, depth + 1)
    if _is_json_array(value):
        return [_map_param_value(key, item, transform, is_secret, depth + 1) for item in value]
    return value


def _secret_strings(
    litellm_params: Mapping[str, object], parent_is_secret: bool = False, depth: int = 0
) -> Iterator[tuple[str, str]]:
    """Every ``(key, value)`` string that ``_map_litellm_param_strings`` would pass with ``is_secret`` true."""
    return chain.from_iterable(
        _secret_strings_in(
            key, value, parent_is_secret or VECTOR_STORE_SECRET_PARAM_MASKER.is_sensitive_key(key), depth
        )
        for key, value in litellm_params.items()
    )


def _secret_strings_in(key: str, value: object, is_secret: bool, depth: int) -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        return iter(((key, value),) if is_secret else ())
    if depth >= DEFAULT_MAX_RECURSE_DEPTH:
        return iter(())
    if _is_json_object(value):
        return _secret_strings(value, is_secret, depth + 1)
    if _is_json_array(value):
        return chain.from_iterable(_secret_strings_in(key, item, is_secret, depth + 1) for item in value)
    return iter(())


def encrypt_vector_store_litellm_params(litellm_params: Mapping[str, object]) -> dict[str, object]:
    """Encrypt the secret values of a vector store's ``litellm_params`` for storage in the database."""

    def encrypt(key: str, value: str, is_secret: bool) -> str:
        if not is_secret or not value:
            return value
        try:
            return CALLBACK_VAR_ENCRYPTED_PREFIX + encrypt_value_helper(value)
        except Exception as e:  # noqa: BLE001  # no master key or salt key configured, so the value is stored as given
            verbose_proxy_logger.warning(
                "Vector store litellm_params[%r] stored unencrypted: %s", key, type(e).__name__
            )
            return value

    return _map_litellm_param_strings(litellm_params, encrypt)


def decrypt_vector_store_litellm_params(vector_store: LiteLLM_ManagedVectorStore) -> LiteLLM_ManagedVectorStore:
    """Return ``vector_store`` with the encrypted values of its ``litellm_params`` decrypted; plaintext is kept."""

    def decrypt(key: str, value: str, is_secret: bool) -> str:
        decrypted: Final = _decrypt_secret_value(key, value) if is_secret else None
        return value if decrypted is None else decrypted

    litellm_params: Final = vector_store.get("litellm_params")
    if not isinstance(litellm_params, dict):
        return vector_store
    return vector_store | LiteLLM_ManagedVectorStore(litellm_params=_map_litellm_param_strings(litellm_params, decrypt))


def holds_undecrypted_secret(vector_store: LiteLLM_ManagedVectorStore) -> bool:
    """Whether a decrypted ``vector_store`` still has an encrypted secret value, one this proxy's key cannot read."""
    litellm_params: Final = vector_store.get("litellm_params")
    return isinstance(litellm_params, dict) and any(
        value.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX) for _, value in _secret_strings(litellm_params)
    )


def _decrypt_secret_value(key: str, value: str) -> str | None:
    """The plaintext of an encrypted ``litellm_params`` value, or None when ``value`` is not one this proxy can read."""
    if not value.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX):
        return None
    return decrypt_value_helper(
        value=value.removeprefix(CALLBACK_VAR_ENCRYPTED_PREFIX),
        key=key,
        exception_type="debug",
        return_original_value=False,
    )


async def reencrypt_vector_store_litellm_params(prisma_client: "PrismaClient", new_master_key: str) -> int:
    """Re-encrypt every stored vector store's encrypted ``litellm_params`` values for a master-key rotation.

    Values are re-encrypted under ``LITELLM_SALT_KEY`` when it is set, else under ``new_master_key``. Plaintext
    values and values that do not decrypt are left as stored. All rows are rewritten in one transaction. Returns the
    number of rows rewritten.
    """
    salt_key: Final = os.getenv("LITELLM_SALT_KEY")
    new_encryption_key: Final = new_master_key if salt_key is None else salt_key

    def reencrypt(key: str, value: str, is_secret: bool) -> str:
        plaintext: Final = _decrypt_secret_value(key, value) if is_secret else None
        if plaintext is None:
            return value
        return CALLBACK_VAR_ENCRYPTED_PREFIX + encrypt_value_helper(plaintext, new_encryption_key=new_encryption_key)

    stored_rows: Final = tuple(
        LiteLLM_ManagedVectorStore(**dict(row))
        for row in await ManagedVectorStoresRepository(prisma_client).table.find_many()
    )
    for stored in stored_rows:
        _warn_about_undecryptable_values(stored)
    rewrites: Final = MappingProxyType(
        {
            vector_store_id: reencrypted
            for stored in stored_rows
            if isinstance(litellm_params := stored.get("litellm_params"), dict)
            and (vector_store_id := stored.get("vector_store_id")) is not None
            and (reencrypted := _map_litellm_param_strings(litellm_params, reencrypt)) != litellm_params
        }
    )
    if rewrites:
        async with prisma_client.tx(timeout=timedelta(minutes=2)) as tx:
            for vector_store_id, reencrypted in rewrites.items():
                await tx.litellm_managedvectorstorestable.update_many(
                    where={"vector_store_id": vector_store_id},
                    data={"litellm_params": safe_dumps(reencrypted)},
                )
    return len(rewrites)


def _warn_about_undecryptable_values(vector_store: LiteLLM_ManagedVectorStore) -> None:
    litellm_params: Final = vector_store.get("litellm_params")
    undecryptable_keys: Final = (
        sorted(
            key
            for key, value in _secret_strings(litellm_params)
            if value.startswith(CALLBACK_VAR_ENCRYPTED_PREFIX) and _decrypt_secret_value(key, value) is None
        )
        if isinstance(litellm_params, dict)
        else []
    )
    if undecryptable_keys:
        verbose_proxy_logger.warning(
            "Vector store %s: litellm_params %s do not decrypt with the current key and were not re-encrypted",
            vector_store.get("vector_store_id"),
            undecryptable_keys,
        )
