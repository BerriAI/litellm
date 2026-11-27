"""DB helpers for the experimental secrets vault.

A row in ``LiteLLM_VaultSecrets`` is one named secret owned by one LiteLLM
user. ``value_b64`` always stores the value encrypted with the proxy salt key,
and plaintext only exists in the value returned to the owner.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import PrismaTableRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient


class _VaultSecretsRepository(PrismaTableRepository["prisma_models.LiteLLM_VaultSecrets"]):
    table_name = "litellm_vaultsecrets"


def _table(prisma_client: "PrismaClient") -> TableActions["prisma_models.LiteLLM_VaultSecrets"]:
    return _VaultSecretsRepository(prisma_client, use_writer=True).table


def _owned(user_id: str, secret_name: str) -> Mapping[str, Mapping[str, str]]:
    return {"user_id_secret_name": {"user_id": user_id, "secret_name": secret_name}}


def decrypt_vault_secret(stored: str) -> str | None:
    return decrypt_value_helper(
        value=stored,
        key="vault_secret",
        exception_type="debug",
        return_original_value=False,
    )


async def upsert_vault_secret(
    prisma_client: "PrismaClient",
    user_id: str,
    secret_name: str,
    value: str,
    description: str | None,
) -> "prisma_models.LiteLLM_VaultSecrets":
    encrypted: Final = encrypt_value_helper(value)
    return await _table(prisma_client).upsert(
        where=_owned(user_id, secret_name),
        data={
            "create": {
                "user_id": user_id,
                "secret_name": secret_name,
                "description": description,
                "value_b64": encrypted,
            },
            "update": {"description": description, "value_b64": encrypted},
        },
    )


async def list_vault_secrets(
    prisma_client: "PrismaClient",
    user_id: str,
) -> Sequence["prisma_models.LiteLLM_VaultSecrets"]:
    return await _table(prisma_client).find_many(where={"user_id": user_id}, order={"secret_name": "asc"})


async def get_vault_secret(
    prisma_client: "PrismaClient",
    user_id: str,
    secret_name: str,
) -> "prisma_models.LiteLLM_VaultSecrets | None":
    return await _table(prisma_client).find_unique(where=_owned(user_id, secret_name))


async def mark_vault_secret_accessed(
    prisma_client: "PrismaClient",
    user_id: str,
    secret_name: str,
) -> datetime:
    accessed_at: Final = datetime.now(timezone.utc)
    await _table(prisma_client).update_many(
        where={"user_id": user_id, "secret_name": secret_name},
        data={"last_accessed_at": accessed_at},
    )
    return accessed_at


async def delete_vault_secret(
    prisma_client: "PrismaClient",
    user_id: str,
    secret_name: str,
) -> bool:
    deleted: Final = await _table(prisma_client).delete_many(where={"user_id": user_id, "secret_name": secret_name})
    return deleted > 0
