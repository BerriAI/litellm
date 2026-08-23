"""Shared helper for resolving a named Credential's values server-side.

Memory first (``litellm.credential_list``, already decrypted -- matching
``CredentialAccessor.get_credential_values``), then a DB decrypt fallback for a pod whose
in-memory list has not yet picked up a credential another pod just wrote or updated.
"""

from types import MappingProxyType
from typing import Final

import litellm
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.proxy.utils import PrismaClient
from litellm.repositories.credentials_repository import CredentialsRepository
from litellm.types.utils import CredentialItem


async def hydrate_named_credential(
    credential_name: str,
    prisma_client: PrismaClient | None,
) -> CredentialItem | None:
    for credential in litellm.credential_list:
        if credential.credential_name == credential_name:
            return credential
    if prisma_client is None:
        return None
    db_credential: Final = await CredentialsRepository(prisma_client).find_by_name(credential_name)
    if db_credential is None:
        return None
    decrypted_values: Final = MappingProxyType(
        {
            key: decrypt_value_helper(value=value, key=key) or value
            for key, value in db_credential.credential_values.items()
        }
    )
    return CredentialItem(
        credential_name=db_credential.credential_name,
        credential_values=decrypted_values,
        credential_info=db_credential.credential_info,
    )
