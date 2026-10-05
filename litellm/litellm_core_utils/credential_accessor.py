"""Utils for accessing credentials."""

from collections.abc import Sequence
from types import MappingProxyType
from typing import Final

import litellm
from litellm.types.utils import CredentialItem


class CredentialAccessor:
    @staticmethod
    def find_credential(credential_name: str) -> CredentialItem | None:
        return next(
            (credential for credential in litellm.credential_list if credential.credential_name == credential_name),
            None,
        )

    @staticmethod
    def get_credential_values(credential_name: str) -> dict:
        """Safe accessor for credentials."""

        credential: Final = CredentialAccessor.find_credential(credential_name)
        return {} if credential is None else credential.credential_values.copy()

    @staticmethod
    def upsert_credentials(credentials: Sequence[CredentialItem]) -> None:
        """Add credentials to the list, replacing the first existing entry with the same name in place."""
        first_index_by_name: Final = MappingProxyType(
            {cred.credential_name: i for i, cred in reversed(tuple(enumerate(litellm.credential_list)))}
        )

        for credential in credentials:
            if credential.credential_name in first_index_by_name:
                litellm.credential_list[first_index_by_name[credential.credential_name]] = credential
            else:
                litellm.credential_list.append(credential)
