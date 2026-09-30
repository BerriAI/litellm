"""Utils for accessing credentials."""

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
    def upsert_credentials(credentials: list[CredentialItem]):
        """Add credentials to the list, replacing the first existing entry with the same name in place."""
        index_by_name: Final[dict[str, int]] = {}
        for i, cred in enumerate(litellm.credential_list):
            index_by_name.setdefault(cred.credential_name, i)

        for credential in credentials:
            existing_index = index_by_name.get(credential.credential_name)
            if existing_index is not None:
                litellm.credential_list[existing_index] = credential
            else:
                litellm.credential_list.append(credential)
