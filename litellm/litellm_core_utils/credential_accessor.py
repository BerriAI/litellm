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
        """Add credentials to the list, replacing any existing entry with the same name in place.

        Runs on every config reload with the full DB credential table, so it indexes the
        current list once instead of rescanning it per credential.
        """
        index_by_name: dict[str, int] = {cred.credential_name: i for i, cred in enumerate(litellm.credential_list)}

        for credential in credentials:
            existing_index = index_by_name.get(credential.credential_name)
            if existing_index is not None:
                litellm.credential_list[existing_index] = credential
            else:
                index_by_name[credential.credential_name] = len(litellm.credential_list)
                litellm.credential_list.append(credential)
