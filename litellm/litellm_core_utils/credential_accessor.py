"""Utils for accessing credentials."""

from collections.abc import Mapping
from typing import Final

import litellm
from litellm.types.router import holds_api_key_secret_reference
from litellm.types.utils import (
    API_KEY_SECRET_REFERENCE_REJECTION,
    CredentialItem,
    credential_api_key_secret_reference,
)


class CredentialAccessor:
    @staticmethod
    def find_credential(credential_name: str) -> CredentialItem | None:
        return next(
            (credential for credential in litellm.credential_list if credential.credential_name == credential_name),
            None,
        )

    @staticmethod
    def get_credential_values(
        credential_name: str,
    ) -> dict[str, object]:  # mutable-ok: callers merge the returned credential map
        """Safe accessor for credentials."""

        credential: Final = CredentialAccessor.find_credential(credential_name)
        return {} if credential is None else dict(credential_api_key_secret_reference(credential.credential_values))

    @staticmethod
    def upsert_credentials(credentials: list[CredentialItem]):
        """Add a credential to the list of credentials."""

        credential_names: Final = [cred.credential_name for cred in litellm.credential_list]

        for credential in credentials:
            if credential.credential_name in credential_names:
                # Find and replace the existing credential in the list
                for i, existing_cred in enumerate(litellm.credential_list):
                    if existing_cred.credential_name == credential.credential_name:
                        litellm.credential_list[i] = credential
                        break
            else:
                litellm.credential_list.append(credential)


def sends_api_key_secret_reference(params: Mapping[str, object]) -> bool:
    named: Final = params.get("litellm_credential_name")
    credential_values: Final = CredentialAccessor.get_credential_values(named) if isinstance(named, str) else {}
    return holds_api_key_secret_reference({**credential_values, **params})


def reject_api_key_secret_reference(params: Mapping[str, object]) -> None:
    if holds_api_key_secret_reference(params):
        raise ValueError(API_KEY_SECRET_REFERENCE_REJECTION)
    named: Final = params.get("litellm_credential_name")
    if isinstance(named, str) and holds_api_key_secret_reference(CredentialAccessor.get_credential_values(named)):
        raise ValueError(
            f"Rejected Request: litellm_credential_name={named!r} names a credential holding an oidc/ api_key "
            "secret reference, which only a deployment a proxy admin configures can attach."
        )
