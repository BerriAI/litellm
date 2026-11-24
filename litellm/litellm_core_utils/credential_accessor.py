"""Utils for accessing credentials."""

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
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
        if credential is None:
            return {}
        return deepcopy(
            {
                key: value
                for key, value in credential_api_key_secret_reference(credential.credential_values).items()
                if key != "github_copilot_app_private_key"
            }
        )

    @staticmethod
    def upsert_credentials(credentials: list[CredentialItem]) -> None:
        """Replace the first loaded credential with each incoming name, append names not loaded at call start."""

        first_index_by_name: Final = MappingProxyType(
            {cred.credential_name: i for i, cred in reversed(tuple(enumerate(litellm.credential_list)))}
        )

        for credential in credentials:
            match first_index_by_name.get(credential.credential_name):
                case None:
                    litellm.credential_list.append(credential)
                case existing_index:
                    litellm.credential_list[existing_index] = credential


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
