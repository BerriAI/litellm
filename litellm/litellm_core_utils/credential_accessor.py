"""Utils for accessing credentials."""

from collections.abc import Mapping
from typing import Final

import litellm
from litellm.types.router import server_owned_wif_fields_present
from litellm.types.utils import CredentialItem


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
        return {} if credential is None else credential.credential_values.copy()

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


def reject_federated_credential_reference(body: Mapping[str, object]) -> None:
    """Raise ``ValueError`` if a request body picks a server-owned identity by credential name.

    ``load_credentials_from_list`` merges a named credential's values into the call, so a body
    naming a federated credential moves the token exchange onto that credential's federation rule
    and organization exactly as sending the federation fields inline would, which
    ``reject_server_owned_wif_params`` already refuses, and a body naming one whose ``api_key`` is
    an ``oidc/`` secret reference gets the proxy to read that secret for a call it shapes. Attaching
    either is a deployment decision, so it is refused with no client-side opt-in to relax it, the
    same as the inline form.

    Only credentials already loaded into memory can be resolved here, which is every credential the
    proxy would resolve for the call itself: ``load_credentials_from_list`` reads the same list.
    ``route_manages_deployments`` names the routes this is not applied to, where choosing the
    credential a deployment federates through is the point of the call.
    """
    named: Final = body.get("litellm_credential_name")
    if not isinstance(named, str) or not named:
        return
    wif_fields: Final = server_owned_wif_fields_present(CredentialAccessor.get_credential_values(named))
    if wif_fields:
        raise ValueError(
            f"Rejected Request: litellm_credential_name={named!r} names a credential configured for "
            f"workload identity federation or OAuth token exchange, or holding an oidc/ api_key secret "
            f"reference ({wif_fields[0]}), which a request body cannot choose. "
            "A proxy admin attaches it to a deployment."
        )
