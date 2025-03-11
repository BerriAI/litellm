"""Utils for accessing credentials."""

import litellm


class CredentialAccessor:
    @staticmethod
    def get_credential_values(credential_name: str) -> dict:
        """Safe accessor for credentials."""
        if not litellm.credential_list:
            return {}
        for credential in litellm.credential_list:
            if credential.credential_name == credential_name:
                return credential.credential_values.copy()
        return {}
