import hashlib
import json

from litellm.types.router import CredentialLiteLLMParams


def get_litellm_params_sensitive_credential_hash(litellm_params: dict) -> str:
    """
    Hash of the credential params, used for mapping the file id to the right model
    """
    sensitive_params = CredentialLiteLLMParams(**litellm_params)
    return hashlib.sha256(
        json.dumps(sensitive_params.model_dump()).encode()
    ).hexdigest()
