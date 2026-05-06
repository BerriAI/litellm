"""
Thin encryption wrapper for the Cloud Agents settings endpoints.

We do NOT roll a new KMS path here — we reuse the existing virtual-key
nacl/SecretBox helpers (`encrypt_value_helper` / `decrypt_value_helper`) so
operators only need to manage one `LITELLM_SALT_KEY`. The wrapper exists only
to give the agent endpoints a single import surface and to centralize the
"None passes through" semantics so callers don't sprinkle `if value is None`
guards everywhere.

Used for:
* AWS BYOC creds on `LiteLLM_AgentVMConfig` (access key, secret key, role ARN)
* Per-team secret values on `LiteLLM_AgentSecret`
"""

from typing import Optional

from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)


def encrypt_optional(value: Optional[str]) -> Optional[str]:
    """Encrypt a value, passing None through.

    Returns base64(urlsafe)-encoded ciphertext, or None if the input was None
    or empty. Empty strings are normalized to None so `aws_role_arn=""` from
    the wire round-trips cleanly to NULL in the DB.
    """
    if value is None or value == "":
        return None
    return encrypt_value_helper(value)


def decrypt_optional(value: Optional[str], *, key: str) -> Optional[str]:
    """Decrypt a previously-encrypted value, passing None through.

    `key` is a debug-label only (used by the underlying helper to surface
    which field failed to decrypt) — it is NOT a signing key.
    """
    if value is None or value == "":
        return None
    return decrypt_value_helper(value=value, key=key)
