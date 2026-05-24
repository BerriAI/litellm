import base64
import os
from typing import Literal, Optional

from litellm._logging import verbose_proxy_logger

# Prefix used to identify AES-256-GCM encrypted values (FIPS-compliant).
# Values without this prefix are treated as legacy nacl-encrypted.
_AES_GCM_PREFIX = "aes256gcm:"


def _is_fips_mode() -> bool:
    """Return True when LITELLM_FIPS_MODE is set to a truthy value."""
    return os.getenv("LITELLM_FIPS_MODE", "").lower() in ("true", "1", "yes")


def _get_salt_key():
    from litellm.proxy.proxy_server import master_key

    salt_key = os.getenv("LITELLM_SALT_KEY", None)

    if salt_key is None:
        salt_key = master_key

    return salt_key


def _encrypt_aes_gcm(value: str, signing_key: str) -> bytes:
    """Encrypt *value* with AES-256-GCM using a SHA-256 derived key.

    Returns: nonce (12 B) + ciphertext + auth-tag (16 B).
    Uses the ``cryptography`` library which delegates to OpenSSL and is
    FIPS-compliant when the underlying OpenSSL build is FIPS-enabled.
    """
    import hashlib

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = hashlib.sha256(signing_key.encode()).digest()
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, value.encode("utf-8"), None)
    return nonce + ciphertext


def _decrypt_aes_gcm(data: bytes, signing_key: str) -> str:
    """Decrypt AES-256-GCM ciphertext produced by :func:`_encrypt_aes_gcm`."""
    import hashlib

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if len(data) < 12:
        raise ValueError("AES-GCM ciphertext too short")

    key = hashlib.sha256(signing_key.encode()).digest()
    nonce, ciphertext = data[:12], data[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")


def encrypt_value_helper(value: str, new_encryption_key: Optional[str] = None):
    signing_key = new_encryption_key or _get_salt_key()

    try:
        if isinstance(value, str):
            if _is_fips_mode():
                encrypted_bytes = _encrypt_aes_gcm(value, signing_key)  # type: ignore
                return _AES_GCM_PREFIX + base64.urlsafe_b64encode(
                    encrypted_bytes
                ).decode("utf-8")

            encrypted_value = encrypt_value(value=value, signing_key=signing_key)  # type: ignore
            # Use urlsafe_b64encode for URL-safe base64 encoding (replaces + with - and / with _)
            encrypted_value = base64.urlsafe_b64encode(encrypted_value).decode("utf-8")

            return encrypted_value

        verbose_proxy_logger.debug(
            f"Invalid value type passed to encrypt_value: {type(value)} for Value: {value}\n Value must be a string"
        )
        # if it's not a string - do not encrypt it and return the value
        return value
    except Exception as e:
        raise e


def decrypt_value_helper(
    value: str,
    key: str,  # this is just for debug purposes, showing the k,v pair that's invalid. not a signing key.
    exception_type: Literal["debug", "error"] = "error",
    return_original_value: bool = False,
):
    signing_key = _get_salt_key()

    try:
        if isinstance(value, str):
            if value.startswith(_AES_GCM_PREFIX):
                # AES-256-GCM encrypted value (FIPS-compliant path)
                b64_part = value[len(_AES_GCM_PREFIX) :]
                encrypted_bytes = base64.urlsafe_b64decode(b64_part)
                return _decrypt_aes_gcm(encrypted_bytes, signing_key)  # type: ignore

            # Legacy nacl-encrypted value
            # Try URL-safe base64 decoding first (new format)
            # Fall back to standard base64 decoding for backwards compatibility (old format)
            try:
                decoded_b64 = base64.urlsafe_b64decode(value)
            except Exception:
                # If URL-safe decoding fails, try standard base64 decoding for backwards compatibility
                decoded_b64 = base64.b64decode(value)

            value = decrypt_value(value=decoded_b64, signing_key=signing_key)  # type: ignore
            return value

        # if it's not str - do not decrypt it, return the value
        return value
    except Exception as e:
        error_message = f"Error decrypting value for key: {key}, Did your master_key/salt key change recently? \nError: {str(e)}\nSet permanent salt key - https://docs.litellm.ai/docs/proxy/prod#5-set-litellm-salt-key"
        if exception_type == "debug":
            verbose_proxy_logger.debug(error_message)
            return value if return_original_value else None

        verbose_proxy_logger.debug(
            f"Unable to decrypt value={value} for key: {key}, returning None"
        )
        if return_original_value:
            return value
        else:
            verbose_proxy_logger.exception(error_message)
            # [Non-Blocking Exception. - this should not block decrypting other values]
            return None


def encrypt_value(value: str, signing_key: str):
    import hashlib

    import nacl.secret
    import nacl.utils

    # get 32 byte master key #
    hash_object = hashlib.sha256(signing_key.encode())
    hash_bytes = hash_object.digest()

    # initialize secret box #
    box = nacl.secret.SecretBox(hash_bytes)

    # encode message #
    value_bytes = value.encode("utf-8")

    encrypted = box.encrypt(value_bytes)

    return encrypted


def decrypt_value(value: bytes, signing_key: str) -> str:
    import hashlib

    import nacl.secret
    import nacl.utils

    # get 32 byte master key #
    hash_object = hashlib.sha256(signing_key.encode())
    hash_bytes = hash_object.digest()

    # initialize secret box #
    box = nacl.secret.SecretBox(hash_bytes)

    # Convert the bytes object to a string
    try:
        if len(value) == 0:
            return ""

        plaintext = box.decrypt(value)
        plaintext = plaintext.decode("utf-8")  # type: ignore
        return plaintext  # type: ignore
    except Exception as e:
        raise e
