import base64
import os
from typing import Literal, Optional, cast

from litellm._logging import verbose_proxy_logger

# Versioned ciphertext marker for AES-256-GCM values.
# Format: "v2:gcm:" + base64url(nonce(12) || ciphertext || tag(16)).
# Legacy XSalsa20-Poly1305 (nacl) values carry no marker; the colon in the
# prefix can never appear in base64url(nacl output), so the prefix check is an
# unambiguous discriminator between the two formats on read.
_V2_GCM_PREFIX = "v2:gcm:"

# general_settings key selecting the at-rest encryption algorithm for new writes.
# Default preserves the legacy algorithm so existing deployments are byte-for-byte
# unchanged until they explicitly opt in. Decrypt is always format-detecting, so
# flipping this flag forward (or back) never strands previously-written data.
_ENCRYPTION_ALGORITHM_SETTING = "encryption_algorithm"
_ALGO_AES_GCM = "aes-256-gcm"
_ALGO_XSALSA20 = "xsalsa20-poly1305"


def _get_salt_key():
    from litellm.proxy.proxy_server import master_key

    salt_key = os.getenv("LITELLM_SALT_KEY", None)

    if salt_key is None:
        salt_key = master_key

    return salt_key


def _get_encryption_algorithm() -> str:
    """
    Resolve the configured at-rest encryption algorithm for *new writes*.

    Read from ``general_settings.encryption_algorithm`` at write time. Defaults to
    the legacy XSalsa20-Poly1305 algorithm so deployments that have not opted in
    keep producing byte-for-byte identical ciphertext.
    """
    try:
        from litellm.proxy.proxy_server import general_settings

        algo = general_settings.get(_ENCRYPTION_ALGORITHM_SETTING, _ALGO_XSALSA20)
    except Exception:
        # general_settings may not be importable in some contexts (e.g. SDK-only
        # use of these helpers). Fall back to the legacy algorithm.
        return _ALGO_XSALSA20

    if isinstance(algo, str) and algo.lower() == _ALGO_AES_GCM:
        return _ALGO_AES_GCM
    return _ALGO_XSALSA20


def _derive_key(signing_key: str) -> bytes:
    """Derive a 32-byte key from the salt/master key (shared by both algorithms).

    Known limitation: this is a single-pass, unsalted ``SHA-256`` of the key, not
    a dedicated KDF (HKDF/PBKDF2). It is the *same* derivation the legacy nacl
    path already uses, so the AES path introduces no new weakness and stays
    interoperable with existing key sourcing; AES-256-GCM's per-value 12-byte
    random nonce gives the unique (key, nonce) pairs GCM requires. Moving both
    algorithms to HKDF-SHA256 would be more defensible in an audit but is a
    separate, coordinated change (it must re-derive or re-encrypt existing data).
    """
    import hashlib

    return hashlib.sha256(signing_key.encode()).digest()


def _encrypt_aes_gcm(value: str, signing_key: str) -> str:
    """Encrypt under AES-256-GCM and return the versioned ``v2:gcm:`` string."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    # AESGCM.encrypt returns ciphertext || tag(16); wire format is nonce || that.
    blob = AESGCM(_derive_key(signing_key)).encrypt(nonce, value.encode("utf-8"), None)
    return _V2_GCM_PREFIX + base64.urlsafe_b64encode(nonce + blob).decode("utf-8")


def _decrypt_aes_gcm(value: str, signing_key: str) -> str:
    """Decrypt a versioned ``v2:gcm:`` string produced by :func:`_encrypt_aes_gcm`."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    raw = base64.urlsafe_b64decode(value[len(_V2_GCM_PREFIX) :])
    # An empty plaintext still serializes to nonce(12) || tag(16) = 28 bytes, so a
    # short/empty buffer here is a corrupt value: let AESGCM.decrypt raise and be
    # swallowed by decrypt_value_helper (returns None/original), same as legacy.
    nonce, blob = raw[:12], raw[12:]
    return AESGCM(_derive_key(signing_key)).decrypt(nonce, blob, None).decode("utf-8")


def encrypt_value_helper(value: str, new_encryption_key: Optional[str] = None):
    signing_key = new_encryption_key or _get_salt_key()

    try:
        if isinstance(value, str):
            if _get_encryption_algorithm() == _ALGO_AES_GCM:
                # AES path: the v2:gcm: output is already a base64url string, so it
                # is returned directly with no extra base64 wrapper.
                return _encrypt_aes_gcm(value=value, signing_key=cast(str, signing_key))

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
            # Versioned AES-256-GCM values are detected before any base64 decode.
            # The prefix is the algorithm tag the legacy nacl format never carried.
            if value.startswith(_V2_GCM_PREFIX):
                return _decrypt_aes_gcm(value=value, signing_key=cast(str, signing_key))

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

        verbose_proxy_logger.debug(f"Unable to decrypt value={value} for key: {key}, returning None")
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
