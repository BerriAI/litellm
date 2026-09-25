import base64
import hashlib
import os
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Final, Literal, cast

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.fips import FipsModeError, is_fips_mode

if TYPE_CHECKING:
    from nacl.secret import SecretBox

# Versioned ciphertext markers for AES-256-GCM values, each
# "<version>:gcm:" + base64url(nonce(12) || ciphertext || tag(16)).
# v3 derives the key with HKDF-SHA256, v2 with a raw SHA-256 of the salt key.
# Legacy XSalsa20-Poly1305 (nacl) values carry no marker; the colon in the
# prefix can never appear in base64url(nacl output), so the prefix check is an
# unambiguous discriminator between the formats on read.
_V3_GCM_PREFIX: Final = "v3:gcm:"
_V2_GCM_PREFIX: Final = "v2:gcm:"
_GCM_PREFIXES: Final = (_V3_GCM_PREFIX, _V2_GCM_PREFIX)
_HKDF_INFO: Final = b"litellm-at-rest-v3"

_ENCRYPTION_ALGORITHM_SETTING: Final = "encryption_algorithm"
_ALGO_AES_GCM: Final = "aes-256-gcm"
_ALGO_XSALSA20: Final = "xsalsa20-poly1305"

_NACL_MIN_CIPHERTEXT_BYTES: Final = 40
_LEGACY_ENCRYPTION_HELP: Final = (
    "Install the legacy-encryption extra (pip install 'litellm[legacy-encryption]') on a non-FIPS image and "
    "re-encrypt stored credentials with `litellm-proxy encryption migrate` (POST /credentials/migrate-encryption) "
    "before running without PyNaCl"
)


class LegacyEncryptionUnavailableError(RuntimeError):
    pass


def is_versioned_gcm(value: str) -> bool:
    return value.startswith(_GCM_PREFIXES)


def _get_salt_key():
    from litellm.proxy.proxy_server import master_key

    salt_key = os.getenv("LITELLM_SALT_KEY", None)

    if salt_key is None:
        salt_key = master_key

    return salt_key


def _get_encryption_algorithm() -> str:
    """Resolve the at-rest encryption algorithm for new writes from ``general_settings.encryption_algorithm``.

    Defaults to AES-256-GCM. ``xsalsa20-poly1305`` stays available as an explicit opt-in for deployments that
    still need byte-for-byte legacy output, except under ``LITELLM_FIPS_MODE`` where it is refused.
    """
    try:
        from litellm.proxy.proxy_server import general_settings

        algo: Final = general_settings.get(_ENCRYPTION_ALGORITHM_SETTING, _ALGO_AES_GCM)
    except Exception:  # noqa: BLE001  # proxy_server is not importable in SDK-only use of these helpers
        return _ALGO_AES_GCM

    if not isinstance(algo, str) or algo.lower() != _ALGO_XSALSA20:
        return _ALGO_AES_GCM
    if is_fips_mode():
        raise FipsModeError(
            f"general_settings.{_ENCRYPTION_ALGORITHM_SETTING}={_ALGO_XSALSA20} is not allowed under "
            f"LITELLM_FIPS_MODE: XSalsa20-Poly1305 is not a FIPS approved algorithm. Remove the setting to write "
            f"{_ALGO_AES_GCM}"
        )
    return _ALGO_XSALSA20


def _derive_key_sha256(signing_key: str) -> bytes:
    """Historical derivation shared by legacy nacl values and ``v2:gcm:`` values: one unsalted SHA-256."""
    return hashlib.sha256(signing_key.encode()).digest()


def _derive_key_hkdf(signing_key: str) -> bytes:
    """Derivation for ``v3:gcm:`` values: HKDF-SHA256 with a fixed info string."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_HKDF_INFO).derive(signing_key.encode())


def _encrypt_aes_gcm(value: str, signing_key: str) -> str:
    """Encrypt under AES-256-GCM and return the versioned ``v3:gcm:`` string."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce: Final = os.urandom(12)
    # AESGCM.encrypt returns ciphertext || tag(16); wire format is nonce || that.
    blob: Final = AESGCM(_derive_key_hkdf(signing_key)).encrypt(nonce, value.encode("utf-8"), None)
    return _V3_GCM_PREFIX + base64.urlsafe_b64encode(nonce + blob).decode("utf-8")


def _decrypt_aes_gcm(value: str, signing_key: str) -> str:
    """Decrypt a versioned ``v3:gcm:`` or ``v2:gcm:`` string, deriving the key the way its version was written."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if value.startswith(_V3_GCM_PREFIX):
        key, prefix = _derive_key_hkdf(signing_key), _V3_GCM_PREFIX
    else:
        key, prefix = _derive_key_sha256(signing_key), _V2_GCM_PREFIX
    raw: Final = base64.urlsafe_b64decode(value[len(prefix) :])
    # An empty plaintext still serializes to nonce(12) || tag(16) = 28 bytes, so a
    # short/empty buffer here is a corrupt value: let AESGCM.decrypt raise and be
    # swallowed by decrypt_value_helper (returns None/original), same as legacy.
    nonce, blob = raw[:12], raw[12:]
    return AESGCM(key).decrypt(nonce, blob, None).decode("utf-8")


def encrypt_value_helper(value: str, new_encryption_key: str | None = None):
    signing_key: Final = new_encryption_key or _get_salt_key()

    try:
        if isinstance(value, str):
            if _get_encryption_algorithm() == _ALGO_AES_GCM:
                # AES path: the v3:gcm: output is already a base64url string, so it
                # is returned directly with no extra base64 wrapper.
                return _encrypt_aes_gcm(value=value, signing_key=cast(str, signing_key))

            encrypted_value = encrypt_value(value=value, signing_key=signing_key)
            # Use urlsafe_b64encode for URL-safe base64 encoding (replaces + with - and / with _)
            encrypted_value = base64.urlsafe_b64encode(encrypted_value).decode("utf-8")

            return encrypted_value

        verbose_proxy_logger.debug(
            "Invalid value type passed to encrypt_value: %s for Value: %s\n Value must be a string", type(value), value
        )
        # if it's not a string - do not encrypt it and return the value
        return value
    except Exception as e:
        raise e


def _legacy_ciphertext_bytes(value: str) -> bytes:
    # Try URL-safe base64 decoding first (new format)
    # Fall back to standard base64 decoding for backwards compatibility (old format)
    try:
        return base64.urlsafe_b64decode(value)
    except Exception:
        return base64.b64decode(value)


def _decrypt_with_signing_key(value: str, signing_key: str) -> str:
    # Versioned AES-256-GCM values are detected before any base64 decode.
    # The prefix is the algorithm tag the legacy nacl format never carried.
    if is_versioned_gcm(value):
        return _decrypt_aes_gcm(value=value, signing_key=signing_key)

    return decrypt_value(value=_legacy_ciphertext_bytes(value), signing_key=signing_key)


def decrypt_if_encrypted_with(value: str, signing_key: str) -> str | None:
    """None unless value is a ciphertext under signing_key.

    A legacy ciphertext met without PyNaCl installed is logged with the re-encrypt path and read as None,
    never handed back as if it were the plaintext.
    """
    try:
        # base64 decoding skips characters outside its alphabet, so "" and "*" decode to no bytes,
        # which decrypt_value reads as an empty plaintext under any key.
        decodes_to_nothing: Final = not is_versioned_gcm(value) and not _legacy_ciphertext_bytes(value)
        return None if decodes_to_nothing else _decrypt_with_signing_key(value=value, signing_key=signing_key)
    except LegacyEncryptionUnavailableError as error:
        verbose_proxy_logger.error("%s", error)
        return None
    except Exception:  # noqa: BLE001  # base64, nacl and AES-GCM each raise their own "not a ciphertext" type
        return None


def decrypt_value_helper(
    value: str,
    key: str,  # this is just for debug purposes, showing the k,v pair that's invalid. not a signing key.
    exception_type: Literal["debug", "error"] = "error",
    return_original_value: bool = False,
) -> str | None:
    signing_key: Final = _get_salt_key()

    try:
        if isinstance(value, str):
            return _decrypt_with_signing_key(value=value, signing_key=cast(str, signing_key))

        # if it's not str - do not decrypt it, return the value
        return value
    except LegacyEncryptionUnavailableError as error:
        verbose_proxy_logger.error("Cannot decrypt value for key: %s. %s", key, error)
        return None
    except Exception as e:
        error_message = f"Error decrypting value for key: {key}, Did your master_key/salt key change recently? \nError: {e}\nSet permanent salt key - https://docs.litellm.ai/docs/proxy/prod#5-set-litellm-salt-key"
        if exception_type == "debug":
            verbose_proxy_logger.debug(error_message)
            return value if return_original_value else None

        verbose_proxy_logger.debug("Unable to decrypt value for key: %s, returning None", key)
        if return_original_value:
            return value
        else:
            verbose_proxy_logger.exception(error_message)
            # [Non-Blocking Exception. - this should not block decrypting other values]
            return None


def legacy_encryption_available() -> bool:
    """True when PyNaCl is importable, so legacy xsalsa20-poly1305 ciphertext can be read."""
    try:
        import nacl.secret  # noqa: F401  # probe only
    except ImportError:
        return False
    return True


def needs_legacy_reader(value: object) -> bool:
    """True for a stored string that only PyNaCl can tell apart from plaintext: non empty and unprefixed."""
    return isinstance(value, str) and value != "" and not is_versioned_gcm(value)


def require_legacy_reader_for(values: Iterable[object], purpose: str) -> None:
    """Refuse a decrypt-then-rewrite pass when PyNaCl is missing and one of the values is unprefixed: it would
    read as unreadable and be dropped, double wrapped or miscounted as plaintext. Versioned gcm and non string
    values never need PyNaCl, so a fully migrated store passes."""
    if legacy_encryption_available() or not any(needs_legacy_reader(value) for value in values):
        return
    raise LegacyEncryptionUnavailableError(
        f"Cannot {purpose}: PyNaCl is needed to read legacy {_ALGO_XSALSA20} values. {_LEGACY_ENCRYPTION_HELP}"
    )


def _legacy_secret_box(signing_key: str, purpose: str) -> "SecretBox":
    try:
        import nacl.secret
    except ImportError as error:
        raise LegacyEncryptionUnavailableError(
            f"Cannot {purpose} with the legacy {_ALGO_XSALSA20} algorithm: PyNaCl is not installed. "
            f"{_LEGACY_ENCRYPTION_HELP}"
        ) from error
    return nacl.secret.SecretBox(_derive_key_sha256(signing_key))


def encrypt_value(value: str, signing_key: str) -> bytes:
    return bytes(_legacy_secret_box(signing_key, "encrypt").encrypt(value.encode("utf-8")))


def decrypt_value(value: bytes, signing_key: str) -> str:
    if len(value) == 0:
        return ""
    if len(value) < _NACL_MIN_CIPHERTEXT_BYTES:
        raise ValueError(f"Value of {len(value)} bytes is too short to be a {_ALGO_XSALSA20} ciphertext")
    return _legacy_secret_box(signing_key, "decrypt a stored value").decrypt(value).decode("utf-8")


class SecretMapDecodeError(RuntimeError):
    pass


_SECRET_MAP: Final = TypeAdapter(Mapping[str, str])
_STORED_SECRET_MAP: Final = TypeAdapter(Mapping[str, str] | str)
_SECRET_STRING: Final = TypeAdapter(str)


def encrypt_secret_map(value: Mapping[str, str], new_encryption_key: str | None = None) -> str:
    if not value:
        return "{}"
    ciphertext: Final = _SECRET_STRING.validate_python(
        encrypt_value_helper(_SECRET_MAP.dump_json(value).decode(), new_encryption_key=new_encryption_key), strict=True
    )
    return _SECRET_STRING.dump_json(ciphertext).decode()


def decode_secret_map(value: object, *, key: str) -> Mapping[str, str] | None:
    if value is None:
        return None
    try:
        stored: Final = (
            _STORED_SECRET_MAP.validate_json(value, strict=True)
            if isinstance(value, str) and value.lstrip().startswith(("{", '"'))
            else _STORED_SECRET_MAP.validate_python(value, strict=True)
        )
        if not isinstance(stored, str):
            return stored
        decrypted: Final = decrypt_value_helper(
            value=stored, key=key, exception_type="debug", return_original_value=False
        )
        return _SECRET_MAP.validate_json(decrypted, strict=True)
    except ValidationError:
        raise SecretMapDecodeError(f"Cannot decode encrypted MCP {key}; check LITELLM_SALT_KEY") from None
