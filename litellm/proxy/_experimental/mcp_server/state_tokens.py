from __future__ import annotations

import base64
import binascii
import os
from enum import Enum
from typing import Final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error, Ok, Result

_PREFIX: Final = "mcp_state_v1."
_MAX_TOKEN_LENGTH: Final = 65536
_NONCE_BYTES: Final = 12


class StateTokenError(str, Enum):
    MISSING_KEY = "Set the same LITELLM_SALT_KEY on every replica to enable pagination"
    INVALID = "Invalid pagination state; start a fresh listing"
    EXPIRED = "Pagination state expired; start a fresh listing"
    TOO_LARGE = "Pagination state exceeds the supported size"


class _Envelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    expires_at: int
    value: JsonValue


def _cipher(purpose: str) -> Result[AESGCM, StateTokenError]:
    salt_key: Final = os.getenv("LITELLM_SALT_KEY")
    if not salt_key:
        return Error(StateTokenError.MISSING_KEY)
    if not purpose:
        return Error(StateTokenError.INVALID)
    key: Final = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"litellm:mcp:state:v1",
        info=purpose.encode("utf-8"),
    ).derive(salt_key.encode("utf-8"))
    return Ok(AESGCM(key))


def seal_state(value: JsonValue, *, purpose: str, expires_at: int, now: int) -> Result[str, StateTokenError]:
    cipher: Final = _cipher(purpose)
    if isinstance(cipher, Error):
        return cipher
    if expires_at <= now:
        return Error(StateTokenError.EXPIRED)
    plaintext: Final = _Envelope(expires_at=expires_at, value=value).model_dump_json().encode("utf-8")
    if len(plaintext) > _MAX_TOKEN_LENGTH:
        return Error(StateTokenError.TOO_LARGE)
    nonce: Final = os.urandom(_NONCE_BYTES)
    ciphertext: Final = cipher.ok.encrypt(nonce, plaintext, (_PREFIX + purpose).encode("utf-8"))
    token: Final = _PREFIX + base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii").rstrip("=")
    return Error(StateTokenError.TOO_LARGE) if len(token) > _MAX_TOKEN_LENGTH else Ok(token)


def open_state(token: str, *, purpose: str, now: int) -> Result[JsonValue, StateTokenError]:
    cipher: Final = _cipher(purpose)
    if isinstance(cipher, Error):
        return cipher
    if len(token) > _MAX_TOKEN_LENGTH:
        return Error(StateTokenError.TOO_LARGE)
    if not token.startswith(_PREFIX):
        return Error(StateTokenError.INVALID)
    encoded: Final = token[len(_PREFIX) :]
    try:
        sealed: Final = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        if len(sealed) < _NONCE_BYTES + 16 or base64.urlsafe_b64encode(sealed).decode("ascii").rstrip("=") != encoded:
            return Error(StateTokenError.INVALID)
        plaintext: Final = cipher.ok.decrypt(
            sealed[:_NONCE_BYTES], sealed[_NONCE_BYTES:], (_PREFIX + purpose).encode("utf-8")
        )
        envelope: Final = _Envelope.model_validate_json(plaintext)
    except (binascii.Error, ValueError, InvalidTag, ValidationError):
        return Error(StateTokenError.INVALID)
    if envelope.expires_at <= now:
        return Error(StateTokenError.EXPIRED)
    return Ok(envelope.value)
