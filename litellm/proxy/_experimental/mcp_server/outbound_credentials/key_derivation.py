"""Key derivation for the MCP outbound-credential token families.

Bridge envelopes and gateway session tokens are keyed by HKDF-SHA256 (RFC 5869) over the
high-entropy proxy ``master_key`` with a per-family domain label as ``info``. The legacy
scrypt derivation is kept for one release as an opt-in read fallback so tokens minted
before the rotation still open during the grace window; it is never used under FIPS.
"""

import hashlib
import os
from collections.abc import Callable
from typing import Final

from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from litellm.proxy.common_utils.fips import is_fips_mode

LEGACY_KDF_GRACE_ENV_VAR: Final = "LITELLM_MCP_LEGACY_KDF_GRACE"
DERIVED_KEY_BYTES: Final = 32

# RFC 7914 work factors of the pre-rotation derivation; maxmem is twice the working set.
_SCRYPT_N: Final = 2**15
_SCRYPT_R: Final = 8
_SCRYPT_P: Final = 1
_SCRYPT_MAXMEM: Final = 128 * _SCRYPT_N * _SCRYPT_R * _SCRYPT_P * 2


def hkdf_sha256(master_key: str, info: bytes) -> str:
    """Derive a 256-bit subkey from ``master_key`` bound to the domain label ``info``."""
    return HKDF(algorithm=SHA256(), length=DERIVED_KEY_BYTES, salt=None, info=info).derive(master_key.encode()).hex()


def legacy_scrypt(master_key: str, salt: bytes) -> str | None:
    """The pre-rotation derivation. ``None`` when hashlib.scrypt is unavailable (some
    OpenSSL builds, including FIPS providers, do not expose it)."""
    try:
        return hashlib.scrypt(
            master_key.encode(),
            salt=salt,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            maxmem=_SCRYPT_MAXMEM,
            dklen=DERIVED_KEY_BYTES,
        ).hex()
    except (AttributeError, ValueError):
        return None


def legacy_kdf_grace_enabled(environ: Callable[[str], str | None] = os.environ.get) -> bool:
    """True only when the operator opted the legacy read fallback in and the deployment is
    not in FIPS mode (scrypt is not a FIPS-approved KDF)."""
    if is_fips_mode(environ):
        return False
    return (environ(LEGACY_KDF_GRACE_ENV_VAR) or "").strip().lower() == "true"
