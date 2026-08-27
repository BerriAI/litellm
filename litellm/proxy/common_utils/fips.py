"""FIPS mode selection for the proxy's cryptographic primitives.

``LITELLM_FIPS_MODE=true`` restricts the proxy to FIPS 140-approved algorithms:
PBKDF2-HMAC-SHA256 replaces scrypt for password hashing and master-key
derivation, and AES-256-GCM replaces XSalsa20-Poly1305 for at-rest credential
encryption. Decrypt/verify stays format-detecting where the algorithm is
FIPS-approved; values produced by non-approved algorithms are rejected with an
actionable error instead of being silently accepted.
"""

import hashlib
from typing import Final

FIPS_MODE_ENV_VAR: Final = "LITELLM_FIPS_MODE"

# OWASP recommended floor for PBKDF2-HMAC-SHA256 (2023 password storage cheat sheet).
PBKDF2_ITERATIONS: Final = 600_000


def is_fips_mode() -> bool:
    from litellm.secret_managers.main import get_secret_bool

    return get_secret_bool(FIPS_MODE_ENV_VAR, False) is True


def derive_master_subkey_hex(master_key: str, domain_salt: bytes, dklen: int) -> str:
    """Derive a domain-separated subkey from the proxy master key as lowercase hex.

    PBKDF2-HMAC-SHA256 under FIPS mode, scrypt (RFC 7914, n=2**15/r=8/p=1) otherwise.
    Both are deliberately expensive so a captured token is not a cheap offline oracle
    for the master key. The two modes derive different keys, so toggling
    ``LITELLM_FIPS_MODE`` invalidates outstanding tokens minted under the other mode.
    """
    if is_fips_mode():
        return hashlib.pbkdf2_hmac("sha256", master_key.encode(), domain_salt, PBKDF2_ITERATIONS, dklen=dklen).hex()
    scrypt_n: Final = 2**15
    scrypt_r: Final = 8
    scrypt_p: Final = 1
    return hashlib.scrypt(
        master_key.encode(),
        salt=domain_salt,
        n=scrypt_n,
        r=scrypt_r,
        p=scrypt_p,
        maxmem=128 * scrypt_n * scrypt_r * scrypt_p * 2,
        dklen=dklen,
    ).hex()
