"""Turns on LiteLLM's general_settings.fips_mode for the fixture scripts, on an OpenSSL with no FIPS provider.

With fips_mode on, Python runs KBKDF and AES-256-GCM through OpenSSL's EVP_KDF and EVP_CIPHER, fetched with the
properties fips=yes from the libcrypto that _hashlib loaded. Here the same ctypes code is pointed at the system
libcrypto with default properties, so OpenSSL's default provider runs it. KBKDF and AES-GCM are the same
algorithms in either provider, and the derived key is checked against the cryptography library's KBKDFHMAC.
Import this after stubbing litellm.proxy.proxy_server.
"""

import _hashlib
import ctypes.util
import sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.kbkdf import KBKDFHMAC, CounterLocation, Mode

from litellm.proxy.common_utils import encrypt_decrypt_utils, fips


def enter_fips_mode() -> None:
    libcrypto = ctypes.util.find_library("crypto")
    if libcrypto is None:
        sys.exit("No system libcrypto to run EVP_KDF and EVP_CIPHER")
    setattr(_hashlib, "__file__", libcrypto)
    setattr(encrypt_decrypt_utils, "_FIPS_PROPERTIES", b"")
    fips._FIPS_MODE_ACTIVE.set()


def kbkdf(key: bytes, label: bytes, context: bytes) -> bytes:
    """Python's own KBKDF through OpenSSL, after checking it against cryptography's KBKDFHMAC with OpenSSL's default
    parameters (32-bit counter before the fixed input, zero separator, 32-bit L)."""
    derived = encrypt_decrypt_utils._kbkdf_hmac_sha256(key, label, context)
    expected = KBKDFHMAC(
        algorithm=hashes.SHA256(),
        mode=Mode.CounterMode,
        length=len(derived),
        rlen=4,
        llen=4,
        location=CounterLocation.BeforeFixed,
        label=label,
        context=context,
        fixed=None,
    ).derive(key)
    if derived != expected:
        sys.exit(f"OpenSSL and cryptography disagree on KBKDF for label {label!r}")
    return derived
