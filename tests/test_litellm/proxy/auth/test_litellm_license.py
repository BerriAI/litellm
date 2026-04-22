import base64
import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from litellm.proxy.auth.litellm_license import LicenseCheck


def test_read_public_key_loads_successfully():
    """Ensure public_key.pem is valid PEM with no leading whitespace."""
    license_check = LicenseCheck()
    assert (
        license_check.public_key is not None
    ), "public_key.pem could not be loaded — check for leading whitespace or malformed PEM header"


def test_is_over_limit():
    license_check = LicenseCheck()
    license_check.airgapped_license_data = {"max_users": 100}
    assert license_check.is_over_limit(101) is True
    assert license_check.is_over_limit(100) is False
    assert license_check.is_over_limit(99) is False

    license_check.airgapped_license_data = {}
    assert license_check.is_over_limit(101) is False
    assert license_check.is_over_limit(100) is False
    assert license_check.is_over_limit(99) is False

    license_check.airgapped_license_data = None
    assert license_check.is_over_limit(101) is False
    assert license_check.is_over_limit(100) is False
    assert license_check.is_over_limit(99) is False


def _generate_signed_license(payload: dict) -> tuple[str, object]:
    """Generate a license string signed with a fresh RSA-2048 keypair.

    Returns (license_str, public_key) so tests can inject the public key
    into LicenseCheck without depending on the bundled public_key.pem.

    Format matches LiteLLM's license encoding:
      base64( json_bytes + b"." + rsa_pss_signature_bytes )
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    message = json.dumps(payload).encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    license_bytes = message + b"." + signature
    return base64.b64encode(license_bytes).decode("ascii"), public_key


def test_verify_license_without_dot_in_payload():
    """Baseline: license whose JSON payload contains no '.' verifies correctly.

    Covers the "legacy happy path" that the current split(b".", 1) implementation
    relies on. Must continue to pass after any fix to the split logic.
    """
    license_str, public_key = _generate_signed_license(
        {
            "expiration_date": "2099-12-31",
            "user_id": "no-dot-user",
            "allowed_features": ["*"],
            "max_users": 10,
            "max_teams": 2,
        }
    )
    lc = LicenseCheck()
    assert (
        lc.verify_license_without_api_request(
            public_key=public_key, license_key=license_str
        )
        is True
    )


def test_verify_license_with_dot_in_user_id():
    """Regression for the split-on-first-dot bug.

    When user_id contains a literal '.' (e.g. domain-style "foo.co.jp-license"),
    `decoded.split(b".", 1)` splits the bytes at the first '.' in the JSON
    payload instead of the delimiter between JSON and signature, causing
    signature verification to fail for a properly-signed license.

    This test builds a valid license with such a user_id and asserts that
    local verification succeeds. Fails on main (L178 split bug); should
    pass once split logic handles JSON-embedded dots correctly.
    """
    license_str, public_key = _generate_signed_license(
        {
            "expiration_date": "2099-12-31",
            "user_id": "acme.co.jp-license-litellm",
            "allowed_features": ["*"],
            "max_users": 3000,
            "max_teams": 5,
        }
    )
    lc = LicenseCheck()
    assert (
        lc.verify_license_without_api_request(
            public_key=public_key, license_key=license_str
        )
        is True
    ), (
        "License with '.' in user_id must verify locally. "
        "The split(b'.', 1) implementation misparses JSON-embedded dots."
    )


def test_verify_license_with_multiple_dots_in_payload():
    """Boundary: license whose JSON payload contains multiple '.' bytes.

    Ensures the length-based split handles payloads with several dots
    (e.g. email-like fields, versioned identifiers) without misparsing.
    """
    license_str, public_key = _generate_signed_license(
        {
            "expiration_date": "2099-12-31",
            "user_id": "a.b.c.d-multi-dot",
            "allowed_features": ["*"],
            "max_users": 100,
            "max_teams": 10,
        }
    )
    lc = LicenseCheck()
    assert (
        lc.verify_license_without_api_request(
            public_key=public_key, license_key=license_str
        )
        is True
    )


def test_verify_license_rejects_invalid_format():
    """Licenses with unrecognized signature length must fail gracefully.

    Feeds a string that base64-decodes to arbitrary bytes without a valid
    delimiter at any supported RSA signature length. The outer try/except
    inside verify_license_without_api_request should swallow the ValueError
    and return False (not propagate or return True).
    """
    # 100 random bytes, no '.' at positions expected by length-based split
    garbage = base64.b64encode(b"\x00" * 100).decode("ascii")
    lc = LicenseCheck()
    # public_key is irrelevant here since we fail before signature verification
    assert (
        lc.verify_license_without_api_request(
            public_key=lc.public_key, license_key=garbage
        )
        is False
    )
