import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey


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


def test_heuristic_v2_router_limit() -> None:
    """Only the signed license's auto_router feature lifts the one-router limit; an API-verified
    license (no airgapped data) and an airgapped license without the feature keep it."""
    license_check = LicenseCheck()
    license_check.airgapped_license_data = {"expiration_date": "2999-01-01", "allowed_features": ["auto_router"]}
    assert license_check.heuristic_v2_router_limit() is None

    license_check.airgapped_license_data = {
        "expiration_date": "2999-01-01",
        "allowed_features": ["sso", "auto_router", "audit_logs"],
    }
    assert license_check.heuristic_v2_router_limit() is None

    license_check.airgapped_license_data = {"expiration_date": "2999-01-01", "allowed_features": ["sso"]}
    assert license_check.heuristic_v2_router_limit() == 1

    license_check.airgapped_license_data = {"expiration_date": "2999-01-01"}
    assert license_check.heuristic_v2_router_limit() == 1

    license_check.airgapped_license_data = None
    assert license_check.heuristic_v2_router_limit() == 1


def _signed_license(expiration_date: str) -> tuple[RSAPublicKey, str]:
    import base64

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    message = json.dumps(
        {"expiration_date": expiration_date, "user_id": "u", "allowed_features": ["auto_router"]}
    ).encode()
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    return private_key.public_key(), base64.b64encode(message + b"." + signature).decode()


def test_expired_or_unreadable_license_grants_no_features() -> None:
    """The verifier stores the signed payload only after the expiry check passes and clears it when a
    later verify rejects the license, so a stale payload cannot keep lifting the heuristic_v2 limit."""
    license_check = LicenseCheck()
    public_key, valid_key = _signed_license("2999-01-01")
    assert license_check.verify_license_without_api_request(public_key=public_key, license_key=valid_key) is True
    assert license_check.heuristic_v2_router_limit() is None

    _, expired_key = _signed_license("2000-01-01")
    assert license_check.verify_license_without_api_request(public_key=public_key, license_key=expired_key) is not True
    assert license_check.airgapped_license_data is None
    assert license_check.heuristic_v2_router_limit() == 1

    assert license_check.verify_license_without_api_request(public_key=public_key, license_key=valid_key) is True
    assert license_check.verify_license_without_api_request(public_key=public_key, license_key="not-a-license") is not True
    assert license_check.airgapped_license_data is None


def test_valid_signed_license_with_auto_router_lifts_the_limit() -> None:
    license_check = LicenseCheck()
    public_key, license_key = _signed_license("2999-01-01")

    assert license_check.verify_license_without_api_request(public_key=public_key, license_key=license_key) is True
    assert license_check.heuristic_v2_router_limit() is None
