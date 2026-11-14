from typing import Final

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from litellm.proxy.auth.litellm_license import Entitlements, LicenseCheck
from tests._support.license import license_verifier, serve_license_server, sign_license


@pytest.fixture(scope="module")
def private_key() -> RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _licensed(monkeypatch: pytest.MonkeyPatch, private_key: RSAPrivateKey, **claims: object) -> LicenseCheck:
    monkeypatch.setenv("LITELLM_LICENSE", sign_license(private_key, {"expiration_date": "2999-01-01", **claims}))
    license_check: Final = LicenseCheck(license_verifier(private_key))
    assert license_check.is_premium() is True
    return license_check


def test_signed_claims_reach_the_proxy_as_license_data(
    monkeypatch: pytest.MonkeyPatch, private_key: RSAPrivateKey
) -> None:
    license_check: Final = _licensed(
        monkeypatch, private_key, user_id="u", allowed_features=["sso"], max_users=5, unknown_claim=True
    )

    assert license_check.airgapped_license_data == {
        "expiration_date": "2999-01-01",
        "user_id": "u",
        "allowed_features": ["sso"],
        "max_users": 5,
    }


@pytest.mark.parametrize(("allowed_features", "limit"), [(["auto_router"], None), (["sso"], 1)])
def test_auto_router_limit_lifts_only_for_a_license_granting_it(
    monkeypatch: pytest.MonkeyPatch, private_key: RSAPrivateKey, allowed_features: list[str], limit: int | None
) -> None:
    license_check: Final = _licensed(monkeypatch, private_key, allowed_features=allowed_features)

    assert license_check.auto_router_capability_limit() == limit


def test_an_unknown_feature_name_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown license feature: auto-router"):
        Entitlements.COMMUNITY.grants("auto-router")  # pyright: ignore[reportArgumentType]  # proves the runtime check


def test_a_rejected_license_replaces_the_previous_entitlements(
    monkeypatch: pytest.MonkeyPatch, private_key: RSAPrivateKey
) -> None:
    monkeypatch.setenv(
        "LITELLM_LICENSE", sign_license(private_key, {"expiration_date": "2999-01-01", "allowed_features": ["*"]})
    )
    with serve_license_server(lambda _: (200, b'{"verify": false}')) as (url, _requested):
        license_check: Final = LicenseCheck(license_verifier(private_key, url))
        assert license_check.is_premium() is True
        license_check.license_str = sign_license(
            private_key, {"expiration_date": "2000-01-01", "allowed_features": ["*"]}
        )
        assert license_check.is_premium() is False

    assert license_check.airgapped_license_data is None
    assert license_check.auto_router_capability_limit() == 1
