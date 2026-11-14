import json
import logging
import socket
from typing import Final

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from litellm.proxy.auth.litellm_license import MISSING_AUTO_ROUTER_LICENSE, Entitlements, LicenseCheck
from tests._support.license import license_verifier, serve_license_server, sign_license

UNSIGNED_LICENSE: Final = "test-only-private-license-marker"


@pytest.fixture(scope="module")
def private_key() -> RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def no_env_license(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_LICENSE", raising=False)


def _licensed(monkeypatch: pytest.MonkeyPatch, private_key: RSAPrivateKey, **claims: object) -> LicenseCheck:
    monkeypatch.setenv("LITELLM_LICENSE", sign_license(private_key, {"expiration_date": "2999-01-01", **claims}))
    license_check: Final = LicenseCheck(license_verifier(private_key))
    assert license_check.is_premium() is True
    return license_check


def test_default_verifier_rejects_a_license_the_litellm_key_did_not_sign(private_key: RSAPrivateKey) -> None:
    entitlements: Final = LicenseCheck().verifier.entitlements(
        sign_license(private_key, {"expiration_date": "2999-01-01"})
    )

    assert entitlements.license is None


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


@pytest.mark.parametrize(("total", "over"), [(99, False), (100, False), (101, True)])
def test_seat_and_team_limits_come_from_the_signed_license(
    monkeypatch: pytest.MonkeyPatch, private_key: RSAPrivateKey, total: int, over: bool
) -> None:
    license_check: Final = _licensed(monkeypatch, private_key, max_users=100, max_teams=100)

    assert license_check.user_seat_limit() == 100
    assert license_check.is_over_limit(total) is over
    assert license_check.is_team_count_over_limit(total) is over


def test_no_license_means_community_without_limits(no_env_license: None) -> None:
    license_check: Final = LicenseCheck()

    assert license_check.is_premium() is False
    assert license_check.airgapped_license_data is None
    assert license_check.user_seat_limit() is None
    assert license_check.is_over_limit(10**9) is False
    assert license_check.is_team_count_over_limit(10**9) is False


@pytest.mark.parametrize(
    ("allowed_features", "limit"),
    [
        (["auto_router"], None),
        (["sso", "auto_router", "audit_logs"], None),
        (["*"], None),
        (["sso", "*"], None),
        ("*", None),
        (["sso"], 1),
        (None, 1),
    ],
)
def test_auto_router_limit_lifts_only_for_a_license_granting_it(
    monkeypatch: pytest.MonkeyPatch, private_key: RSAPrivateKey, allowed_features: object, limit: int | None
) -> None:
    claims: Final = {} if allowed_features is None else {"allowed_features": allowed_features}

    assert _licensed(monkeypatch, private_key, **claims).auto_router_capability_limit() == limit


def test_missing_auto_router_license_names_the_license_feature() -> None:
    assert MISSING_AUTO_ROUTER_LICENSE == "A LiteLLM license with the 'auto_router' feature lifts the limit."


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


@pytest.mark.parametrize(
    ("reply", "premium"),
    [
        ({"verify": True}, True),
        ({"verify": False}, False),
        (["verify", True], False),
        ({"verify": "true"}, False),
        ({"verified": True}, False),
    ],
)
def test_is_premium_follows_the_license_server_for_an_unsigned_license(
    monkeypatch: pytest.MonkeyPatch, private_key: RSAPrivateKey, reply: object, premium: bool
) -> None:
    monkeypatch.setenv("LITELLM_LICENSE", UNSIGNED_LICENSE)
    with serve_license_server(lambda _: (200, json.dumps(reply).encode())) as (url, requested):
        license_check: Final = LicenseCheck(license_verifier(private_key, url))
        assert license_check.is_premium() is premium

    assert requested == [f"/verify_license/{UNSIGNED_LICENSE}"]
    assert license_check.airgapped_license_data is None
    assert license_check.auto_router_capability_limit() == 1


def test_license_server_errors_are_retried_three_times_then_not_premium(
    monkeypatch: pytest.MonkeyPatch, private_key: RSAPrivateKey
) -> None:
    monkeypatch.setenv("LITELLM_LICENSE", UNSIGNED_LICENSE)
    with serve_license_server(lambda _: (503, b"")) as (url, requested):
        assert LicenseCheck(license_verifier(private_key, url)).is_premium() is False

    assert len(requested) == 3


def test_a_signed_license_never_asks_the_license_server(
    monkeypatch: pytest.MonkeyPatch, private_key: RSAPrivateKey
) -> None:
    monkeypatch.setenv("LITELLM_LICENSE", sign_license(private_key, {"expiration_date": "2999-01-01"}))
    with serve_license_server(lambda _: (200, b'{"verify": false}')) as (url, requested):
        assert LicenseCheck(license_verifier(private_key, url)).is_premium() is True

    assert requested == []


def _closed_port_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


@pytest.mark.parametrize("license_server", ["unreachable", "failing", "malformed"])
def test_license_verification_logs_never_include_the_license(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    private_key: RSAPrivateKey,
    license_server: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="LiteLLM")
    caplog.set_level(logging.DEBUG, logger="LiteLLM Proxy")
    monkeypatch.setenv("LITELLM_LICENSE", UNSIGNED_LICENSE)
    replies: Final = {"failing": (500, UNSIGNED_LICENSE.encode()), "malformed": (200, b'{"verify": "x"}')}

    if license_server == "unreachable":
        assert LicenseCheck(license_verifier(private_key, _closed_port_url())).is_premium() is False
    else:
        with serve_license_server(lambda _: replies[license_server]) as (url, _requested):
            assert LicenseCheck(license_verifier(private_key, url)).is_premium() is False

    assert "license not verified by the license server" in caplog.text
    assert "License configured after refresh: True" in caplog.text
    assert UNSIGNED_LICENSE not in caplog.text
