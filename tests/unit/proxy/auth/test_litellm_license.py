import json
import logging
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.proxy.auth.litellm_license import LicenseCheck


def test_read_public_key_loads_successfully():
    """Ensure public_key.pem is valid PEM with no leading whitespace."""
    license_check = LicenseCheck()
    assert license_check.public_key is not None, (
        "public_key.pem could not be loaded — check for leading whitespace or malformed PEM header"
    )


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


def test_auto_router_capability_limit() -> None:
    """The signed license's auto_router feature or its "*" wildcard lifts the one-router limit; an
    API-verified license (no airgapped data) and an airgapped license without either keep it."""
    license_check = LicenseCheck()
    license_check.airgapped_license_data = {"expiration_date": "2999-01-01", "allowed_features": ["auto_router"]}
    assert license_check.auto_router_capability_limit() is None

    license_check.airgapped_license_data = {
        "expiration_date": "2999-01-01",
        "allowed_features": ["sso", "auto_router", "audit_logs"],
    }
    assert license_check.auto_router_capability_limit() is None

    license_check.airgapped_license_data = {"expiration_date": "2999-01-01", "allowed_features": ["*"]}
    assert license_check.auto_router_capability_limit() is None

    license_check.airgapped_license_data = {"expiration_date": "2999-01-01", "allowed_features": ["sso", "*"]}
    assert license_check.auto_router_capability_limit() is None

    license_check.airgapped_license_data = {"expiration_date": "2999-01-01", "allowed_features": ["sso"]}
    assert license_check.auto_router_capability_limit() == 1

    license_check.airgapped_license_data = {"expiration_date": "2999-01-01", "allowed_features": "*"}
    assert license_check.auto_router_capability_limit() is None

    license_check.airgapped_license_data = {"expiration_date": "2999-01-01"}
    assert license_check.auto_router_capability_limit() == 1

    license_check.airgapped_license_data = None
    assert license_check.auto_router_capability_limit() == 1


def _signed_license(
    expiration_date: str, allowed_features: tuple[str, ...] = ("auto_router",)
) -> tuple[RSAPublicKey, str]:
    import base64

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    message = json.dumps(
        {"expiration_date": expiration_date, "user_id": "u", "allowed_features": list(allowed_features)}
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
    assert license_check.auto_router_capability_limit() is None

    _, expired_key = _signed_license("2000-01-01")
    assert license_check.verify_license_without_api_request(public_key=public_key, license_key=expired_key) is not True
    assert license_check.airgapped_license_data is None
    assert license_check.auto_router_capability_limit() == 1

    assert license_check.verify_license_without_api_request(public_key=public_key, license_key=valid_key) is True
    assert (
        license_check.verify_license_without_api_request(public_key=public_key, license_key="not-a-license") is not True
    )
    assert license_check.airgapped_license_data is None


def test_valid_signed_license_with_auto_router_lifts_the_limit() -> None:
    license_check = LicenseCheck()
    public_key, license_key = _signed_license("2999-01-01")

    assert license_check.verify_license_without_api_request(public_key=public_key, license_key=license_key) is True
    assert license_check.auto_router_capability_limit() is None


def test_valid_signed_wildcard_license_lifts_the_limit() -> None:
    """The license generator defaults allowed_features to ["*"], meaning every feature, so a wildcard
    license grants auto_router the same way a license that names it does."""
    license_check = LicenseCheck()
    public_key, license_key = _signed_license("2999-01-01", allowed_features=("*",))

    assert license_check.verify_license_without_api_request(public_key=public_key, license_key=license_key) is True
    assert license_check.grants_feature("auto_router") is True
    assert license_check.auto_router_capability_limit() is None

    named_public_key, named_key = _signed_license("2999-01-01", allowed_features=("sso", "audit_logs"))
    assert license_check.verify_license_without_api_request(public_key=named_public_key, license_key=named_key) is True
    assert license_check.grants_feature("auto_router") is False
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
def test_is_premium_follows_the_license_server_reply_for_an_unsigned_license(
    monkeypatch: pytest.MonkeyPatch, reply: object, premium: bool
) -> None:
    monkeypatch.setenv("LITELLM_LICENSE", "license-the-public-key-did-not-sign")
    requested: Final[list[str]] = []

    def license_server(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, json=reply)

    license_check: Final = LicenseCheck()
    license_check.http_handler = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(license_server)))

    assert license_check.is_premium() is premium
    assert set(requested) == {"https://license.litellm.ai/verify_license/license-the-public-key-did-not-sign"}


@pytest.fixture
def license_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    logger: Final = logging.getLogger("test.license_verification")
    monkeypatch.setattr("litellm.proxy.auth.litellm_license.verbose_proxy_logger", logger)
    caplog.set_level(logging.DEBUG, logger=logger.name)
    return caplog


def _assert_no_license_in_logs(caplog: pytest.LogCaptureFixture, license_value: str) -> None:
    records: Final = tuple(record for record in caplog.records if record.name == "test.license_verification")
    assert records
    for record in records:
        assert license_value not in logging.Formatter().format(record)
        assert license_value not in repr(record.args)
        assert record.exc_info is None


@pytest.mark.parametrize("configured_at_init", [True, False])
@pytest.mark.parametrize("premium", [True, False])
def test_license_success_logs_do_not_include_license(
    monkeypatch: pytest.MonkeyPatch,
    license_logs: pytest.LogCaptureFixture,
    configured_at_init: bool,
    premium: bool,
) -> None:
    license_value: Final = "test-only-private-license-marker"
    monkeypatch.delenv("LITELLM_LICENSE", raising=False)
    if configured_at_init:
        monkeypatch.setenv("LITELLM_LICENSE", license_value)
    license_check: Final = LicenseCheck()
    monkeypatch.setenv("LITELLM_LICENSE", license_value)
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"verify": premium}))
    ) as client:
        license_check.http_handler = HTTPHandler(client=client)
        assert license_check.is_premium() is premium
    assert "License configured after refresh: True" in license_logs.text
    assert f"License is premium={premium}" in license_logs.text
    _assert_no_license_in_logs(license_logs, license_value)


@pytest.mark.parametrize("status_code", [404, 401, 403, 500, 503])
def test_license_http_error_logs_preserve_status_without_license(
    license_logs: pytest.LogCaptureFixture,
    status_code: int,
) -> None:
    license_value: Final = "test-only-private-license-marker"
    license_check: Final = LicenseCheck()
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(status_code, text=license_value))
    ) as client:
        license_check.http_handler = HTTPHandler(client=client)
        assert license_check._verify(license_value) is False
    assert f"error_type=HTTPStatusError status_code={status_code}" in license_logs.text
    _assert_no_license_in_logs(license_logs, license_value)


@pytest.mark.parametrize("error_class", [httpx.ReadTimeout, httpx.ConnectError, ValueError])
def test_license_exception_logs_omit_secret_message_and_traceback(
    license_logs: pytest.LogCaptureFixture,
    error_class: type[Exception],
) -> None:
    license_value: Final = "test-only-private-license-marker"
    license_check: Final = LicenseCheck()

    def fail(request: httpx.Request) -> httpx.Response:
        try:
            raise RuntimeError(license_value)
        except RuntimeError as cause:
            raise error_class(f"Failed verification: {request.url}") from cause

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        license_check.http_handler = HTTPHandler(client=client)
        assert license_check._verify(license_value) is False
    assert f"error_type={error_class.__name__}" in license_logs.text
    _assert_no_license_in_logs(license_logs, license_value)


def test_license_malformed_response_does_not_log_echoed_license(
    license_logs: pytest.LogCaptureFixture,
) -> None:
    license_value: Final = "test-only-private-license-marker"
    license_check: Final = LicenseCheck()
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"verify": license_value}))
    ) as client:
        license_check.http_handler = HTTPHandler(client=client)
        assert license_check._verify(license_value) is False
    _assert_no_license_in_logs(license_logs, license_value)


@pytest.mark.parametrize("expiration_date", ["2999-01-01", "2000-01-01", "private-invalid-expiry-marker"])
def test_local_license_logs_omit_signed_payload_and_validation_errors(
    license_logs: pytest.LogCaptureFixture,
    expiration_date: str,
) -> None:
    payload_marker: Final = "test-only-private-payload-marker"
    public_key, license_value = _signed_license(expiration_date, allowed_features=(payload_marker,))
    license_check: Final = LicenseCheck()
    result: Final = license_check.verify_license_without_api_request(public_key, license_value)
    assert (result is True) is (expiration_date == "2999-01-01")
    _assert_no_license_in_logs(license_logs, license_value)
    assert payload_marker not in license_logs.text
    assert expiration_date not in license_logs.text


def test_local_license_exception_does_not_log_license(
    license_logs: pytest.LogCaptureFixture,
) -> None:
    _public_key, license_value = _signed_license("2999-01-01")
    failing_public_key: Final = MagicMock()
    failing_public_key.verify.side_effect = ValueError(license_value)
    license_check: Final = LicenseCheck()
    assert license_check.verify_license_without_api_request(failing_public_key, license_value) is False
    assert "error_type=ValueError" in license_logs.text
    _assert_no_license_in_logs(license_logs, license_value)
