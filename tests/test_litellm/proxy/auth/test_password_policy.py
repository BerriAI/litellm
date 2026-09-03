"""
Tests for the configurable password-strength policy in
`litellm.proxy.auth.password_policy`, enforced on every path that persists a
new or changed password for a locally-managed user.

The breach-check (HIBP) tests inject a real AsyncHTTPHandler wrapping an
httpx.MockTransport, so no network is touched and nothing is monkeypatched.
"""

import hashlib

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.password_policy import (
    DEFAULT_MIN_LENGTH,
    MIN_ALLOWED_LENGTH,
    PasswordPolicy,
    get_password_policy,
    validate_password_not_breached,
    validate_password_policy,
)

STRONG_PASSWORD = "Str0ng!Passw0rd"


def _sha1_upper(password: str) -> str:
    return hashlib.sha1(password.encode("utf-8"), usedforsecurity=False).hexdigest().upper()


def _client_with_transport(handler) -> AsyncHTTPHandler:
    http_handler = AsyncHTTPHandler()
    http_handler.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return http_handler


def _client_never_called() -> AsyncHTTPHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected HTTP call to {request.url}")

    return _client_with_transport(handler)


def _client_returning(body: str, status_code: int = 200) -> AsyncHTTPHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=body)

    return _client_with_transport(handler)


def test_get_password_policy_defaults_to_pif_baseline():
    policy = get_password_policy({})
    assert policy == PasswordPolicy(
        min_length=DEFAULT_MIN_LENGTH,
        require_uppercase=True,
        require_lowercase=True,
        require_numbers=True,
        require_special_characters=True,
    )


def test_get_password_policy_reads_overrides_from_general_settings():
    policy = get_password_policy(
        {
            "password_policy_min_length": 20,
            "password_policy_require_uppercase": False,
            "password_policy_require_lowercase": False,
            "password_policy_require_numbers": False,
            "password_policy_require_special_characters": False,
        }
    )
    assert policy == PasswordPolicy(
        min_length=20,
        require_uppercase=False,
        require_lowercase=False,
        require_numbers=False,
        require_special_characters=False,
    )


def test_validate_password_policy_accepts_strong_password():
    assert validate_password_policy(STRONG_PASSWORD, {}) is None


@pytest.mark.parametrize(
    "password,expected_fragment",
    [
        ("Sh0rt!Pw", "12 characters"),
        ("weakpassword123!", "uppercase"),
        ("WEAKPASSWORD123!", "lowercase"),
        ("WeakPassword!!!!", "number"),
        ("WeakPassword12345", "special character"),
    ],
)
def test_validate_password_policy_rejects_each_missing_class(password, expected_fragment):
    with pytest.raises(ProxyException) as exc_info:
        validate_password_policy(password, {})
    assert exc_info.value.code == "400"
    assert exc_info.value.type == ProxyErrorTypes.validation_error
    assert exc_info.value.param == "password"
    assert expected_fragment in exc_info.value.message


def test_validate_password_policy_reports_every_violation_at_once():
    with pytest.raises(ProxyException) as exc_info:
        validate_password_policy("weak", {})
    assert "12 characters" in exc_info.value.message
    assert "uppercase" in exc_info.value.message
    assert "number" in exc_info.value.message
    assert "special character" in exc_info.value.message


def test_validate_password_policy_honors_relaxed_config():
    general_settings = {
        "password_policy_min_length": MIN_ALLOWED_LENGTH,
        "password_policy_require_special_characters": False,
    }
    # 8 chars, has upper/lower/number, no special char: fails default policy,
    # passes the relaxed one above.
    validate_password_policy("Abcd1234", general_settings)
    with pytest.raises(ProxyException):
        validate_password_policy("Abcd1234", {})


def test_validate_password_policy_honors_stricter_min_length():
    general_settings = {"password_policy_min_length": 20}
    with pytest.raises(ProxyException) as exc_info:
        validate_password_policy(STRONG_PASSWORD, general_settings)
    assert "20 characters" in exc_info.value.message


@pytest.mark.parametrize("configured_min_length", [0, -1, -100, 1, 7])
def test_get_password_policy_floors_nonpositive_or_too_low_min_length(configured_min_length):
    """A misconfigured min_length must never disable the length check
    entirely: it floors at MIN_ALLOWED_LENGTH instead of passing through."""
    policy = get_password_policy({"password_policy_min_length": configured_min_length})
    assert policy.min_length == MIN_ALLOWED_LENGTH


def test_validate_password_policy_rejects_short_password_even_with_zero_min_length_configured():
    general_settings = {"password_policy_min_length": 0}
    with pytest.raises(ProxyException) as exc_info:
        validate_password_policy("a", general_settings)
    assert f"{MIN_ALLOWED_LENGTH} characters" in exc_info.value.message


def test_get_password_policy_ignores_boolean_min_length():
    """`bool` is a subclass of `int` in Python; a stray `true`/`false` value
    must not silently coerce into a min_length of 1 or 0."""
    policy = get_password_policy({"password_policy_min_length": False})
    assert policy.min_length == DEFAULT_MIN_LENGTH


def test_validate_password_policy_rejects_unicode_letter_as_special_character():
    """Regression: an ASCII-only `[^A-Za-z0-9]` check would miscount an
    accented letter as the required special character, so a letters-and-
    digits-only password like this one (no real symbol) must still be
    rejected."""
    with pytest.raises(ProxyException) as exc_info:
        validate_password_policy("Passwörd1234", {})
    assert "special character" in exc_info.value.message


def test_validate_password_policy_accepts_real_special_character_with_unicode_letters():
    """Same base password as the rejection test above, plus an actual symbol."""
    assert validate_password_policy("Passwörd1234!", {}) is None


@pytest.mark.asyncio
async def test_breach_check_skipped_when_disabled():
    result = await validate_password_not_breached(
        password="password12345",  # breached in reality, but the check is off
        general_settings={"password_policy_check_breached_passwords": False},
        client=_client_never_called(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_rejects_breached_password():
    password = "correct horse battery staple"
    sha1 = _sha1_upper(password)
    body = f"AAAA000000000000000000000000000000A:0\r\n{sha1[5:]}:42\r\nBBBB000000000000000000000000000000B:7"

    with pytest.raises(ProxyException) as exc_info:
        await validate_password_not_breached(password=password, general_settings={}, client=_client_returning(body))
    assert exc_info.value.code == "400"
    assert exc_info.value.type == ProxyErrorTypes.validation_error
    assert exc_info.value.param == "password"
    assert "data breaches" in exc_info.value.message


@pytest.mark.asyncio
async def test_only_sha1_prefix_leaves_the_proxy():
    password = "a very secret password"
    sha1 = _sha1_upper(password)
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, text="0000000000000000000000000000000000A:1")

    result = await validate_password_not_breached(
        password=password, general_settings={}, client=_client_with_transport(handler)
    )
    assert result is None

    (request,) = captured_requests
    assert request.url.path == f"/range/{sha1[:5]}"
    assert sha1[5:] not in str(request.url)
    assert request.headers["Add-Padding"] == "true"
    assert "litellm" in request.headers["User-Agent"]


@pytest.mark.asyncio
async def test_ignores_padding_entries_with_zero_count():
    """HIBP padding entries (requested via Add-Padding) carry count 0 and must
    not be treated as breaches when they collide with the password's suffix."""
    password = "a padded-away password"
    sha1 = _sha1_upper(password)

    result = await validate_password_not_breached(
        password=password, general_settings={}, client=_client_returning(f"{sha1[5:]}:0")
    )
    assert result is None


@pytest.mark.asyncio
async def test_accepts_password_absent_from_breach_corpus():
    result = await validate_password_not_breached(
        password="a genuinely novel password",
        general_settings={},
        client=_client_returning("0018A45C4D1DEF81644B54AB7F969B88D65:1\r\n00D4F6E8FA6EECAD2A3AA415EEC418D38EC:2"),
    )
    assert result is None


@pytest.mark.asyncio
async def test_breach_check_fails_open_on_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    result = await validate_password_not_breached(
        password="password12345",  # breached, but HIBP is unreachable
        general_settings={},
        client=_client_with_transport(handler),
    )
    assert result is None


@pytest.mark.asyncio
async def test_breach_check_fails_open_on_http_error_status():
    result = await validate_password_not_breached(
        password="password12345",
        general_settings={},
        client=_client_returning("service unavailable", status_code=503),
    )
    assert result is None


@pytest.mark.asyncio
async def test_breach_check_fails_open_on_malformed_response_body():
    result = await validate_password_not_breached(
        password="password12345",
        general_settings={},
        client=_client_returning(f"{_sha1_upper('password12345')[5:]}:not-a-number"),
    )
    assert result is None
