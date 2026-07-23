"""Tests for the single choke point that signs the UI session cookie."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
import pytest

from litellm.constants import LITELLM_UI_SESSION_DURATION
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy.auth.ui_session_jwt import encode_ui_session_jwt, ui_session_expires_at
from litellm.types.proxy.ui_sso import ReturnedUITokenObject

MASTER_KEY = "sk-master-for-tests"


def _token_object(**overrides) -> ReturnedUITokenObject:
    base = ReturnedUITokenObject(
        user_id="user-1",
        key="sk-inner-key",
        user_email="user@example.com",
        user_role="proxy_admin",
        login_method="sso",
        premium_user=False,
        auth_header_name="Authorization",
        disabled_non_admin_personal_key_creation=False,
        server_root_path="/",
    )
    base.update(overrides)  # type: ignore[typeddict-item]  # test helper takes arbitrary claim overrides
    return base


def test_encode_stamps_a_bounded_expiry():
    """The cookie expires LITELLM_UI_SESSION_DURATION after it is minted."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    decoded = jwt.decode(
        encode_ui_session_jwt(_token_object(), MASTER_KEY, now=now),
        MASTER_KEY,
        algorithms=["HS256"],
        # The claim's VALUE is what is under test, so expiry enforcement is off; a fixed mint
        # instant keeps the assertion independent of when the suite runs.
        options={"require": ["exp"], "verify_exp": False},
    )

    assert decoded["exp"] == int((now + timedelta(seconds=duration_in_seconds(LITELLM_UI_SESSION_DURATION))).timestamp())


def test_encode_preserves_every_payload_claim():
    """Stamping exp must not drop or rewrite anything the dashboard reads out of the cookie."""
    token_object = _token_object()

    decoded = jwt.decode(
        encode_ui_session_jwt(token_object, MASTER_KEY),
        MASTER_KEY,
        algorithms=["HS256"],
        options={"require": ["exp"]},
    )

    assert {claim: decoded[claim] for claim in token_object} == dict(token_object)


def test_a_cookie_past_its_expiry_is_rejected():
    long_ago = datetime.now(timezone.utc) - timedelta(seconds=duration_in_seconds(LITELLM_UI_SESSION_DURATION) + 60)
    expired = encode_ui_session_jwt(_token_object(), MASTER_KEY, now=long_ago)

    with pytest.raises(jwt.ExpiredSignatureError):
        jwt.decode(expired, MASTER_KEY, algorithms=["HS256"])


def test_a_cookie_signed_with_another_key_is_rejected():
    foreign = encode_ui_session_jwt(_token_object(), "sk-some-other-master-key")

    with pytest.raises(jwt.InvalidSignatureError):
        jwt.decode(foreign, MASTER_KEY, algorithms=["HS256"])


def test_ui_session_expires_at_is_the_configured_window():
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)

    assert ui_session_expires_at(now) - now == timedelta(seconds=duration_in_seconds(LITELLM_UI_SESSION_DURATION))


def test_minted_cookie_is_accepted_by_the_session_cookie_reader(monkeypatch):
    """The regression this module exists for.

    ``_user_id_from_session_cookie`` requires an ``exp`` claim, so before the mints stamped one it
    rejected every genuine login cookie and the MCP SSO interpose was an unconditional
    redirect-to-login loop. A cookie in the shape the mints previously produced must still be
    rejected, and one from the choke point must resolve the user.
    """
    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import _user_id_from_session_cookie

    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", MASTER_KEY, raising=False)

    def _request_with_cookie(cookie_value: str):
        from fastapi import Request

        return Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [(b"cookie", f"token={cookie_value}".encode())],
            }
        )

    token_object = _token_object()
    without_exp = jwt.encode(dict(token_object), MASTER_KEY, algorithm="HS256")
    with_exp = encode_ui_session_jwt(token_object, MASTER_KEY)

    assert _user_id_from_session_cookie(_request_with_cookie(without_exp)) is None
    assert _user_id_from_session_cookie(_request_with_cookie(with_exp)) == "user-1"


def test_no_mint_site_signs_a_ui_token_object_outside_the_choke_point():
    """Pins the 'every UI session cookie goes through encode_ui_session_jwt' claim.

    A new endpoint that mints a session cookie with a bare ``jwt.encode`` would silently
    reintroduce the unbounded-lifetime hole, and no behavioral test would catch it because the
    cookie it produces still works everywhere except the readers that require ``exp``.
    """
    proxy_root = Path(__file__).resolve().parents[4] / "litellm" / "proxy"
    offenders = [
        path
        for path in proxy_root.rglob("*.py")
        if path.name != "ui_session_jwt.py" and "returned_ui_token_object" in path.read_text() and _signs_it(path)
    ]

    assert offenders == [], f"these sign a UI token object directly instead of using encode_ui_session_jwt: {offenders}"


def _signs_it(path: Path) -> bool:
    source = path.read_text()
    return any(
        "returned_ui_token_object" in source[match : match + 200]
        for match in (i for i in range(len(source)) if source.startswith("jwt.encode(", i))
    )
