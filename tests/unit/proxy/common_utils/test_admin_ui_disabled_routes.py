"""DISABLE_ADMIN_UI must stop an Admin UI SESSION being minted, by any route.

The flag already gates the SSO paths in ``litellm/proxy/management_endpoints/ui_sso.py``.
The security property is that no Admin UI session can be created while the switch is on, not
merely that the usual login page is unreachable, so the authoritative check lives in
``encode_ui_session_jwt`` — the choke point its own docstring says every UI login path uses
(``/login``, ``/v2/login``, ``/v3/login``, ``/v3/login/exchange`` and the SSO callback).
The two route-level guards remain for the friendlier HTML response on the paths a human hits.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_login_blocked_when_admin_ui_disabled():
    """POST /login must not mint a UI session while the admin UI is disabled."""
    from litellm.proxy.proxy_server import login

    with patch.dict(os.environ, {"DISABLE_ADMIN_UI": "true"}):
        response = await login(SimpleNamespace())

    assert response.status_code == 200
    assert "Admin UI is Disabled" in response.body.decode()


@pytest.mark.asyncio
async def test_fallback_login_blocked_when_admin_ui_disabled():
    """GET /fallback/login must not redirect into the UI while the admin UI is disabled."""
    from litellm.proxy.proxy_server import fallback_login

    with patch.dict(os.environ, {"DISABLE_ADMIN_UI": "true"}):
        response = await fallback_login(SimpleNamespace())

    assert response.status_code == 200
    assert "Admin UI is Disabled" in response.body.decode()


@pytest.mark.asyncio
async def test_login_not_blocked_when_admin_ui_enabled():
    """Control: with the flag off the guard must not fire, so the handler proceeds past it and
    fails on the empty request object instead of returning the disabled page."""
    from litellm.proxy.proxy_server import login

    with patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}):
        with pytest.raises(AttributeError):
            await login(SimpleNamespace())


def test_disabled_page_is_the_flag_helper_s_own_page():
    """The page the guards return is the one the flag's own helper builds."""
    from litellm.proxy.common_utils.admin_ui_utils import admin_ui_disabled, is_admin_ui_disabled

    assert "Admin UI is Disabled" in admin_ui_disabled().body.decode()

    with patch.dict(os.environ, {"DISABLE_ADMIN_UI": "true"}):
        assert is_admin_ui_disabled() is True
    with patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}):
        assert is_admin_ui_disabled() is False


def test_session_jwt_refused_when_admin_ui_disabled():
    """The choke point: no route may mint a UI session while the flag is set."""
    from fastapi import HTTPException

    from litellm.proxy.auth.login_utils import encode_ui_session_jwt

    with patch.dict(os.environ, {"DISABLE_ADMIN_UI": "true"}):
        with pytest.raises(HTTPException) as excinfo:
            encode_ui_session_jwt({"user_id": "u", "key": "sk-x"}, "master")

    assert excinfo.value.status_code == 401


def test_session_jwt_minted_when_admin_ui_enabled():
    """Control: with the flag off the choke point still mints a token."""
    from litellm.proxy.auth.login_utils import encode_ui_session_jwt

    with patch.dict(os.environ, {"DISABLE_ADMIN_UI": "false"}):
        token = encode_ui_session_jwt({"user_id": "u", "key": "sk-x"}, "master")

    assert isinstance(token, str) and token.count(".") == 2


@pytest.mark.asyncio
async def test_onboarding_blocked_when_admin_ui_disabled():
    """Onboarding mints a UI session with jwt.encode directly, so the issuance choke point
    never sees it and the route needs its own guard."""
    from litellm.proxy.proxy_server import onboarding

    with patch.dict(os.environ, {"DISABLE_ADMIN_UI": "true"}):
        response = await onboarding(invite_link="abc", request=SimpleNamespace())

    assert response.status_code == 200
    assert "Admin UI is Disabled" in response.body.decode()
