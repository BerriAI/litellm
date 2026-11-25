"""DISABLE_ADMIN_UI must be enforced on the routes that serve and authenticate the admin UI.

The flag already gates the SSO paths in ``litellm/proxy/management_endpoints/ui_sso.py``.
These tests cover the two routes in ``proxy_server`` that reach the same UI session:
``POST /login``, which mints the UI JWT, and ``GET /fallback/login``, which redirects into
the login page and sets the SSO fallback cookie.
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
