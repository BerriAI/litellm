"""LIT-3038 regression: promoted proxy admins should not be locked out of
the proxy model list when their user/key record still carries the
``models=["no-default-models"]`` restriction inherited from a prior
internal_user role.

These tests exercise ``litellm.proxy.utils.get_available_models_for_user``
directly without spinning up the FastAPI app, by constructing the same
``UserAPIKeyAuth`` shape that the auth layer produces in the buggy
scenario.
"""
from unittest.mock import MagicMock

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.utils import get_available_models_for_user


def _fake_router(model_names):
    r = MagicMock()
    r.get_model_names.return_value = list(model_names)
    r.get_model_access_groups.return_value = {}
    r.get_model_list.side_effect = lambda model_name=None: None
    return r


@pytest.mark.asyncio
async def test_promoted_proxy_admin_with_no_default_models_sees_proxy_model_list():
    """A user promoted to PROXY_ADMIN whose stale key/user record still
    carries ``models=["no-default-models"]`` must see the full proxy model
    list (the previous behaviour returned the literal placeholder string)."""
    user = UserAPIKeyAuth(
        api_key="sk-stale",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        models=["no-default-models"],
        team_models=[],
    )
    router = _fake_router(["gpt-3.5-turbo", "gpt-4"])

    models = await get_available_models_for_user(
        user_api_key_dict=user,
        llm_router=router,
        general_settings={},
        user_model=None,
    )

    assert "no-default-models" not in models, (
        "PROXY_ADMIN bypass should hide the placeholder; got "
        + repr(models)
    )
    assert set(models) == {"gpt-3.5-turbo", "gpt-4"}


@pytest.mark.asyncio
async def test_promoted_proxy_admin_with_empty_key_models_still_sees_proxy_models():
    user = UserAPIKeyAuth(
        api_key="sk-clean",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        models=[],
        team_models=[],
    )
    router = _fake_router(["gpt-3.5-turbo", "gpt-4"])

    models = await get_available_models_for_user(
        user_api_key_dict=user,
        llm_router=router,
        general_settings={},
        user_model=None,
    )

    assert set(models) == {"gpt-3.5-turbo", "gpt-4"}


@pytest.mark.asyncio
async def test_internal_user_with_no_default_models_remains_blocked():
    """The fix is scoped to PROXY_ADMIN; internal users (and viewer roles)
    must still be filtered by their key/user model restriction."""
    user = UserAPIKeyAuth(
        api_key="sk-intern",
        user_role=LitellmUserRoles.INTERNAL_USER,
        models=["no-default-models"],
        team_models=[],
    )
    router = _fake_router(["gpt-3.5-turbo", "gpt-4"])

    models = await get_available_models_for_user(
        user_api_key_dict=user,
        llm_router=router,
        general_settings={},
        user_model=None,
    )

    # before-the-fix behaviour: the placeholder string flows through; the
    # fix must NOT change this for non-admin roles.
    assert "no-default-models" in models
    assert "gpt-3.5-turbo" not in models


@pytest.mark.asyncio
async def test_proxy_admin_viewer_with_no_default_models_remains_blocked():
    """``proxy_admin_viewer`` is a read-only role and is NOT the same as
    PROXY_ADMIN; ensure the bypass does not accidentally cover it."""
    user = UserAPIKeyAuth(
        api_key="sk-viewer",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        models=["no-default-models"],
        team_models=[],
    )
    router = _fake_router(["gpt-3.5-turbo", "gpt-4"])

    models = await get_available_models_for_user(
        user_api_key_dict=user,
        llm_router=router,
        general_settings={},
        user_model=None,
    )

    assert "no-default-models" in models
    assert "gpt-3.5-turbo" not in models


@pytest.mark.asyncio
async def test_proxy_admin_with_explicit_team_id_uses_team_path():
    """Passing an explicit team_id should still go through the team validation
    path (not the admin bypass) so that the /v1/models?team_id=... endpoint
    surfaces team-scoped models even for an admin. The bypass guards on
    ``team_id`` being falsy."""
    user = UserAPIKeyAuth(
        api_key="sk-admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        models=["no-default-models"],
        team_models=[],
    )
    router = _fake_router(["gpt-3.5-turbo", "gpt-4"])

    # Without prisma_client/proxy_logging_obj/user_api_key_cache the team_id
    # branch inside the function is a no-op, so the path falls through to the
    # legacy key/team filter — and we should see the placeholder. Crucially,
    # we should NOT see the admin bypass kick in here.
    models = await get_available_models_for_user(
        user_api_key_dict=user,
        llm_router=router,
        general_settings={},
        user_model=None,
        team_id="t-1",
    )

    assert "no-default-models" in models
