"""
Tests for get_available_models_for_user.

Covers the proxy_admin bypass introduced to fix LIT-3038:
  When an internal_user is promoted to proxy_admin their user record may still
  carry models=["no-default-models"], which the /models endpoint used to return
  verbatim.  Proxy admins should always see all proxy models regardless of any
  key/team model restriction inherited from a previous role.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def _make_key_dict(role: LitellmUserRoles, models: list) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="test-key",
        models=models,
        team_models=[],
        user_role=role,
    )


def _make_router(model_names: list):
    """Return a minimal mock Router with get_model_names / get_model_access_groups."""
    router = MagicMock()
    router.get_model_names.return_value = model_names
    router.get_model_access_groups.return_value = {}
    return router


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestGetAvailableModelsForUser:
    """Unit tests for get_available_models_for_user."""

    def test_proxy_admin_bypasses_no_default_models(self):
        """
        A proxy_admin whose key still carries no-default-models (inherited from a
        previous internal_user role) must receive the full proxy model list, not an
        empty set.
        """
        from litellm.proxy.utils import get_available_models_for_user

        proxy_models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
        key_dict = _make_key_dict(
            role=LitellmUserRoles.PROXY_ADMIN,
            models=["no-default-models"],
        )

        result = _run(
            get_available_models_for_user(
                user_api_key_dict=key_dict,
                llm_router=_make_router(proxy_models),
                general_settings={},
                user_model=None,
            )
        )

        assert set(proxy_models).issubset(set(result)), (
            "Proxy admin should see all proxy models; got: %s" % result
        )
        assert (
            "no-default-models" not in result
        ), "no-default-models must not appear in the proxy_admin result"

    def test_proxy_admin_with_empty_models_sees_all(self):
        """Proxy admin with empty models list also gets the full proxy model list."""
        from litellm.proxy.utils import get_available_models_for_user

        proxy_models = ["model-a", "model-b"]
        key_dict = _make_key_dict(
            role=LitellmUserRoles.PROXY_ADMIN,
            models=[],
        )

        result = _run(
            get_available_models_for_user(
                user_api_key_dict=key_dict,
                llm_router=_make_router(proxy_models),
                general_settings={},
                user_model=None,
            )
        )

        assert set(proxy_models).issubset(set(result))

    def test_internal_user_no_default_models_returns_marker(self):
        """
        An internal_user with no-default-models in their key should still receive
        no-default-models in the response so the UI can enforce team selection.
        """
        from litellm.proxy.utils import get_available_models_for_user

        proxy_models = ["gpt-4"]
        key_dict = _make_key_dict(
            role=LitellmUserRoles.INTERNAL_USER,
            models=["no-default-models"],
        )

        result = _run(
            get_available_models_for_user(
                user_api_key_dict=key_dict,
                llm_router=_make_router(proxy_models),
                general_settings={},
                user_model=None,
            )
        )

        assert "no-default-models" in result, (
            "internal_user with no-default-models should still see that marker; "
            "got: %s" % result
        )

    def test_internal_user_with_explicit_models(self):
        """
        An internal_user with an explicit model list gets only those models.
        Proxy models that aren't in the list should not appear.
        """
        from litellm.proxy.utils import get_available_models_for_user

        allowed = ["gpt-4"]
        proxy_models = ["gpt-4", "gpt-3.5-turbo"]
        key_dict = _make_key_dict(
            role=LitellmUserRoles.INTERNAL_USER,
            models=allowed,
        )

        result = _run(
            get_available_models_for_user(
                user_api_key_dict=key_dict,
                llm_router=_make_router(proxy_models),
                general_settings={},
                user_model=None,
            )
        )

        assert "gpt-4" in result
        # gpt-3.5-turbo should NOT be in the result because the key restricts to gpt-4
        assert "gpt-3.5-turbo" not in result

    def test_proxy_admin_no_router_returns_empty(self):
        """When there is no router (no models configured), proxy_admin gets empty list."""
        from litellm.proxy.utils import get_available_models_for_user

        key_dict = _make_key_dict(
            role=LitellmUserRoles.PROXY_ADMIN,
            models=["no-default-models"],
        )

        result = _run(
            get_available_models_for_user(
                user_api_key_dict=key_dict,
                llm_router=None,
                general_settings={},
                user_model=None,
            )
        )

        assert result == [], "No router => no models, even for proxy_admin"
        assert "no-default-models" not in result
