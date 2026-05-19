"""Tests for app_id propagation through auth (S4-04)."""

from unittest.mock import patch

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def test_user_api_key_auth_carries_app_id_attribute():
    """The UserAPIKeyAuth dataclass exposes app_id via inheritance."""
    auth = UserAPIKeyAuth(
        api_key="sk-x",
        user_id="u-1",
        user_role=LitellmUserRoles.INTERNAL_USER,
        app_id="xct-chat",
    )
    assert auth.app_id == "xct-chat"


def test_user_api_key_auth_token_type_field_default_none():
    """token_type is None for legacy keys (S4-02 column nullable)."""
    auth = UserAPIKeyAuth(
        api_key="sk-x",
        user_id="u-1",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    assert auth.token_type is None
    assert auth.app_id is None


@pytest.mark.asyncio
async def test_header_app_id_fills_in_when_token_app_id_absent():
    """x-xct-app-id header sets app_id when the token row didn't carry one."""
    from fastapi import Request

    from litellm.proxy.auth import user_api_key_auth as uak_mod

    auth_obj = UserAPIKeyAuth(
        api_key="sk-x",
        user_id="u-1",
        user_role=LitellmUserRoles.INTERNAL_USER,
        app_id=None,  # not on the token
    )

    async def fake_builder(**kwargs):
        return auth_obj

    async def fake_checks(**kwargs):
        return None

    # Minimal Request — we only read request.headers and request_data.
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"x-xct-app-id", b"xct-chat")],
        "query_string": b"",
    }
    request = Request(scope)

    with (
        patch.object(uak_mod, "_user_api_key_auth_builder", fake_builder),
        patch.object(uak_mod, "_run_centralized_common_checks", fake_checks),
        patch.object(uak_mod, "_read_request_body", lambda request: _async_return({})),
        patch.object(
            uak_mod,
            "populate_request_with_path_params",
            lambda request_data, request: request_data,
        ),
        patch.object(
            uak_mod, "get_request_route", lambda request: "/v1/chat/completions"
        ),
        patch.object(
            uak_mod.RouteChecks, "should_call_route", lambda route, valid_token: None
        ),
    ):
        result = await uak_mod.user_api_key_auth(
            request=request,
            api_key="sk-x",
            azure_api_key_header=None,
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            custom_litellm_key_header=None,
        )
    assert result.app_id == "xct-chat"


@pytest.mark.asyncio
async def test_token_app_id_wins_over_header():
    """Header MUST NOT override an app_id that's baked into the token row.

    Otherwise a leaked admin key could impersonate any app's traffic by
    setting x-xct-app-id on every request.
    """
    from fastapi import Request

    from litellm.proxy.auth import user_api_key_auth as uak_mod

    auth_obj = UserAPIKeyAuth(
        api_key="sk-x",
        user_id="u-1",
        user_role=LitellmUserRoles.INTERNAL_USER,
        app_id="xct-home",  # baked into the token
    )

    async def fake_builder(**kwargs):
        return auth_obj

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"x-xct-app-id", b"xct-chat")],  # attacker tries to override
        "query_string": b"",
    }
    request = Request(scope)

    with (
        patch.object(uak_mod, "_user_api_key_auth_builder", fake_builder),
        patch.object(
            uak_mod, "_run_centralized_common_checks", lambda **kw: _async_return(None)
        ),
        patch.object(uak_mod, "_read_request_body", lambda request: _async_return({})),
        patch.object(
            uak_mod,
            "populate_request_with_path_params",
            lambda request_data, request: request_data,
        ),
        patch.object(
            uak_mod, "get_request_route", lambda request: "/v1/chat/completions"
        ),
        patch.object(
            uak_mod.RouteChecks, "should_call_route", lambda route, valid_token: None
        ),
    ):
        result = await uak_mod.user_api_key_auth(
            request=request,
            api_key="sk-x",
            azure_api_key_header=None,
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            custom_litellm_key_header=None,
        )
    assert result.app_id == "xct-home"  # baked value preserved


async def _async_return(value):
    return value
