"""
What is this?
Regression tests for #38195: config-file-defined pass-through endpoints are
merged into the GET list but can never be managed by the DB-backed CRUD
handlers. DELETE/UPDATE must return a targeted error pointing the operator
back to config.yaml instead of a misleading "not found".
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import ConfigFieldInfo, UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    delete_pass_through_endpoints,
    update_pass_through_endpoints,
)

CONFIG_ENDPOINT = {
    "path": "/config-defined-endpoint",
    "target": "https://example.com/config-defined",
    "headers": {},
    "id": "config-endpoint-id",
}

DB_ENDPOINT = {
    "path": "/db-endpoint",
    "target": "https://example.com/db",
    "headers": {},
    "id": "db-endpoint-id",
}


def _user() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id="user-1")


def _db_getter():
    """get_config_general_settings stand-in that only sees the DB copy."""

    async def _get(field_name: str, user_api_key_dict=None):
        return ConfigFieldInfo(
            field_name=field_name,
            field_value=[dict(DB_ENDPOINT)],
        )

    return _get


@pytest.mark.asyncio
async def test_delete_config_defined_endpoint_returns_targeted_error():
    with (
        patch(
            "litellm.proxy.proxy_server.config_passthrough_endpoints",
            [dict(CONFIG_ENDPOINT)],
        ),
        patch(
            "litellm.proxy.proxy_server.get_config_general_settings",
            side_effect=_db_getter(),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await delete_pass_through_endpoints(
                endpoint_id="config-endpoint-id",
                user_api_key_dict=_user(),
            )

    assert exc_info.value.status_code == 400
    assert "defined in your config file" in str(exc_info.value.detail)
    assert "config.yaml" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_update_config_defined_endpoint_returns_targeted_error():
    from litellm.proxy._types import PassThroughGenericEndpoint

    with (
        patch(
            "litellm.proxy.proxy_server.config_passthrough_endpoints",
            [dict(CONFIG_ENDPOINT)],
        ),
        patch(
            "litellm.proxy.proxy_server.get_config_general_settings",
            side_effect=_db_getter(),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await update_pass_through_endpoints(
                endpoint_id="config-endpoint-id",
                data=PassThroughGenericEndpoint(
                    path="/config-defined-endpoint",
                    target="https://example.com/updated",
                ),
                request=None,
                user_api_key_dict=_user(),
            )

    assert exc_info.value.status_code == 400
    assert "defined in your config file" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_delete_unknown_endpoint_keeps_generic_not_found():
    with (
        patch(
            "litellm.proxy.proxy_server.config_passthrough_endpoints",
            [dict(CONFIG_ENDPOINT)],
        ),
        patch(
            "litellm.proxy.proxy_server.get_config_general_settings",
            side_effect=_db_getter(),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await delete_pass_through_endpoints(
                endpoint_id="does-not-exist",
                user_api_key_dict=_user(),
            )

    assert exc_info.value.status_code == 400
    assert "was not found in pass-through endpoint list" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_delete_db_endpoint_still_works():
    db_getter = _db_getter()

    with (
        patch(
            "litellm.proxy.proxy_server.config_passthrough_endpoints",
            [dict(CONFIG_ENDPOINT)],
        ),
        patch(
            "litellm.proxy.proxy_server.get_config_general_settings",
            side_effect=db_getter,
        ),
        patch(
            "litellm.proxy.proxy_server.update_config_general_settings",
            new_callable=AsyncMock,
        ) as mock_update,
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.remove_endpoint_routes"
        ),
    ):
        response = await delete_pass_through_endpoints(
            endpoint_id="db-endpoint-id",
            user_api_key_dict=_user(),
        )

    assert response.endpoints[0].id == "db-endpoint-id"
    saved_value = mock_update.call_args.kwargs["data"].field_value
    assert saved_value == []
