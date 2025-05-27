from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from litellm_enterprise.proxy.auth.user_api_key_auth import enterprise_custom_auth


@pytest.mark.asyncio
async def test_enterprise_custom_auth_none_user_auth():
    # Test when user_custom_auth is None
    request = MagicMock(spec=Request)
    result = await enterprise_custom_auth(request, "test-api-key", None)
    assert result is None


@pytest.mark.asyncio
async def test_enterprise_custom_auth_mode_on():
    # Test when mode is "on"
    mock_user_auth = AsyncMock(return_value={"user_id": "test-user"})
    request = MagicMock(spec=Request)

    with patch(
        "litellm_enterprise.proxy.proxy_server.custom_auth_settings", {"mode": "on"}
    ):
        result = await enterprise_custom_auth(request, "test-api-key", mock_user_auth)
        assert result == {"user_id": "test-user"}
        mock_user_auth.assert_called_once_with(request, "test-api-key")


@pytest.mark.asyncio
async def test_enterprise_custom_auth_mode_auto_with_error():
    # Test when mode is "auto" and user_auth raises an exception
    mock_user_auth = AsyncMock(side_effect=Exception("Auth failed"))
    request = MagicMock(spec=Request)

    with patch(
        "litellm_enterprise.proxy.proxy_server.custom_auth_settings", {"mode": "auto"}
    ):
        result = await enterprise_custom_auth(request, "test-api-key", mock_user_auth)
        assert result is None
        mock_user_auth.assert_called_once_with(request, "test-api-key")
