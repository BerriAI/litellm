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


@pytest.mark.asyncio
async def test_enterprise_custom_auth_returns_string():
    from litellm.proxy._types import hash_token

    # Test when enterprise_custom_auth returns a string (LiteLLM virtual key)
    mock_user_auth = AsyncMock(return_value="sk-test-key")
    request = MagicMock(spec=Request)

    with patch(
        "litellm.proxy.auth.user_api_key_auth.enterprise_custom_auth", mock_user_auth
    ), patch("litellm.proxy.proxy_server.master_key", "sk-1234"), patch(
        "litellm.proxy.proxy_server.prisma_client", MagicMock()
    ):
        # Verify the key is correctly handled in _user_api_key_auth_builder
        with patch(
            "litellm.proxy.auth.user_api_key_auth.get_key_object"
        ) as mock_get_key_object:
            mock_get_key_object.return_value = MagicMock(
                token="sk-test-key",
                user_role="internal_user",
                team_id=None,
                user_id="test-user",
            )

            # Call _user_api_key_auth_builder with the returned key
            from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

            try:
                auth_obj = await _user_api_key_auth_builder(
                    request=request,
                    api_key="my-custom-key",
                    azure_api_key_header="",
                    anthropic_api_key_header=None,
                    google_ai_studio_api_key_header=None,
                    azure_apim_header=None,
                    request_data={},
                    custom_litellm_key_header=None,
                )
            except Exception as e:
                print("error:", e)

            # Verify get_key_object was called with the correct key
            mock_get_key_object.assert_called_once()
            # The key should be hashed before being passed to get_key_object
            assert mock_get_key_object.call_args[1]["hashed_token"] == hash_token(
                "sk-test-key"
            )
