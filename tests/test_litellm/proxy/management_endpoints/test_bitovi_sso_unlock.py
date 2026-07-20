import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from litellm.proxy.management_endpoints.ui_sso import google_login


@pytest.mark.asyncio
async def test_google_login_allows_sso_when_billable_users_exceed_five():
    """
    Bitovi fork: non-premium SSO must not block login when billable users > 5.
    """
    mock_request = MagicMock(spec=Request)
    mock_request.url = "https://proxy.example.com/sso/key/generate"

    mock_prisma = MagicMock()

    async def _count(*args, where=None, **kwargs):
        return 0 if where is not None else 10

    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_usertable.count = _count

    billable_users_mock = AsyncMock(return_value=10)

    with (
        patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "test-client", "LITELLM_MASTER_KEY": "sk-test"}, clear=False),
        patch("litellm.proxy.proxy_server.premium_user", False),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.user_custom_ui_sso_sign_in_handler", None),
        patch(
            "litellm.repositories.user_repository.UserRepository.count_billable_users",
            billable_users_mock,
        ),
        patch(
            "litellm.proxy.management_endpoints.ui_sso.SSOAuthenticationHandler.should_use_sso_handler",
            return_value=True,
        ),
        patch(
            "litellm.proxy.management_endpoints.ui_sso.SSOAuthenticationHandler.get_sso_login_redirect",
            new_callable=AsyncMock,
            return_value=MagicMock(status_code=307),
        ),
        patch(
            "litellm.proxy.management_endpoints.ui_sso.show_missing_vars_in_env",
            return_value=None,
        ),
    ):
        response = await google_login(request=mock_request)

    assert response is not None
    billable_users_mock.assert_not_called()
