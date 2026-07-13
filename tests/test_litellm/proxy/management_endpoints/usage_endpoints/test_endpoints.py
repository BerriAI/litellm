"""Endpoint-boundary tests for /usage/ai/chat.

Security regression: a non-admin caller with user_id=None (a service-account
key) must be rejected at the endpoint before any scope/provider is built, so it
can never fall through to an unscoped global query.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.usage_endpoints.endpoints import (
    ChatMessage,
    UsageAIChatRequest,
    usage_ai_chat,
)


class TestServiceAccountGuard:
    @pytest.mark.asyncio
    async def test_non_admin_with_user_id_none_is_rejected(self):
        service_account_key = UserAPIKeyAuth(user_id=None, user_role=LitellmUserRoles.INTERNAL_USER)
        body = UsageAIChatRequest(messages=[ChatMessage(role="user", content="hi")], model="m")

        with pytest.raises(HTTPException) as exc_info:
            await usage_ai_chat(data=body, request=MagicMock(), user_api_key_dict=service_account_key)

        assert exc_info.value.status_code == 403
        assert "Service-account keys" in str(exc_info.value.detail)


class TestScopeSelection:
    @pytest.mark.asyncio
    async def test_admin_caller_builds_admin_scope(self):
        admin_key = UserAPIKeyAuth(user_id="admin-1", user_role=LitellmUserRoles.PROXY_ADMIN)
        body = UsageAIChatRequest(messages=[ChatMessage(role="user", content="hi")], model="m")

        captured = {}

        async def _fake_stream(*, provider, messages, model):
            captured["is_admin"] = provider.is_admin
            captured["messages"] = messages
            if False:
                yield ""  # pragma: no cover

        # Inject a fake agent stream + prisma via patching the lazily imported names.
        import litellm.proxy.proxy_server as proxy_server

        original_prisma = getattr(proxy_server, "prisma_client", None)
        proxy_server.prisma_client = MagicMock()
        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(
                    "litellm.proxy.management_endpoints.usage_endpoints.agent.stream_usage_ai_chat",
                    _fake_stream,
                )
                response = await usage_ai_chat(data=body, request=MagicMock(), user_api_key_dict=admin_key)
                # Drain the streaming body so the generator runs.
                async for _ in response.body_iterator:
                    pass
        finally:
            proxy_server.prisma_client = original_prisma

        assert captured["is_admin"] is True
        assert captured["messages"] == [{"role": "user", "content": "hi"}]
