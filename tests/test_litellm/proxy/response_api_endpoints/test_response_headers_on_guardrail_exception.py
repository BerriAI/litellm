"""
Test that post_call_response_headers_hook is called on ModifyResponseException
in the /responses endpoint, so custom headers appear even on guardrail failures.
"""

import os
import sys
import pytest
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth


class GuardrailHeaderLogger(CustomLogger):
    """Logger that injects headers — used to verify hook fires on guardrail path."""

    async def async_post_call_response_headers_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, str]]:
        return {"x-guardrail-header": "injected"}


@pytest.mark.asyncio
async def test_modify_response_exception_calls_response_headers_hook():
    """
    When a guardrail raises ModifyResponseException on /responses,
    the response should still include custom headers from the hook.
    """
    from litellm.integrations.custom_guardrail import ModifyResponseException
    from litellm.proxy.proxy_server import app
    from fastapi.testclient import TestClient

    guardrail_logger = GuardrailHeaderLogger()

    with patch("litellm.callbacks", [guardrail_logger]):
        with patch("litellm.proxy.proxy_server.user_api_key_auth") as mock_auth:
            mock_auth.return_value = MagicMock(
                token="test_token",
                user_id="test_user",
                team_id=None,
            )

            # Make base_process_llm_request raise ModifyResponseException
            with patch(
                "litellm.proxy.response_api_endpoints.endpoints.ProxyBaseLLMRequestProcessing"
            ) as MockProcessor:
                mock_instance = MockProcessor.return_value
                mock_instance.base_process_llm_request = AsyncMock(
                    side_effect=ModifyResponseException(
                        message="Content blocked by guardrail",
                        model="gpt-4o",
                        request_data={"model": "gpt-4o"},
                    )
                )

                client = TestClient(app)
                response = client.post(
                    "/v1/responses",
                    json={"model": "gpt-4o", "input": "blocked content"},
                    headers={"Authorization": "Bearer sk-1234"},
                )

                assert response.status_code == 200
                assert response.headers.get("x-guardrail-header") == "injected"
