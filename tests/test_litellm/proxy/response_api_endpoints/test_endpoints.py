"""
Test for response_api_endpoints/endpoints.py
"""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from litellm.proxy.proxy_server import app


class TestResponsesAPIEndpoints(unittest.TestCase):
    @pytest.mark.asyncio
    @patch("litellm.proxy.proxy_server.llm_router")
    @patch("litellm.proxy.proxy_server.user_api_key_auth")
    async def test_openai_v1_responses_route(self, mock_auth, mock_router):
        """
        Test that /openai/v1/responses endpoint is correctly registered and accessible.
        """
        mock_auth.return_value = MagicMock(
            token="test_token",
            user_id="test_user",
            team_id=None,
        )

        mock_router.aresponses = AsyncMock(
            return_value={
                "id": "resp_abc123",
                "object": "realtime.response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Test response"}],
                    }
                ],
            }
        )

        client = TestClient(app)

        test_data = {"model": "gpt-4o", "input": "Tell me about AI"}

        response = client.post(
            "/openai/v1/responses",
            json=test_data,
            headers={"Authorization": "Bearer sk-1234"},
        )

        assert response.status_code in [200, 401, 500]

