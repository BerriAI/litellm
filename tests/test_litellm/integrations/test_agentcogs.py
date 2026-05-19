import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.agentcogs import AgentCOGSLogger


class TestAgentCOGSIntegration:
    def setup_method(self):
        os.environ["AGENTCOGS_API_KEY"] = "test-api-key"
        os.environ["AGENTCOGS_WORKSPACE_ID"] = "ws-test-uuid"
        os.environ["AGENTCOGS_ENDPOINT"] = "https://api.agentcogs.test"

    def teardown_method(self):
        for key in (
            "AGENTCOGS_API_KEY",
            "AGENTCOGS_WORKSPACE_ID",
            "AGENTCOGS_ENDPOINT",
        ):
            os.environ.pop(key, None)

    def test_logger_initialization(self):
        logger = AgentCOGSLogger()
        assert logger is not None

    def test_logger_missing_api_key(self):
        os.environ.pop("AGENTCOGS_API_KEY", None)
        with pytest.raises(Exception, match="Missing keys.*AGENTCOGS_API_KEY"):
            AgentCOGSLogger()

    def test_build_event_with_user(self):
        logger = AgentCOGSLogger()
        kwargs = {
            "user": "acme_corp",
            "model": "gpt-4o-mini",
            "response_cost": 0.002,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "metadata": {"agentcogs_workflow_id": "support_bot"},
        }
        event = logger._build_event(kwargs, {}, status="completed")
        assert event is not None
        assert event["customer_id"] == "acme_corp"
        assert event["workspace_id"] == "ws-test-uuid"
        assert event["status"] == "completed"
        assert event["workflow_id"] == "support_bot"
        assert event["ts"] == pytest.approx(int(__import__("time").time()), abs=5)
        assert event["models"]["gpt-4o-mini"]["input_tokens"] == 10
        assert event["models"]["gpt-4o-mini"]["usd"] == 0.002
        assert event["metadata"]["source"] == "litellm"

    def test_build_event_skips_without_customer(self):
        logger = AgentCOGSLogger()
        kwargs = {
            "model": "gpt-4o-mini",
            "response_cost": 0.001,
        }
        assert logger._build_event(kwargs, {}, status="completed") is None

    def test_build_event_customer_from_metadata(self):
        logger = AgentCOGSLogger()
        kwargs = {
            "litellm_params": {
                "metadata": {"agentcogs_customer_id": "meta_tenant"},
            },
            "model": "gpt-4",
            "response_cost": 0.01,
        }
        event = logger._build_event(kwargs, {}, status="completed")
        assert event is not None
        assert event["customer_id"] == "meta_tenant"

    def test_build_event_error_status(self):
        logger = AgentCOGSLogger()
        kwargs = {
            "user": "acme",
            "model": "gpt-4",
            "response_cost": 0,
            "exception": "rate limited",
        }
        event = logger._build_event(
            kwargs, {}, status="error", error="rate limited"
        )
        assert event is not None
        assert event["status"] == "error"
        assert event["error"] == "rate limited"

    @patch("litellm.integrations.agentcogs.HTTPHandler")
    def test_log_success_event_posts(self, mock_http_handler):
        mock_post = MagicMock()
        mock_http_handler.return_value.post = mock_post

        logger = AgentCOGSLogger()
        kwargs = {
            "user": "test-user",
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        logger.log_success_event(kwargs, {}, None, None)
        mock_post.assert_called_once()
        payload = json.loads(mock_post.call_args[1]["data"])
        assert payload["customer_id"] == "test-user"
        assert payload["status"] == "completed"

    @patch("litellm.integrations.agentcogs.get_async_httpx_client")
    @pytest.mark.asyncio
    async def test_async_log_success_event_posts(self, mock_get_client):
        mock_post = AsyncMock()
        mock_client = MagicMock()
        mock_client.post = mock_post
        mock_get_client.return_value = mock_client

        logger = AgentCOGSLogger()
        kwargs = {
            "user": "async-user",
            "model": "gpt-4",
            "response_cost": 0.002,
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        await logger.async_log_success_event(kwargs, {}, None, None)
        mock_post.assert_called_once()
        payload = json.loads(mock_post.call_args[1]["data"])
        assert payload["customer_id"] == "async-user"

    @patch("litellm.integrations.agentcogs.get_async_httpx_client")
    @pytest.mark.asyncio
    async def test_async_log_skips_without_customer(self, mock_get_client):
        mock_post = AsyncMock()
        mock_client = MagicMock()
        mock_client.post = mock_post
        mock_get_client.return_value = mock_client

        logger = AgentCOGSLogger()
        kwargs = {"model": "gpt-4", "response_cost": 0.001}
        await logger.async_log_success_event(kwargs, {}, None, None)
        mock_post.assert_not_called()

    @patch("litellm.integrations.agentcogs.get_async_httpx_client")
    @pytest.mark.asyncio
    async def test_async_log_swallows_post_errors(self, mock_get_client):
        mock_post = AsyncMock(side_effect=Exception("network down"))
        mock_client = MagicMock()
        mock_client.post = mock_post
        mock_get_client.return_value = mock_client

        logger = AgentCOGSLogger()
        kwargs = {
            "user": "u1",
            "model": "gpt-4",
            "response_cost": 0.001,
        }
        await logger.async_log_success_event(kwargs, {}, None, None)
