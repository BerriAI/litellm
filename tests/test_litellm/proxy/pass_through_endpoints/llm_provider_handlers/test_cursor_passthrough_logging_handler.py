import os
import sys
from datetime import datetime
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.cursor_passthrough_logging_handler import (
    CursorPassthroughLoggingHandler,
    _classify_cursor_request,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)


class TestClassifyCursorRequest:
    def test_should_classify_create_agent(self):
        assert _classify_cursor_request("POST", "/v0/agents") == "cursor:agent:create"

    def test_should_classify_list_agents(self):
        assert _classify_cursor_request("GET", "/v0/agents") == "cursor:agent:list"

    def test_should_classify_agent_status(self):
        assert (
            _classify_cursor_request("GET", "/v0/agents/bc_abc123")
            == "cursor:agent:status"
        )

    def test_should_classify_agent_conversation(self):
        assert (
            _classify_cursor_request("GET", "/v0/agents/bc_abc123/conversation")
            == "cursor:agent:conversation"
        )

    def test_should_classify_agent_followup(self):
        assert (
            _classify_cursor_request("POST", "/v0/agents/bc_abc123/followup")
            == "cursor:agent:followup"
        )

    def test_should_classify_agent_stop(self):
        assert (
            _classify_cursor_request("POST", "/v0/agents/bc_abc123/stop")
            == "cursor:agent:stop"
        )

    def test_should_classify_agent_delete(self):
        assert (
            _classify_cursor_request("DELETE", "/v0/agents/bc_abc123")
            == "cursor:agent:delete"
        )

    def test_should_classify_me_endpoint(self):
        assert _classify_cursor_request("GET", "/v0/me") == "cursor:account:info"

    def test_should_classify_models_endpoint(self):
        assert _classify_cursor_request("GET", "/v0/models") == "cursor:models:list"

    def test_should_classify_repositories_endpoint(self):
        assert (
            _classify_cursor_request("GET", "/v0/repositories")
            == "cursor:repositories:list"
        )


class TestCursorRouteDetection:
    def test_should_detect_cursor_route_by_custom_llm_provider(self):
        handler = PassThroughEndpointLogging()
        assert handler.is_cursor_route("https://api.cursor.com/v0/agents", "cursor")

    def test_should_detect_cursor_route_by_hostname(self):
        handler = PassThroughEndpointLogging()
        assert handler.is_cursor_route("https://api.cursor.com/v0/agents")

    def test_should_not_detect_non_cursor_route(self):
        handler = PassThroughEndpointLogging()
        assert not handler.is_cursor_route("https://api.openai.com/v1/completions")


class TestCursorPassthroughHandler:
    def test_should_log_agent_creation_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.request = MagicMock()
        mock_response.request.method = "POST"
        mock_response.text = '{"id": "bc_abc123"}'

        mock_logging_obj = MagicMock()
        mock_logging_obj.litellm_call_id = "test-call-id"
        mock_logging_obj.model_call_details = {}

        response_body = {
            "id": "bc_abc123",
            "name": "Test Agent",
            "status": "CREATING",
        }

        result = CursorPassthroughLoggingHandler.cursor_passthrough_handler(
            httpx_response=mock_response,
            response_body=response_body,
            logging_obj=mock_logging_obj,
            url_route="https://api.cursor.com/v0/agents",
            result='{"id": "bc_abc123"}',
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"prompt": {"text": "Add README"}},
        )

        assert result["result"] is not None
        assert result["kwargs"]["model"] == "cursor/cursor:agent:create"
        assert result["kwargs"]["response_cost"] == 0.0
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "cursor"
