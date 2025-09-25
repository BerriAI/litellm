"""
Test that team callbacks are merged with config callbacks instead of overwriting them.

This tests the fix for issue #12118 where callbacks added through the UI (team callbacks)
were overwriting callbacks set in the config file.
"""

import pytest
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.proxy._types import UserAPIKeyAuth, TeamCallbackMetadata
from unittest.mock import AsyncMock, MagicMock
from fastapi import Request


class TestTeamCallbackMerging:
    @pytest.mark.asyncio
    async def test_team_callbacks_merge_with_existing_callbacks(self):
        """Test that team callbacks are merged with existing callbacks in the request data"""
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.method = "POST"
        mock_request.headers = {}
        mock_request.query_params = {}

        # Create data with existing callbacks (simulating config callbacks)
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "success_callback": ["langfuse"],  # Config callback
            "failure_callback": ["sentry"],    # Config callback
        }

        # Create user API key dict with team metadata containing UI callbacks
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            key_alias="test-alias",
            team_metadata={
                "callback_settings": {
                    "success_callback": ["slack", "datadog"],  # UI callbacks
                    "failure_callback": ["slack"],              # UI callback
                }
            },
        )

        # Create mock proxy config
        mock_proxy_config = MagicMock()
        mock_proxy_config.load_team_config.return_value = {}

        # Call the function
        result = await add_litellm_data_to_request(
            data=data,
            request=mock_request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=mock_proxy_config,
            general_settings={},
        )

        # Verify callbacks were merged, not overwritten
        assert "langfuse" in result["success_callback"]  # Config callback preserved
        assert "slack" in result["success_callback"]     # UI callback added
        assert "datadog" in result["success_callback"]   # UI callback added
        assert len(result["success_callback"]) == 3      # All callbacks present

        assert "sentry" in result["failure_callback"]    # Config callback preserved
        assert "slack" in result["failure_callback"]     # UI callback added
        assert len(result["failure_callback"]) == 2      # All callbacks present

    @pytest.mark.asyncio
    async def test_team_callbacks_handle_empty_existing_callbacks(self):
        """Test that team callbacks work when there are no existing callbacks"""
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.method = "POST"
        mock_request.headers = {}
        mock_request.query_params = {}

        # Create data without existing callbacks
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
        }

        # Create user API key dict with team metadata
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            key_alias="test-alias",
            team_metadata={
                "callback_settings": {
                    "success_callback": ["langfuse"],
                    "failure_callback": ["slack"],
                }
            },
        )

        # Create mock proxy config
        mock_proxy_config = MagicMock()
        mock_proxy_config.load_team_config.return_value = {}

        # Call the function
        result = await add_litellm_data_to_request(
            data=data,
            request=mock_request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=mock_proxy_config,
            general_settings={},
        )

        # Verify team callbacks were added
        assert result["success_callback"] == ["langfuse"]
        assert result["failure_callback"] == ["slack"]

    @pytest.mark.asyncio
    async def test_team_callbacks_deduplicate(self):
        """Test that duplicate callbacks are removed when merging"""
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.method = "POST"
        mock_request.headers = {}
        mock_request.query_params = {}

        # Create data with existing callbacks
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "success_callback": ["langfuse", "slack"],  # Slack already present
        }

        # Create user API key dict with team metadata
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            key_alias="test-alias",
            team_metadata={
                "callback_settings": {
                    "success_callback": ["slack", "datadog"],  # Slack duplicate
                }
            },
        )

        # Create mock proxy config
        mock_proxy_config = MagicMock()
        mock_proxy_config.load_team_config.return_value = {}

        # Call the function
        result = await add_litellm_data_to_request(
            data=data,
            request=mock_request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=mock_proxy_config,
            general_settings={},
        )

        # Verify duplicates were removed
        assert result["success_callback"].count("slack") == 1
        assert "langfuse" in result["success_callback"]
        assert "datadog" in result["success_callback"]
        assert len(result["success_callback"]) == 3

    @pytest.mark.asyncio
    async def test_no_team_callbacks(self):
        """Test that existing callbacks are preserved when no team callbacks exist"""
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.method = "POST"
        mock_request.headers = {}
        mock_request.query_params = {}

        # Create data with existing callbacks
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "success_callback": ["langfuse"],
            "failure_callback": ["sentry"],
        }

        # Create user API key dict without team metadata
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            key_alias="test-alias",
        )

        # Create mock proxy config
        mock_proxy_config = MagicMock()
        mock_proxy_config.load_team_config.return_value = {}

        # Call the function
        result = await add_litellm_data_to_request(
            data=data,
            request=mock_request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=mock_proxy_config,
            general_settings={},
        )

        # Verify existing callbacks are preserved
        assert result["success_callback"] == ["langfuse"]
        assert result["failure_callback"] == ["sentry"]

    @pytest.mark.asyncio
    async def test_team_callbacks_handle_non_list_existing_callbacks(self):
        """Test that team callbacks work when existing callbacks are single strings"""
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.method = "POST"
        mock_request.headers = {}
        mock_request.query_params = {}

        # Create data with single string callbacks (not lists)
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "success_callback": "langfuse",  # Single string, not list
            "failure_callback": "sentry",    # Single string, not list
        }

        # Create user API key dict with team metadata
        user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            key_alias="test-alias",
            team_metadata={
                "callback_settings": {
                    "success_callback": ["slack"],
                    "failure_callback": ["datadog"],
                }
            },
        )

        # Create mock proxy config
        mock_proxy_config = MagicMock()
        mock_proxy_config.load_team_config.return_value = {}

        # Call the function
        result = await add_litellm_data_to_request(
            data=data,
            request=mock_request,
            user_api_key_dict=user_api_key_dict,
            proxy_config=mock_proxy_config,
            general_settings={},
        )

        # Verify callbacks were merged correctly
        assert "langfuse" in result["success_callback"]
        assert "slack" in result["success_callback"]
        assert len(result["success_callback"]) == 2

        assert "sentry" in result["failure_callback"]
        assert "datadog" in result["failure_callback"]
        assert len(result["failure_callback"]) == 2