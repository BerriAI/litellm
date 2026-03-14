"""
Tests for Lago integration - cost logging to Lago.

Covers end_user_id resolution from proxy_server_request.body and from
metadata.user_api_key_end_user_id (when lazy_proxy_request_body omits body).
"""

import os

import pytest

from litellm.integrations.lago import LagoLogger


class TestLagoIntegration:
    """Test suite for Lago integration"""

    def setup_method(self):
        """Set up test environment"""
        os.environ["LAGO_API_KEY"] = "test-api-key"
        os.environ["LAGO_API_BASE"] = "https://test.lago.com"
        os.environ["LAGO_API_EVENT_CODE"] = "litellm-cost"

    def teardown_method(self):
        """Clean up test environment"""
        os.environ.pop("LAGO_API_KEY", None)
        os.environ.pop("LAGO_API_BASE", None)
        os.environ.pop("LAGO_API_EVENT_CODE", None)
        os.environ.pop("LAGO_API_CHARGE_BY", None)

    def test_lago_logger_initialization(self):
        """Test that LagoLogger initializes correctly with required env vars"""
        logger = LagoLogger()
        assert logger is not None

    def test_lago_logger_missing_api_key(self):
        """Test that LagoLogger raises exception when API key is missing"""
        os.environ.pop("LAGO_API_KEY", None)
        with pytest.raises(Exception, match="Missing keys.*LAGO_API_KEY"):
            LagoLogger()

    def test_common_logic_end_user_id_from_body(self):
        """Test that _common_logic gets end_user_id from proxy_server_request.body when present."""
        logger = LagoLogger()
        kwargs = {
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "litellm_call_id": "test-call-id",
            "litellm_params": {
                "metadata": {},
                "proxy_server_request": {"body": {"user": "user-from-body"}},
            },
        }
        response_obj = {"id": "test-response-id"}

        result = logger._common_logic(kwargs, response_obj)

        assert result["event"]["external_subscription_id"] == "user-from-body"
        assert result["event"]["properties"]["model"] == "gpt-3.5-turbo"
        assert result["event"]["properties"]["response_cost"] == 0.001

    def test_common_logic_end_user_id_from_metadata_when_body_absent(self):
        """Test that _common_logic falls back to metadata.user_api_key_end_user_id when body is absent.

        This supports lazy_proxy_request_body: when proxy_server_request.body is omitted
        to reduce memory, Lago uses metadata.user_api_key_end_user_id for charge_by=end_user_id.
        """
        logger = LagoLogger()
        kwargs = {
            "model": "gpt-4",
            "response_cost": 0.002,
            "litellm_call_id": "test-call-id",
            "litellm_params": {
                "metadata": {"user_api_key_end_user_id": "user-from-metadata"},
                "proxy_server_request": {},  # No body - lazy_proxy_request_body case
            },
        }
        response_obj = {"id": "test-response-id"}

        result = logger._common_logic(kwargs, response_obj)

        assert result["event"]["external_subscription_id"] == "user-from-metadata"
        assert result["event"]["properties"]["model"] == "gpt-4"

    def test_common_logic_body_takes_precedence_over_metadata(self):
        """When both body.user and metadata.user_api_key_end_user_id exist, body wins."""
        logger = LagoLogger()
        kwargs = {
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "litellm_params": {
                "metadata": {"user_api_key_end_user_id": "user-from-metadata"},
                "proxy_server_request": {"body": {"user": "user-from-body"}},
            },
        }
        response_obj = {}

        result = logger._common_logic(kwargs, response_obj)

        assert result["event"]["external_subscription_id"] == "user-from-body"

    def test_common_logic_charge_by_user_id(self):
        """Test LAGO_API_CHARGE_BY=user_id uses user_api_key_user_id."""
        os.environ["LAGO_API_CHARGE_BY"] = "user_id"
        logger = LagoLogger()
        kwargs = {
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "litellm_params": {
                "metadata": {
                    "user_api_key_user_id": "user-123",
                    "user_api_key_team_id": "team-456",
                },
                "proxy_server_request": {},
            },
        }
        response_obj = {}

        result = logger._common_logic(kwargs, response_obj)

        assert result["event"]["external_subscription_id"] == "user-123"

    def test_common_logic_charge_by_team_id(self):
        """Test LAGO_API_CHARGE_BY=team_id uses user_api_key_team_id."""
        os.environ["LAGO_API_CHARGE_BY"] = "team_id"
        logger = LagoLogger()
        kwargs = {
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "litellm_params": {
                "metadata": {
                    "user_api_key_user_id": "user-123",
                    "user_api_key_team_id": "team-456",
                },
                "proxy_server_request": {},
            },
        }
        response_obj = {}

        result = logger._common_logic(kwargs, response_obj)

        assert result["event"]["external_subscription_id"] == "team-456"

    def test_common_logic_missing_end_user_id_raises(self):
        """Test that _common_logic raises when charge_by=end_user_id but neither body nor metadata has it."""
        logger = LagoLogger()
        kwargs = {
            "model": "gpt-3.5-turbo",
            "response_cost": 0.001,
            "litellm_params": {
                "metadata": {},
                "proxy_server_request": {},
            },
        }
        response_obj = {}

        with pytest.raises(Exception, match="External Customer ID is not set"):
            logger._common_logic(kwargs, response_obj)
