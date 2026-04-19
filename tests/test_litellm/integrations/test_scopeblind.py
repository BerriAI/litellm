"""
Tests for the ScopeBlind integration.

Tests that the ScopeBlind callback correctly:
1. Validates environment variables
2. Builds payloads from LiteLLM kwargs
3. Skips events when no device identity headers are present
4. Sends events when device identity headers are present
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path


class TestScopeBlindValidation:
    """Test environment validation."""

    def test_missing_api_key_raises(self):
        """ScopeBlindLogger raises if SCOPEBLIND_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if it exists
            os.environ.pop("SCOPEBLIND_API_KEY", None)
            from litellm.integrations.scopeblind import ScopeBlindLogger

            with pytest.raises(Exception, match="Missing keys"):
                ScopeBlindLogger()

    def test_valid_environment(self):
        """ScopeBlindLogger initializes with valid env vars."""
        with patch.dict(
            os.environ,
            {"SCOPEBLIND_API_KEY": "sb_test_key"},
        ):
            from litellm.integrations.scopeblind import ScopeBlindLogger

            logger = ScopeBlindLogger()
            assert logger.scopeblind_api_key == "sb_test_key"
            assert logger.scopeblind_endpoint == "https://api.scopeblind.com"

    def test_custom_endpoint(self):
        """ScopeBlindLogger respects SCOPEBLIND_ENDPOINT env var."""
        with patch.dict(
            os.environ,
            {
                "SCOPEBLIND_API_KEY": "sb_test_key",
                "SCOPEBLIND_ENDPOINT": "https://custom.scopeblind.dev",
            },
        ):
            from litellm.integrations.scopeblind import ScopeBlindLogger

            logger = ScopeBlindLogger()
            assert logger.scopeblind_endpoint == "https://custom.scopeblind.dev"


class TestScopeBlindPayload:
    """Test payload building."""

    @pytest.fixture
    def logger(self):
        with patch.dict(
            os.environ,
            {"SCOPEBLIND_API_KEY": "sb_test_key"},
        ):
            from litellm.integrations.scopeblind import ScopeBlindLogger

            return ScopeBlindLogger()

    def test_build_payload_with_device_headers(self, logger):
        """Payload includes DPoP proof and device ID from headers."""
        kwargs = {
            "litellm_call_id": "call-123",
            "model": "gpt-4",
            "response_cost": 0.03,
            "user": "user-456",
            "litellm_params": {
                "metadata": {
                    "headers": {
                        "x-scopeblind-dpop": "eyJ0eXAiOiJkcG9wK2p3dCJ9.test",
                        "x-scopeblind-device-id": "device-789",
                    }
                }
            },
        }
        response_obj = MagicMock()
        response_obj.__class__.__name__ = "ModelResponse"

        payload = logger._build_payload(
            kwargs, response_obj, "2024-01-01T00:00:00", "2024-01-01T00:00:01", "llm_call_success"
        )

        assert payload["event"] == "llm_call_success"
        assert payload["model"] == "gpt-4"
        assert payload["user"] == "user-456"
        assert payload["cost"] == 0.03
        assert payload["dpop_proof"] == "eyJ0eXAiOiJkcG9wK2p3dCJ9.test"
        assert payload["device_id"] == "device-789"
        assert payload["litellm_call_id"] == "call-123"

    def test_build_payload_without_device_headers(self, logger):
        """Payload has None for device fields when no headers present."""
        kwargs = {
            "litellm_call_id": "call-123",
            "model": "gpt-4",
            "litellm_params": {"metadata": {}},
        }
        response_obj = MagicMock()

        payload = logger._build_payload(
            kwargs, response_obj, "2024-01-01T00:00:00", "2024-01-01T00:00:01", "llm_call_success"
        )

        assert payload["dpop_proof"] is None
        assert payload["device_id"] is None

    def test_build_payload_with_x_device_identity_header(self, logger):
        """Payload extracts X-Device-Identity header."""
        kwargs = {
            "litellm_call_id": "call-123",
            "model": "gpt-4",
            "litellm_params": {
                "metadata": {
                    "headers": {
                        "x-device-identity": 'ScopeBlind/1.0; info="https://scopeblind.com/verify-agent"',
                    }
                }
            },
        }
        response_obj = MagicMock()

        payload = logger._build_payload(
            kwargs, response_obj, "2024-01-01T00:00:00", "2024-01-01T00:00:01", "llm_call_success"
        )

        assert payload["device_id"] == 'ScopeBlind/1.0; info="https://scopeblind.com/verify-agent"'


class TestScopeBlindEventSending:
    """Test that events are sent or skipped correctly."""

    @pytest.fixture
    def logger(self):
        with patch.dict(
            os.environ,
            {"SCOPEBLIND_API_KEY": "sb_test_key"},
        ):
            from litellm.integrations.scopeblind import ScopeBlindLogger

            instance = ScopeBlindLogger()
            instance.sync_http_handler = MagicMock()
            return instance

    def test_skips_event_without_device_identity(self, logger):
        """log_success_event should not send when no device headers."""
        kwargs = {
            "litellm_call_id": "call-123",
            "model": "gpt-4",
            "litellm_params": {"metadata": {}},
        }
        response_obj = MagicMock()

        logger.log_success_event(
            kwargs, response_obj, "2024-01-01T00:00:00", "2024-01-01T00:00:01"
        )

        logger.sync_http_handler.post.assert_not_called()

    def test_sends_event_with_device_identity(self, logger):
        """log_success_event should send when DPoP header is present."""
        kwargs = {
            "litellm_call_id": "call-123",
            "model": "gpt-4",
            "litellm_params": {
                "metadata": {
                    "headers": {
                        "x-scopeblind-dpop": "eyJ0eXAiOiJkcG9wK2p3dCJ9.test",
                    }
                }
            },
        }
        response_obj = MagicMock()

        logger.log_success_event(
            kwargs, response_obj, "2024-01-01T00:00:00", "2024-01-01T00:00:01"
        )

        logger.sync_http_handler.post.assert_called_once()
        call_kwargs = logger.sync_http_handler.post.call_args
        assert "scopeblind" in call_kwargs.kwargs["url"]


class TestScopeBlindRegistration:
    """Test that ScopeBlind is registered as a named callback."""

    def test_scopeblind_in_callbacks_literal(self):
        """scopeblind should be in the known callbacks list."""
        import litellm

        assert "scopeblind" in litellm._known_custom_logger_compatible_callbacks
