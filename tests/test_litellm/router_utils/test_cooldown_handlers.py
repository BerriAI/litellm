"""
Unit tests for cooldown handler logic in litellm.router_utils.cooldown_handlers
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.router_utils.cooldown_handlers import _is_cooldown_required


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.allowed_fails = None
    router.allowed_fails_policy = None
    return router


class TestIsCooldownRequired:
    """Tests for _is_cooldown_required behavior with connection errors."""

    def test_api_connection_error_triggers_cooldown(self, mock_router):
        """
        APIConnectionError (unreachable host) should trigger cooldown so the
        router can fail over to healthy deployments.

        Regression test for https://github.com/BerriAI/litellm/issues/27362
        """
        exception = litellm.APIConnectionError(
            message="Cannot connect to host example:11434",
            llm_provider="ollama",
            model="llama3.1:8b",
        )

        result = _is_cooldown_required(
            litellm_router_instance=mock_router,
            model_id="test_deployment",
            exception_status=500,
            exception_str=str(exception),
        )

        assert result is True

    def test_server_error_triggers_cooldown(self, mock_router):
        """500-level errors should trigger cooldown."""
        result = _is_cooldown_required(
            litellm_router_instance=mock_router,
            model_id="test_deployment",
            exception_status=500,
            exception_str="Internal Server Error",
        )

        assert result is True

    def test_rate_limit_triggers_cooldown(self, mock_router):
        """429 errors should trigger cooldown."""
        result = _is_cooldown_required(
            litellm_router_instance=mock_router,
            model_id="test_deployment",
            exception_status=429,
            exception_str="Rate limit exceeded",
        )

        assert result is True

    def test_bad_request_does_not_trigger_cooldown(self, mock_router):
        """400 errors (except 401, 404, 408, 429) should not trigger cooldown."""
        result = _is_cooldown_required(
            litellm_router_instance=mock_router,
            model_id="test_deployment",
            exception_status=400,
            exception_str="Bad Request",
        )

        assert result is False

    def test_empty_exception_status_does_not_trigger_cooldown(self, mock_router):
        """Empty string exception status should not trigger cooldown."""
        result = _is_cooldown_required(
            litellm_router_instance=mock_router,
            model_id="test_deployment",
            exception_status="",
        )

        assert result is False
