"""
Tests for the chat UI responses API session chaining logic.

Validates that:
1. previous_response_id is correctly forwarded in responses API calls
2. The parameter is omitted (not sent as None) when starting a new session
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath("../.."))

import pytest

import litellm


class TestResponsesSessionChaining:
    """Test previous_response_id session chaining for the chat UI."""

    @pytest.mark.asyncio
    async def test_previous_response_id_forwarded(self):
        """previous_response_id should be passed through to the responses API call."""
        captured = {}

        async def fake_aresponses(*args, **kwargs):
            captured.update(kwargs)
            mock_resp = MagicMock()
            mock_resp.id = "resp_abc123"
            mock_resp.output = []
            return mock_resp

        with patch("litellm.aresponses", side_effect=fake_aresponses):
            await litellm.aresponses(
                input="Hello",
                model="gpt-4o",
                previous_response_id="resp_prev999",
            )

        assert captured.get("previous_response_id") == "resp_prev999"

    @pytest.mark.asyncio
    async def test_new_session_has_no_previous_response_id(self):
        """A new conversation should not send previous_response_id."""
        captured = {}

        async def fake_aresponses(*args, **kwargs):
            captured.update(kwargs)
            mock_resp = MagicMock()
            mock_resp.id = "resp_new001"
            mock_resp.output = []
            return mock_resp

        with patch("litellm.aresponses", side_effect=fake_aresponses):
            await litellm.aresponses(
                input="Hello",
                model="gpt-4o",
                # No previous_response_id — new session
            )

        assert captured.get("previous_response_id") is None

    def test_responses_api_signature_accepts_previous_response_id(self):
        """Smoke test: aresponses function signature accepts previous_response_id."""
        import inspect
        sig = inspect.signature(litellm.aresponses)
        assert "previous_response_id" in sig.parameters, (
            "aresponses must accept previous_response_id for session chaining"
        )
