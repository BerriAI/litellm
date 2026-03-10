"""
Tests for responses API session chaining used by the chat UI.

Verifies that:
1. previous_response_id is correctly forwarded when provided
2. Absence of previous_response_id does not break the call
3. The aresponses function signature exposes the expected parameters
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

import litellm


class TestResponsesSessionChaining:
    """Test previous_response_id session chaining for the chat UI."""

    def test_responses_api_signature_accepts_previous_response_id(self):
        """aresponses must accept previous_response_id and onResponseId-like params."""
        import inspect

        sig = inspect.signature(litellm.aresponses)
        assert "previous_response_id" in sig.parameters, (
            "aresponses must accept previous_response_id for multi-turn session chaining"
        )
        assert "input" in sig.parameters, "aresponses must accept input"
        assert "model" in sig.parameters, "aresponses must accept model"

    @pytest.mark.asyncio
    async def test_previous_response_id_included_in_request_body(self):
        """previous_response_id must appear in the outgoing HTTP request body."""
        import httpx

        captured_body: dict = {}

        async def mock_send(self_transport, request: httpx.Request, **kwargs):
            import json

            try:
                captured_body.update(json.loads(request.content))
            except Exception:
                pass
            # Return a minimal valid responses API response
            response_json = {
                "id": "resp_test123",
                "object": "response",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_001",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                        "status": "completed",
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                "status": "completed",
                "created_at": 1700000000,
            }
            return httpx.Response(
                200,
                json=response_json,
                request=request,
            )

        import unittest.mock as mock

        with mock.patch("httpx.AsyncClient.send", mock_send):
            try:
                await litellm.aresponses(
                    input="hello",
                    model="gpt-4o-mini",
                    previous_response_id="resp_prev_abc",
                    api_key="sk-test-fake",
                )
            except Exception:
                pass  # response parsing may fail; we only care about the outgoing body

        assert captured_body.get("previous_response_id") == "resp_prev_abc", (
            f"Expected previous_response_id in request body, got: {captured_body}"
        )

    @pytest.mark.asyncio
    async def test_no_previous_response_id_omitted_from_request(self):
        """When previous_response_id is None, it must not appear in the request body."""
        import httpx

        captured_body: dict = {}

        async def mock_send(self_transport, request: httpx.Request, **kwargs):
            import json

            try:
                captured_body.update(json.loads(request.content))
            except Exception:
                pass
            response_json = {
                "id": "resp_new001",
                "object": "response",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_001",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                        "status": "completed",
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                "status": "completed",
                "created_at": 1700000000,
            }
            return httpx.Response(200, json=response_json, request=request)

        import unittest.mock as mock

        with mock.patch("httpx.AsyncClient.send", mock_send):
            try:
                await litellm.aresponses(
                    input="hello",
                    model="gpt-4o-mini",
                    previous_response_id=None,
                    api_key="sk-test-fake",
                )
            except Exception:
                pass

        assert "previous_response_id" not in captured_body, (
            "previous_response_id must be omitted from the request body when None"
        )
