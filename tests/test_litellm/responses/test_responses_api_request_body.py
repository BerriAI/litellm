"""
Test that litellm.responses() / litellm.aresponses() send the expected request body
over the wire. Expected JSON bodies are stored in expected_responses_api_request/.
"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm


def _expected_dir() -> Path:
    """Path to expected_responses_api_request folder (sibling of test_litellm/responses)."""
    return Path(__file__).resolve().parent.parent / "expected_responses_api_request"


@pytest.mark.asyncio
async def test_aresponses_context_management_and_shell_request_body_matches_expected():
    """
    Call litellm.aresponses() with context_management and shell tool;
    assert the httpx POST request body matches the expected JSON.
    """
    expected_path = _expected_dir() / "context_management_and_shell.json"
    assert expected_path.exists(), f"Expected file not found: {expected_path}"
    with open(expected_path) as f:
        expected_body = json.load(f)

    # Minimal Responses API response so parsing succeeds
    mock_response = {
        "id": "resp_ctx_shell_test",
        "object": "response",
        "created_at": 1734366691,
        "status": "completed",
        "model": "gpt-4o",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Done.", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": None,
        "truncation": None,
        "user": None,
    }

    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)
            self.headers = httpx.Headers({})

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(mock_response, 200)

        await litellm.aresponses(
            model="openai/gpt-4o",
            input=expected_body["input"],
            context_management=expected_body["context_management"],
            tools=expected_body["tools"],
            tool_choice=expected_body["tool_choice"],
            max_output_tokens=expected_body["max_output_tokens"],
        )

        mock_post.assert_called_once()
        request_body = mock_post.call_args.kwargs["json"]

        for key, expected_value in expected_body.items():
            assert key in request_body, f"Missing key in request body: {key}"
            assert request_body[key] == expected_value, (
                f"Mismatch for key {key}: got {request_body[key]!r}, expected {expected_value!r}"
            )
