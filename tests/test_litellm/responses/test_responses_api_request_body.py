"""
Test that litellm.responses() / litellm.aresponses() send the expected request body
over the wire and surface provider errors correctly. Expected JSON bodies are stored
in expected_responses_api_request/.
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


def _load_expected_body(filename: str) -> dict:
    expected_path = _expected_dir() / filename
    assert expected_path.exists(), f"Expected file not found: {expected_path}"
    with open(expected_path) as f:
        return json.load(f)


def _minimal_responses_api_payload(response_id: str, model: str) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1734366691,
        "status": "completed",
        "model": model,
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


def _assert_request_body_matches(request_body: dict, expected_body: dict) -> None:
    for key, expected_value in expected_body.items():
        assert key in request_body, f"Missing key in request body: {key}"
        assert (
            request_body[key] == expected_value
        ), f"Mismatch for key {key}: got {request_body[key]!r}, expected {expected_value!r}"


@pytest.mark.asyncio
async def test_aresponses_context_management_and_shell_request_body_matches_expected():
    """
    Call litellm.aresponses() with context_management and shell tool;
    assert the httpx POST request body matches the expected JSON.
    """
    expected_body = _load_expected_body("context_management_and_shell.json")

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(
            _minimal_responses_api_payload("resp_ctx_shell_test", "gpt-4o"), 200
        )

        await litellm.aresponses(
            model="openai/gpt-4o",
            input=expected_body["input"],
            context_management=expected_body["context_management"],
            tools=expected_body["tools"],
            tool_choice=expected_body["tool_choice"],
            max_output_tokens=expected_body["max_output_tokens"],
        )

        mock_post.assert_called_once()
        _assert_request_body_matches(mock_post.call_args.kwargs["json"], expected_body)


@pytest.mark.asyncio
async def test_aresponses_azure_shell_tool_request_body_matches_expected():
    """
    Call litellm.aresponses() on the Azure route with the shell tool;
    assert the httpx POST request body carries the shell tool verbatim.
    """
    expected_body = _load_expected_body("azure_shell_tool.json")

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(
            _minimal_responses_api_payload("resp_azure_shell_test", "gpt-5-mini"), 200
        )

        await litellm.aresponses(
            model="azure/gpt-5-mini",
            api_base="https://fake-resource.openai.azure.com",
            api_key="fake-api-key",
            api_version="2025-03-01-preview",
            input=expected_body["input"],
            tools=expected_body["tools"],
            tool_choice=expected_body["tool_choice"],
            max_output_tokens=expected_body["max_output_tokens"],
        )

        mock_post.assert_called_once()
        _assert_request_body_matches(mock_post.call_args.kwargs["json"], expected_body)


@pytest.mark.asyncio
async def test_aresponses_azure_shell_tool_400_maps_to_bad_request_error():
    """
    Azure rejects the shell tool for unsupported deployments with a 400;
    litellm must surface that as litellm.BadRequestError carrying the provider message.
    """
    error_body = {
        "error": {
            "message": "Tool of type 'shell' is not supported with this model.",
            "type": "invalid_request_error",
            "param": "tools",
            "code": None,
        }
    }

    def _raise_azure_400(*args, **kwargs):
        response = httpx.Response(
            status_code=400,
            json=error_body,
            request=httpx.Request(
                "POST",
                kwargs.get(
                    "url",
                    "https://fake-resource.openai.azure.com/openai/responses",
                ),
            ),
        )
        response.raise_for_status()

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.side_effect = _raise_azure_400

        with pytest.raises(litellm.BadRequestError) as excinfo:
            await litellm.aresponses(
                model="azure/gpt-5-mini",
                api_base="https://fake-resource.openai.azure.com",
                api_key="fake-api-key",
                api_version="2025-03-01-preview",
                input="List files in /mnt/data and run python --version.",
                tools=[{"type": "shell", "environment": {"type": "container_auto"}}],
                tool_choice="auto",
                max_output_tokens=256,
            )

    assert excinfo.value.status_code == 400
    assert "shell" in str(excinfo.value).lower()
    assert "not supported" in str(excinfo.value).lower()
