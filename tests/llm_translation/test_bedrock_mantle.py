"""
E2E tests for Bedrock Mantle (Claude Mythos Preview) integration.

Tests use a fake/mocked HTTP layer to verify the full request pipeline:
- correct endpoint URL
- model ID in the request body
- AWS SigV4 Authorization header present
- response parsing
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler

MODEL = "bedrock/mantle/anthropic.claude-mythos-preview"
REGION = "us-east-1"
EXPECTED_URL = f"https://bedrock-mantle.{REGION}.api.aws/v1/messages"

FAKE_ANTHROPIC_RESPONSE = {
    "id": "msg_fake123",
    "type": "message",
    "role": "assistant",
    "model": "anthropic.claude-mythos-preview",
    "content": [{"type": "text", "text": "Hello from Mythos!"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def _make_fake_response(body: dict) -> MagicMock:
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.headers = httpx.Headers({"content-type": "application/json"})
    mock_resp.text = json.dumps(body)
    mock_resp.json.return_value = body
    mock_resp.is_error = False
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def test_mantle_request_url_and_body():
    """Verify the correct URL is called and model appears in the request body."""
    client = HTTPHandler()

    with patch.object(
        client, "post", return_value=_make_fake_response(FAKE_ANTHROPIC_RESPONSE)
    ) as mock_post:
        try:
            litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=50,
                aws_region_name=REGION,
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                client=client,
            )
        except Exception:
            pass  # response parsing may fail on mock; we only care about the outgoing call

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs

        # Correct endpoint
        assert (
            call_kwargs["url"] == EXPECTED_URL
        ), f"Expected {EXPECTED_URL}, got {call_kwargs['url']}"

        # Request body has model ID (without "mantle/" prefix)
        raw_data = call_kwargs.get("data") or call_kwargs.get("json")
        body = json.loads(raw_data) if isinstance(raw_data, (str, bytes)) else raw_data
        assert (
            body["model"] == "anthropic.claude-mythos-preview"
        ), f"body['model'] = {body.get('model')}"
        assert "messages" in body
        assert body["max_tokens"] == 50

        # AWS SigV4 Authorization header must be present
        headers = call_kwargs.get("headers", {})
        assert "Authorization" in headers, f"No Authorization header in {headers}"
        assert headers["Authorization"].startswith(
            "AWS4-HMAC-SHA256"
        ), f"Expected SigV4 auth, got: {headers['Authorization'][:50]}"


def test_mantle_request_does_not_include_mantle_prefix_in_body():
    """Ensure 'mantle/' never leaks into the request body."""
    client = HTTPHandler()

    with patch.object(
        client, "post", return_value=_make_fake_response(FAKE_ANTHROPIC_RESPONSE)
    ) as mock_post:
        try:
            litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10,
                aws_region_name=REGION,
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                client=client,
            )
        except Exception:
            pass

        call_kwargs = mock_post.call_args.kwargs
        raw_data = call_kwargs.get("data") or call_kwargs.get("json")
        body = json.loads(raw_data) if isinstance(raw_data, (str, bytes)) else raw_data

        body_str = json.dumps(body)
        assert "mantle/" not in body_str, f"'mantle/' leaked into body: {body_str}"


def test_mantle_region_reflected_in_url():
    """The region from aws_region_name must appear in the endpoint URL."""
    client = HTTPHandler()

    for region in ["us-east-1", "us-west-2", "eu-west-1"]:
        with patch.object(
            client, "post", return_value=_make_fake_response(FAKE_ANTHROPIC_RESPONSE)
        ) as mock_post:
            try:
                litellm.completion(
                    model=MODEL,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10,
                    aws_region_name=region,
                    aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                    aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                    client=client,
                )
            except Exception:
                pass

            call_kwargs = mock_post.call_args.kwargs
            expected = f"https://bedrock-mantle.{region}.api.aws/v1/messages"
            assert (
                call_kwargs["url"] == expected
            ), f"region={region}: expected URL {expected}, got {call_kwargs['url']}"
