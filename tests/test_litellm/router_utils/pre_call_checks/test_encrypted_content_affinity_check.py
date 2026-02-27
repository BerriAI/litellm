"""
Tests for encrypted_content_affinity pre-call check.

The mechanism works without any cache:
- On response: item IDs for output items with `encrypted_content` are rewritten to
  `encitem_{base64("litellm:model_id:{model_id};item_id:{original_id}")}`.
- On routing: `EncryptedContentAffinityCheck` decodes the `encitem_` prefix to extract
  `model_id` and pins the request to that deployment.
- Before forwarding: `_restore_encrypted_content_item_ids_in_input` decodes the IDs back
  to their original form before sending to the upstream provider.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import json

import litellm
from litellm.responses.utils import ResponsesAPIRequestUtils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockResponse:
    def __init__(self, json_data, status_code):
        self._json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)
        self.headers = {}

    def json(self):
        return self._json_data


def _get_item_id(item) -> str:
    """Extract item ID from either a Pydantic model or a dict."""
    if isinstance(item, dict):
        return item.get("id", "")
    return getattr(item, "id", "") or ""


def _has_encrypted_content(item) -> bool:
    """Check whether an output item carries encrypted_content."""
    if isinstance(item, dict):
        return "encrypted_content" in item
    return hasattr(item, "encrypted_content") and getattr(item, "encrypted_content") is not None


def _extract_encoded_item_id(response) -> str:
    """
    Walk the response output and return the first litellm-encoded item ID
    (i.e. one that starts with ``encitem_``).
    """
    for item in response.output or []:
        item_id = _get_item_id(item)
        if item_id.startswith("encitem_"):
            return item_id
    return ""


# ---------------------------------------------------------------------------
# Unit tests for encoding / decoding utilities
# ---------------------------------------------------------------------------


class TestEncryptedItemIdCodec:
    def test_roundtrip(self):
        model_id = "deployment-1"
        original_item_id = "rs_abc123def456"
        encoded = ResponsesAPIRequestUtils._build_encrypted_item_id(model_id, original_item_id)
        assert encoded.startswith("encitem_")
        decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(encoded)
        assert decoded is not None
        assert decoded["model_id"] == model_id
        assert decoded["item_id"] == original_item_id

    def test_decode_without_padding(self):
        """Decoding must succeed even if base64 padding (=) was stripped in transit."""
        model_id = "gpt-5.1-codex-openai-2"
        original_item_id = "rs_0efb96cb222403210069a01d5d52588196a9dc394ffdb89d00"
        encoded = ResponsesAPIRequestUtils._build_encrypted_item_id(model_id, original_item_id)
        # Strip any trailing '=' to simulate what happens in transit
        stripped = encoded.rstrip("=")
        decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(stripped)
        assert decoded is not None
        assert decoded["model_id"] == model_id
        assert decoded["item_id"] == original_item_id

    def test_non_encoded_id_returns_none(self):
        assert ResponsesAPIRequestUtils._decode_encrypted_item_id("rs_abc123") is None
        assert ResponsesAPIRequestUtils._decode_encrypted_item_id("msg_abc") is None
        assert ResponsesAPIRequestUtils._decode_encrypted_item_id("") is None

    def test_semicolon_in_item_id(self):
        """item_id values containing ';' must survive the roundtrip."""
        model_id = "deployment-1"
        original_item_id = "rs_part1;part2;part3"
        encoded = ResponsesAPIRequestUtils._build_encrypted_item_id(model_id, original_item_id)
        decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(encoded)
        assert decoded is not None
        assert decoded["item_id"] == original_item_id


class TestUpdateEncryptedContentItemIds:
    def test_rewrites_encrypted_items_in_dict_response(self):
        model_id = "deployment-1"
        response = {
            "id": "resp_123",
            "output": [
                {"id": "msg_abc", "type": "message", "content": []},
                {"id": "rs_xyz", "type": "reasoning", "encrypted_content": "secret"},
            ],
        }
        result = ResponsesAPIRequestUtils._update_encrypted_content_item_ids_in_response(
            response, model_id
        )
        # Plain message item untouched
        assert result["output"][0]["id"] == "msg_abc"
        # Reasoning item with encrypted_content gets encoded
        encoded_id = result["output"][1]["id"]
        assert encoded_id.startswith("encitem_")
        decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(encoded_id)
        assert decoded["model_id"] == model_id
        assert decoded["item_id"] == "rs_xyz"

    def test_no_op_when_model_id_is_none(self):
        response = {
            "output": [{"id": "rs_xyz", "type": "reasoning", "encrypted_content": "secret"}]
        }
        result = ResponsesAPIRequestUtils._update_encrypted_content_item_ids_in_response(
            response, None
        )
        assert result["output"][0]["id"] == "rs_xyz"


class TestRestoreEncryptedContentItemIds:
    def test_restores_encoded_ids(self):
        model_id = "deployment-1"
        original_id = "rs_encrypted_item_456"
        encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id(model_id, original_id)

        request_input = [
            {"type": "message", "id": "msg_abc123", "role": "assistant"},
            {"type": "reasoning", "id": encoded_id, "encrypted_content": "secret"},
        ]
        restored = ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input(
            request_input
        )
        assert restored[0]["id"] == "msg_abc123"
        assert restored[1]["id"] == original_id

    def test_no_op_for_plain_string_input(self):
        result = ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input(
            "Hello world"
        )
        assert result == "Hello world"

    def test_no_op_for_unencoded_ids(self):
        request_input = [{"type": "message", "id": "msg_plain"}]
        result = ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input(
            request_input
        )
        assert result[0]["id"] == "msg_plain"


# ---------------------------------------------------------------------------
# Integration tests (router-level)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encrypted_content_affinity_tracks_and_routes():
    """
    The first response rewrites encrypted-content item IDs to encoded form.
    The follow-up request with those encoded IDs is pinned to the same deployment.
    """
    mock_response_data = {
        "id": "resp_mock-123",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "openai/gpt-5.1-codex",
        "output": [
            {
                "type": "message",
                "id": "msg_abc123",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
            },
            {
                "type": "reasoning",
                "id": "rs_encrypted_item_456",
                "status": "completed",
                "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
            },
        ],
        "parallel_tool_calls": True,
        "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
        "error": None,
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-1",
                },
                "model_info": {"id": "deployment-1"},
            },
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-2",
                },
                "model_info": {"id": "deployment-2"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
    )

    selected_deployments = []

    def deterministic_choice(seq):
        if len(selected_deployments) == 0:
            return seq[0]
        return seq[1] if len(seq) > 1 else seq[0]

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post, patch(
        "litellm.router_strategy.simple_shuffle.random.choice",
        side_effect=deterministic_choice,
    ):
        mock_post.return_value = MockResponse(mock_response_data, 200)

        # First request — goes to deployment-1 via deterministic_choice
        first_response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input="Hello, how are you?",
        )
        first_model_id = first_response._hidden_params["model_id"]
        selected_deployments.append(first_model_id)

        # The response must have rewritten the encrypted item's ID to encoded form
        encoded_item_id = _extract_encoded_item_id(first_response)
        assert encoded_item_id.startswith("encitem_"), (
            f"Expected output item ID to be rewritten to encitem_... but got {encoded_item_id!r}"
        )

        # Verify the encoded ID decodes back to the correct deployment + original ID
        decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(encoded_item_id)
        assert decoded is not None
        assert decoded["model_id"] == first_model_id
        assert decoded["item_id"] == "rs_encrypted_item_456"

        # Second request: use the encoded item IDs from the first response
        second_response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input=[
                {"type": "message", "id": "msg_abc123", "role": "assistant"},
                {
                    "type": "reasoning",
                    "id": encoded_item_id,
                    "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
                },
            ],
        )
        second_model_id = second_response._hidden_params["model_id"]

        assert second_model_id == first_model_id, (
            f"Expected affinity to route to {first_model_id}, but got {second_model_id}"
        )


@pytest.mark.asyncio
async def test_encrypted_content_affinity_no_effect_on_chat_completions():
    """
    Encrypted content affinity should not affect regular chat completions.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "mock_response": "Hello from chat completion!",
                },
                "model_info": {"id": "chat-deployment-1"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
    )

    response1 = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
    )
    response2 = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello again"}],
    )
    assert response1.id is not None
    assert response2.id is not None


@pytest.mark.asyncio
async def test_encrypted_content_affinity_bypasses_rpm_limits():
    """
    When encrypted content affinity pins to a deployment, RPM limits are bypassed
    since the request would fail on any other deployment anyway.
    """
    mock_response_data = {
        "id": "resp_mock-rpm-test",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "openai/gpt-5.1-codex",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_encrypted_must_pin",
                "status": "completed",
                "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
        "error": None,
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-1",
                    "rpm": 1,  # Very low limit
                },
                "model_info": {"id": "rpm-limited-deployment"},
            },
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-2",
                    "rpm": 100,
                },
                "model_info": {"id": "high-rpm-deployment"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
        routing_strategy="usage-based-routing-v2",
    )

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(mock_response_data, 200)

        first_response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input="Initial request",
        )
        first_model_id = first_response._hidden_params["model_id"]

        # Extract encoded item ID from the first response output
        encoded_item_id = _extract_encoded_item_id(first_response)
        assert encoded_item_id.startswith("encitem_"), (
            f"Expected encitem_... but got {encoded_item_id!r}"
        )

        # Follow-up with the encoded item ID — should pin to same deployment
        # even if it is at its RPM limit
        second_response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input=[
                {
                    "type": "reasoning",
                    "id": encoded_item_id,
                    "encrypted_content": "gAAAAABpnW_yEYmSNEyOG...",
                },
            ],
        )
        second_model_id = second_response._hidden_params["model_id"]

        assert second_model_id == first_model_id


@pytest.mark.asyncio
async def test_encrypted_content_affinity_no_match_normal_routing():
    """
    Input items with non-encoded IDs (no encitem_ prefix) fall through to
    normal load balancing.
    """
    mock_response_data = {
        "id": "resp_mock-no-match",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "openai/gpt-5.1-codex",
        "output": [
            {
                "type": "message",
                "id": "msg_new",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Response"}],
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
        "error": None,
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-1",
                },
                "model_info": {"id": "deployment-a"},
            },
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-2",
                },
                "model_info": {"id": "deployment-b"},
            },
        ],
        optional_pre_call_checks=["encrypted_content_affinity"],
    )

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(mock_response_data, 200)

        # Non-encoded item ID — no affinity should kick in
        response = await router.aresponses(
            model="openai.gpt-5.1-codex",
            input=[
                {"type": "message", "id": "unknown_item_id_12345"},
            ],
        )
        assert response.id is not None
