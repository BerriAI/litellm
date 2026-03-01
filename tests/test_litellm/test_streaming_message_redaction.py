"""
Tests for streaming message redaction via StandardLoggingPayload.

Verifies that when litellm.turn_off_message_logging=True (global setting),
the StandardLoggingPayload's 'messages' and 'response' fields are redacted
for ALL callbacks — not just those with per-callback turn_off_message_logging.

Fixes: https://github.com/BerriAI/litellm/issues/9664
"""

import os
import sys
from copy import deepcopy
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)

import litellm
from litellm import Choices, Message, ModelResponse
from litellm.integrations.custom_logger import CustomLogger


def _make_model_call_details(
    messages=None, response=None, stream=False, dynamic_turn_off=None
):
    """Build a model_call_details dict with StandardLoggingPayload."""
    if messages is None:
        messages = [{"role": "user", "content": "secret question"}]
    if response is None:
        response = ModelResponse(
            choices=[Choices(message=Message(content="secret answer"))]
        ).model_dump()

    standard_logging_object = {
        "messages": messages,
        "response": response,
        "call_type": "completion",
        "model": "gpt-5",
        "stream": stream,
        "prompt_tokens": 10,
        "completion_tokens": 20,
    }

    details = {
        "standard_logging_object": standard_logging_object,
        "litellm_params": {"metadata": {}},
        "stream": stream,
    }

    if dynamic_turn_off is not None:
        details["standard_callback_dynamic_params"] = {
            "turn_off_message_logging": dynamic_turn_off
        }

    return details


class TestGlobalTurnOffMessageLogging:
    """Global litellm.turn_off_message_logging should redact StandardLoggingPayload."""

    def test_global_setting_redacts_messages(self):
        """When litellm.turn_off_message_logging=True, messages must be redacted."""
        callback = CustomLogger()
        details = _make_model_call_details()

        with patch.object(litellm, "turn_off_message_logging", True):
            result = callback.redact_standard_logging_payload_from_model_call_details(
                model_call_details=details
            )

        slo = result["standard_logging_object"]
        assert slo["messages"][0]["content"] == "redacted-by-litellm"

    def test_global_setting_redacts_response(self):
        """When litellm.turn_off_message_logging=True, response must be redacted."""
        callback = CustomLogger()
        details = _make_model_call_details()

        with patch.object(litellm, "turn_off_message_logging", True):
            result = callback.redact_standard_logging_payload_from_model_call_details(
                model_call_details=details
            )

        slo = result["standard_logging_object"]
        response = slo["response"]
        assert response["choices"][0]["message"]["content"] == "redacted-by-litellm"

    def test_global_setting_redacts_streaming_response(self):
        """Streaming responses must also be redacted when global flag is set."""
        callback = CustomLogger()
        details = _make_model_call_details(stream=True)

        with patch.object(litellm, "turn_off_message_logging", True):
            result = callback.redact_standard_logging_payload_from_model_call_details(
                model_call_details=details
            )

        slo = result["standard_logging_object"]
        assert slo["messages"][0]["content"] == "redacted-by-litellm"
        response = slo["response"]
        assert response["choices"][0]["message"]["content"] == "redacted-by-litellm"

    def test_no_redaction_when_global_off(self):
        """When litellm.turn_off_message_logging=False, messages/response untouched."""
        callback = CustomLogger()
        details = _make_model_call_details()

        with patch.object(litellm, "turn_off_message_logging", False):
            result = callback.redact_standard_logging_payload_from_model_call_details(
                model_call_details=details
            )

        slo = result["standard_logging_object"]
        assert slo["messages"][0]["content"] == "secret question"


class TestPerCallbackTurnOffStillWorks:
    """Per-callback turn_off_message_logging must still work independently."""

    def test_per_callback_redacts_without_global(self):
        """Callback with turn_off_message_logging=True redacts even when global is False."""
        callback = CustomLogger(turn_off_message_logging=True)
        details = _make_model_call_details()

        with patch.object(litellm, "turn_off_message_logging", False):
            result = callback.redact_standard_logging_payload_from_model_call_details(
                model_call_details=details
            )

        slo = result["standard_logging_object"]
        assert slo["messages"][0]["content"] == "redacted-by-litellm"


class TestDynamicParamRedaction:
    """Dynamic per-request turn_off_message_logging should redact StandardLoggingPayload."""

    def test_dynamic_param_redacts_standard_logging_payload(self):
        """Per-request dynamic param should trigger redaction."""
        callback = CustomLogger()
        details = _make_model_call_details(dynamic_turn_off=True)

        with patch.object(litellm, "turn_off_message_logging", False):
            result = callback.redact_standard_logging_payload_from_model_call_details(
                model_call_details=details
            )

        slo = result["standard_logging_object"]
        assert slo["messages"][0]["content"] == "redacted-by-litellm"

    def test_dynamic_param_false_no_redaction(self):
        """Per-request dynamic param False should NOT redact."""
        callback = CustomLogger()
        details = _make_model_call_details(dynamic_turn_off=False)

        with patch.object(litellm, "turn_off_message_logging", False):
            result = callback.redact_standard_logging_payload_from_model_call_details(
                model_call_details=details
            )

        slo = result["standard_logging_object"]
        assert slo["messages"][0]["content"] == "secret question"


class TestOriginalNotMutated:
    """Redaction must not mutate the original model_call_details."""

    def test_original_standard_logging_preserved(self):
        """The original standard_logging_object must not be modified."""
        callback = CustomLogger()
        details = _make_model_call_details()
        original_messages = deepcopy(details["standard_logging_object"]["messages"])

        with patch.object(litellm, "turn_off_message_logging", True):
            callback.redact_standard_logging_payload_from_model_call_details(
                model_call_details=details
            )

        assert details["standard_logging_object"]["messages"] == original_messages


class TestResponsesAPIRedaction:
    """ResponsesAPI format (output field) should also be redacted."""

    def test_responses_api_output_redacted(self):
        """ResponsesAPI-style response with output array must be redacted."""
        callback = CustomLogger()
        response = {
            "output": [
                {
                    "content": [
                        {"text": "secret output", "type": "text"},
                    ],
                    "type": "message",
                }
            ]
        }
        details = _make_model_call_details(response=response)

        with patch.object(litellm, "turn_off_message_logging", True):
            result = callback.redact_standard_logging_payload_from_model_call_details(
                model_call_details=details
            )

        slo = result["standard_logging_object"]
        output_text = slo["response"]["output"][0]["content"][0]["text"]
        assert output_text == "redacted-by-litellm"
