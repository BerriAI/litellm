"""
Tests for fix/guardrail-request-data-v2

Verifies that:
1. BaseTranslation declares request_data on output handler signatures.
2. All concrete handlers accept request_data (type-consistent with interface).
3. Handlers that call apply_guardrail() forward request_data values.

Related issue: https://github.com/BerriAI/litellm/issues/22821
"""

import asyncio
import importlib
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.llms.base_llm.guardrail_translation.base_translation import (
    BaseTranslation,
)


# ---------------------------------------------------------------------------
# Signature conformance tests
# ---------------------------------------------------------------------------


def test_base_translation_process_output_response_has_request_data_param():
    """Abstract base declares request_data so implementations stay consistent."""
    sig = inspect.signature(BaseTranslation.process_output_response)
    assert "request_data" in sig.parameters, (
        "BaseTranslation.process_output_response() must declare 'request_data'"
    )
    param = sig.parameters["request_data"]
    assert param.default is None, "'request_data' must default to None"


def test_base_translation_process_output_streaming_has_request_data_param():
    """Streaming variant also declares request_data."""
    sig = inspect.signature(BaseTranslation.process_output_streaming_response)
    assert "request_data" in sig.parameters, (
        "BaseTranslation.process_output_streaming_response() must declare 'request_data'"
    )
    param = sig.parameters["request_data"]
    assert param.default is None, "'request_data' must default to None"


def test_all_handler_implementations_accept_request_data():
    """Verify concrete handlers accept request_data on process_output_response."""
    handler_modules = [
        "litellm.llms.openai.chat.guardrail_translation.handler",
        "litellm.llms.openai.completion.guardrail_translation.handler",
        "litellm.llms.openai.responses.guardrail_translation.handler",
        "litellm.llms.openai.transcriptions.guardrail_translation.handler",
        "litellm.llms.openai.embeddings.guardrail_translation.handler",
        "litellm.llms.openai.image_generation.guardrail_translation.handler",
        "litellm.llms.openai.speech.guardrail_translation.handler",
        "litellm.llms.anthropic.chat.guardrail_translation.handler",
        "litellm.llms.pass_through.guardrail_translation.handler",
        "litellm.llms.a2a.chat.guardrail_translation.handler",
        "litellm.llms.cohere.rerank.guardrail_translation.handler",
    ]

    validated = 0
    for module_path in handler_modules:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            continue

        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj is BaseTranslation or obj.__module__ != module_path:
                continue
            if not hasattr(obj, "process_output_response"):
                continue

            sig = inspect.signature(obj.process_output_response)
            assert "request_data" in sig.parameters, (
                f"{module_path}.{_name}.process_output_response() "
                f"must accept 'request_data'"
            )
            validated += 1

    assert validated > 0, "No handler classes validated — all imports failed"


# ---------------------------------------------------------------------------
# Behavioral tests: verify request_data reaches apply_guardrail()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_handler_forwards_request_data():
    """Sentinel pii_tokens in request_data must reach apply_guardrail()."""
    from litellm.llms.openai.chat.guardrail_translation.handler import (
        OpenAIChatCompletionsHandler,
    )
    from litellm.types.utils import Choices, Message, ModelResponse

    handler = OpenAIChatCompletionsHandler()

    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                message=Message(content="Hello world", role="assistant"),
                finish_reason="stop",
            )
        ],
        model="gpt-4",
    )

    sentinel = {"pii_tokens": {"<NAME_1>": "Alice"}}
    captured = {}

    async def _capture(inputs, request_data, input_type, logging_obj=None):
        captured.update(request_data)
        return inputs

    guardrail = MagicMock()
    guardrail.guardrail_name = "test-guardrail"
    guardrail.apply_guardrail = AsyncMock(side_effect=_capture)

    await handler.process_output_response(
        response=response,
        guardrail_to_apply=guardrail,
        request_data=sentinel,
    )

    assert "pii_tokens" in captured, (
        "request_data with pii_tokens was not forwarded to apply_guardrail()"
    )
    assert captured["pii_tokens"] == {"<NAME_1>": "Alice"}


@pytest.mark.asyncio
async def test_anthropic_handler_merges_request_data():
    """Anthropic handler must merge caller's request_data, not shadow it."""
    from litellm.llms.anthropic.chat.guardrail_translation.handler import (
        AnthropicMessagesHandler,
    )

    handler = AnthropicMessagesHandler()

    # Minimal Anthropic-style response
    response = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello"}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    sentinel = {"pii_tokens": {"<NAME_1>": "Bob"}}
    captured = {}

    async def _capture(inputs, request_data, input_type, logging_obj=None):
        captured.update(request_data)
        return inputs

    guardrail = MagicMock()
    guardrail.guardrail_name = "test-guardrail"
    guardrail.apply_guardrail = AsyncMock(side_effect=_capture)

    await handler.process_output_response(
        response=response,
        guardrail_to_apply=guardrail,
        request_data=sentinel,
    )

    assert "pii_tokens" in captured, "Caller's pii_tokens were shadowed"
    assert "response" in captured, "Response must be nested under 'response' key"


@pytest.mark.asyncio
async def test_pass_through_handler_merges_request_data():
    """Pass-through handler must merge, not shadow, caller's request_data."""
    from litellm.llms.pass_through.guardrail_translation.handler import (
        PassThroughEndpointHandler,
    )

    handler = PassThroughEndpointHandler()

    response = {"result": "ok"}
    sentinel = {"pii_tokens": {"<NAME_1>": "Charlie"}}
    captured = {}

    async def _capture(inputs, request_data, input_type, logging_obj=None):
        captured.update(request_data)
        return inputs

    guardrail = MagicMock()
    guardrail.guardrail_name = "test-guardrail"
    guardrail.apply_guardrail = AsyncMock(side_effect=_capture)
    guardrail.get_guardrail_from_metadata = MagicMock(return_value=None)

    await handler.process_output_response(
        response=response,
        guardrail_to_apply=guardrail,
        request_data=sentinel,
    )

    assert "pii_tokens" in captured, "Caller's pii_tokens were shadowed"
