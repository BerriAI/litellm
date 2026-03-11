"""
Tests for fix/guardrail-request-data-passthrough

Verifies that:
1. BaseTranslation.process_output_response() declares request_data
   so implementing classes are type-consistent with the abstract interface.
2. Concrete handlers actually forward request_data values to apply_guardrail().

Related issue: https://github.com/BerriAI/litellm/issues/22821
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation


def test_base_translation_process_output_response_has_request_data_param():
    """
    All concrete implementations of process_output_response() accept a
    request_data keyword argument, but the abstract base did not declare it.
    This test ensures the abstract signature matches the implementations.
    """
    sig = inspect.signature(BaseTranslation.process_output_response)
    assert "request_data" in sig.parameters, (
        "BaseTranslation.process_output_response() must declare a 'request_data' "
        "parameter so all implementations are type-consistent with the interface."
    )
    param = sig.parameters["request_data"]
    assert param.default is None, (
        "'request_data' must default to None for backwards compatibility."
    )


def test_all_handler_implementations_accept_request_data():
    """
    Verify a representative set of concrete handler implementations also
    accept request_data (ensuring they conform to the updated abstract sig).
    """
    import importlib

    handlers = [
        "litellm.llms.openai.chat.guardrail_translation.handler",
        "litellm.llms.anthropic.chat.guardrail_translation.handler",
        "litellm.llms.openai.completion.guardrail_translation.handler",
        "litellm.llms.pass_through.guardrail_translation.handler",
        "litellm.llms.mistral.ocr.guardrail_translation.handler",
    ]

    validated_count = 0
    for module_path in handlers:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            continue  # module not installed in this environment; check next handler

        # Find the class that implements process_output_response
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if not hasattr(obj, "process_output_response"):
                continue
            # Skip the abstract base itself
            if obj is BaseTranslation:
                continue
            # Only check classes from this module
            if obj.__module__ != module_path:
                continue

            sig = inspect.signature(obj.process_output_response)
            assert "request_data" in sig.parameters, (
                f"{module_path}.{name}.process_output_response() "
                f"must accept 'request_data' to match BaseTranslation."
            )
            validated_count += 1

    assert validated_count > 0, (
        "No handler classes were validated — all imports failed or no matching classes found."
    )


def test_openai_handler_forwards_request_data_to_apply_guardrail():
    """
    Behavioural test: instantiate the OpenAI chat handler, call
    process_output_response with a sentinel value in request_data, and
    verify that the sentinel reaches apply_guardrail().
    """
    from litellm.llms.openai.chat.guardrail_translation.handler import (
        OpenAIChatCompletionsHandler,
    )
    from litellm.types.utils import Choices, Message, ModelResponse

    handler = OpenAIChatCompletionsHandler()

    # Build a real ModelResponse so _has_text_content passes
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

    # Sentinel value simulating PII tokens stored during input masking
    sentinel = {"pii_tokens": {"[NAME_1]": "Alice"}}

    # Capture what apply_guardrail receives
    captured_request_data = {}

    async def _capture_apply_guardrail(inputs, request_data, input_type, logging_obj=None):
        captured_request_data.update(request_data)
        return inputs  # pass through unchanged

    guardrail = MagicMock()
    guardrail.guardrail_name = "test-guardrail"
    guardrail.apply_guardrail = AsyncMock(side_effect=_capture_apply_guardrail)

    asyncio.get_event_loop().run_until_complete(
        handler.process_output_response(
            response=response,
            guardrail_to_apply=guardrail,
            request_data=sentinel,
        )
    )

    # The sentinel PII tokens must be visible inside apply_guardrail's request_data
    assert "pii_tokens" in captured_request_data, (
        "request_data with pii_tokens was not forwarded to apply_guardrail()"
    )
    assert captured_request_data["pii_tokens"] == {"[NAME_1]": "Alice"}
