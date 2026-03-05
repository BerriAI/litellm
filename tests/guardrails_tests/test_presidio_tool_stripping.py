"""
Regression test for: fix(presidio): strip OpenAI-converted keys before returning
from apply_guardrail.

Root cause
----------
AnthropicMessagesHandler.process_input_messages() converts the Anthropic request
to OpenAI format and puts the converted tools into the GenericGuardrailAPIInputs
dict before calling apply_guardrail():

    inputs["tools"]              = [{"type": "function", ...}]   # OpenAI format
    inputs["structured_messages"] = [...]
    inputs["model"]              = "anthropic/..."
    inputs["images"]             = [...]

After apply_guardrail() returns, the handler applies the result back:

    guardrailed_tools = guardrailed_inputs.get("tools")
    if guardrailed_tools is not None:
        data["tools"] = guardrailed_tools          # overwrites native Anthropic tools!

If apply_guardrail() returns those keys unchanged, the caller overwrites the
native Anthropic tool definitions (bash_20250124, text_editor_20250124, …) with
the OpenAI type:"function" format, triggering a 400 from the Anthropic API:

    Input tag 'function' found using 'type' does not match expected tags
    'bash_20250124', 'text_editor_20250124', ...

Fix
---
apply_guardrail() now strips ("tools", "structured_messages", "model", "images")
from inputs before returning — Presidio only processes texts, so these keys carry
no useful return value.
"""
import os
import sys

import pytest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking


@pytest.mark.asyncio
async def test_apply_guardrail_strips_openai_converted_keys() -> None:
    """apply_guardrail() must not return keys added during OpenAI-format conversion.

    Specifically: 'tools', 'structured_messages', 'model', and 'images' are added
    by AnthropicMessagesHandler (and OpenAIChatCompletionsHandler) before calling
    apply_guardrail().  Presidio does not modify these keys, so returning them
    causes callers to blindly overwrite the original request data with the
    OpenAI-converted versions, breaking the Anthropic native API path.
    """
    guardrail = _OPTIONAL_PresidioPIIMasking(
        pii_entities_config={},
        presidio_analyzer_api_base="http://mock-analyzer",
        presidio_anonymizer_api_base="http://mock-anonymizer",
    )

    # Simulate the inputs dict that AnthropicMessagesHandler constructs and passes
    # to apply_guardrail() — mirrors lines 108-118 of anthropic/chat/guardrail_translation/handler.py.
    openai_format_tool = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    inputs = {
        "texts": ["Hello, my name is John Smith"],
        "tools": [openai_format_tool],
        "structured_messages": [{"role": "user", "content": "Hello, my name is John Smith"}],
        "model": "anthropic/claude-3-5-sonnet-20241022",
        "images": [],
    }

    # Patch check_pii to avoid a live Presidio instance; return text unchanged.
    with patch.object(
        guardrail,
        "check_pii",
        new=AsyncMock(return_value="Hello, my name is <PERSON>"),
    ):
        result = await guardrail.apply_guardrail(
            inputs=inputs,  # type: ignore[arg-type]
            request_data={},
            input_type="request",
        )

    # "texts" must be present and processed.
    assert "texts" in result
    assert result["texts"] == ["Hello, my name is <PERSON>"]

    # The OpenAI-conversion keys must NOT appear in the return value.
    # If they did, callers like AnthropicMessagesHandler would write:
    #   data["tools"] = guardrailed_inputs["tools"]   # OpenAI format!
    # overwriting the native Anthropic tools and triggering a 400.
    assert "tools" not in result, (
        "apply_guardrail() returned 'tools' — this would overwrite native "
        "Anthropic tool definitions (bash_20250124 etc.) with OpenAI type:'function' format"
    )
    assert "structured_messages" not in result, (
        "apply_guardrail() returned 'structured_messages' — unexpected side-effect key"
    )
    assert "model" not in result, (
        "apply_guardrail() returned 'model' — unexpected side-effect key"
    )
    assert "images" not in result, (
        "apply_guardrail() returned 'images' — unexpected side-effect key"
    )
