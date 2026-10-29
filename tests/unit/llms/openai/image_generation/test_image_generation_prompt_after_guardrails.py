"""
Regression for #46473: pre-call PII masking on /v1/images/generations.

The endpoint copies prompt -> messages for guardrails, then used to rebuild
prompt from messages. Image handlers write the redacted text to data["prompt"]
and left messages alone, so the rebuild restored the original PII.
"""

from typing import Any

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.openai.image_generation.guardrail_translation.handler import (
    OpenAIImageGenerationHandler,
)
from litellm.proxy.image_endpoints.endpoints import (
    resolve_image_generation_prompt_after_guardrails,
)


def test_prefers_guardrail_updated_prompt_over_unchanged_messages():
    original = "Draw a portrait of SSN 123-45-6789"
    redacted = "Draw a portrait of SSN <US_SSN>"
    data = {
        "prompt": redacted,
        "messages": [{"role": "user", "content": original}],
    }

    resolve_image_generation_prompt_after_guardrails(
        data, prompt_before_hook=original
    )

    assert data["prompt"] == redacted
    assert "messages" not in data


def test_rebuilds_from_messages_when_prompt_unchanged():
    original = "A red balloon"
    redacted_via_messages = "A <COLOR> balloon"
    data = {
        "prompt": original,
        "messages": [{"role": "user", "content": redacted_via_messages}],
    }

    resolve_image_generation_prompt_after_guardrails(
        data, prompt_before_hook=original
    )

    assert data["prompt"] == redacted_via_messages
    assert "messages" not in data


def test_leaves_prompt_when_no_messages():
    data = {"prompt": "unchanged"}
    resolve_image_generation_prompt_after_guardrails(
        data, prompt_before_hook="unchanged"
    )
    assert data["prompt"] == "unchanged"
    assert "messages" not in data


@pytest.mark.asyncio
async def test_handler_syncs_temporary_messages_with_redacted_prompt():
    class AppendTag(CustomGuardrail):
        async def apply_guardrail(
            self, inputs: dict, request_data: dict, input_type: str, **kwargs: Any
        ) -> dict:
            return {"texts": [f"{t} [MASKED]" for t in inputs.get("texts", [])]}

    original = "secret@example.com in a landscape"
    data = {
        "model": "dall-e-3",
        "prompt": original,
        "messages": [{"role": "user", "content": original}],
    }
    result = await OpenAIImageGenerationHandler().process_input_messages(
        data, AppendTag(guardrail_name="tag")
    )
    assert result["prompt"] == f"{original} [MASKED]"
    assert result["messages"][0]["content"] == result["prompt"]


@pytest.mark.asyncio
async def test_endpoint_rebuild_keeps_handler_redaction():
    """Exact #46473 sequence: handler redacts prompt; rebuild must not restore PII."""

    class MaskSSN(CustomGuardrail):
        async def apply_guardrail(
            self, inputs: dict, request_data: dict, input_type: str, **kwargs: Any
        ) -> dict:
            texts = inputs.get("texts", [])
            return {
                "texts": [
                    t.replace("219-09-9999", "<US_SSN>").replace("Jane Doe", "<PERSON>")
                    for t in texts
                ]
            }

    original = "Generate an ID badge for Jane Doe, SSN 219-09-9999"
    data = {
        "model": "dall-e-3",
        "prompt": original,
        "messages": [{"role": "user", "content": original}],
    }

    data = await OpenAIImageGenerationHandler().process_input_messages(
        data, MaskSSN(guardrail_name="mask")
    )
    assert "219-09-9999" not in data["prompt"]
    assert "Jane Doe" not in data["prompt"]

    # Even if messages were somehow still original, prefer updated prompt.
    data["messages"] = [{"role": "user", "content": original}]
    resolve_image_generation_prompt_after_guardrails(
        data, prompt_before_hook=original
    )
    assert "219-09-9999" not in data["prompt"]
    assert "Jane Doe" not in data["prompt"]
    assert "<US_SSN>" in data["prompt"]
    assert "<PERSON>" in data["prompt"]
