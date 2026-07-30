"""
Unit tests for A2A Protocol Guardrail Translation Handler

Regression coverage for the "data"-kind part guardrail bypass: A2A responses
can carry structured content in `kind: "data"` parts, which
`extract_text_from_a2a_message` (used to build the completion text callers
see) folds into the final text, but the guardrail handler previously only
inspected `kind: "text"` parts, so guarded output checks were skipped for
that content path.
"""

import os
import sys
from typing import Any, Literal, Optional

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.a2a.chat.guardrail_translation.handler import A2AGuardrailHandler
from litellm.types.utils import GenericGuardrailAPIInputs


class MockGuardrail(CustomGuardrail):
    """Mock guardrail that uppercases text so we can assert exactly what was scanned and where the result landed."""

    def __init__(self, guardrail_name: str = "test"):
        super().__init__(guardrail_name=guardrail_name)
        self.last_inputs: Optional[GenericGuardrailAPIInputs] = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.last_inputs = inputs
        texts = inputs.get("texts", [])
        return {"texts": [text.upper() for text in texts]}


@pytest.mark.asyncio
async def test_process_output_response_scans_data_parts():
    """A `kind: data` part in the output must be sent to the guardrail and the
    guardrailed value written back into `data`, not silently skipped."""
    handler = A2AGuardrailHandler()
    guardrail = MockGuardrail()

    response = {
        "result": {
            "kind": "message",
            "parts": [
                {"kind": "text", "text": "hello"},
                {"kind": "data", "data": {"secret": "leak-me"}},
            ],
        }
    }

    result = await handler.process_output_response(
        response=response,
        guardrail_to_apply=guardrail,
    )

    # The data part's serialized content must have reached the guardrail.
    assert guardrail.last_inputs is not None
    scanned_texts = guardrail.last_inputs["texts"]
    assert any("leak-me" in t for t in scanned_texts)

    # The guardrailed (uppercased) value must be written back into "data",
    # and the part must remain a "data" part, not be silently dropped or
    # converted into an unguarded pass-through.
    data_part = result["result"]["parts"][1]
    assert data_part["kind"] == "data"
    assert "LEAK-ME" in data_part["data"]

    # The text part must still be guardrailed as before (no regression).
    text_part = result["result"]["parts"][0]
    assert text_part["text"] == "HELLO"


@pytest.mark.asyncio
async def test_process_output_response_data_only_still_scanned():
    """A response with ONLY a data part (no text parts at all) must not be
    skipped as "no text content in response"."""
    handler = A2AGuardrailHandler()
    guardrail = MockGuardrail()

    response = {
        "result": {
            "kind": "message",
            "parts": [{"kind": "data", "data": {"result": {"msg": "pong"}}}],
        }
    }

    result = await handler.process_output_response(
        response=response,
        guardrail_to_apply=guardrail,
    )

    assert guardrail.last_inputs is not None
    assert guardrail.last_inputs["texts"]
    assert "PONG" in result["result"]["parts"][0]["data"]


@pytest.mark.asyncio
async def test_process_output_response_scans_nested_parts():
    """A part that itself carries a nested "parts" list (grouping sub-parts)
    must be recursed into, matching extract_text_from_a2a_message's own
    recursion, instead of being silently skipped as neither text nor data."""
    handler = A2AGuardrailHandler()
    guardrail = MockGuardrail()

    response = {
        "result": {
            "kind": "message",
            "parts": [
                {
                    "kind": "group",
                    "parts": [
                        {"kind": "text", "text": "hello"},
                        {"kind": "data", "data": {"secret": "leak-me"}},
                    ],
                },
            ],
        }
    }

    result = await handler.process_output_response(
        response=response,
        guardrail_to_apply=guardrail,
    )

    assert guardrail.last_inputs is not None
    scanned_texts = guardrail.last_inputs["texts"]
    assert "hello" in scanned_texts
    assert any("leak-me" in t for t in scanned_texts)

    nested_parts = result["result"]["parts"][0]["parts"]
    assert nested_parts[0]["text"] == "HELLO"
    assert "LEAK-ME" in nested_parts[1]["data"]


@pytest.mark.asyncio
async def test_process_input_messages_scans_data_parts():
    """The same bypass existed on the request/input side of the handler."""
    handler = A2AGuardrailHandler()
    guardrail = MockGuardrail()

    data = {
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "parts": [{"kind": "data", "data": {"secret": "leak-me"}}],
            }
        }
    }

    result = await handler.process_input_messages(
        data=data,
        guardrail_to_apply=guardrail,
    )

    assert guardrail.last_inputs is not None
    assert any("leak-me" in t for t in guardrail.last_inputs["texts"])

    data_part = result["params"]["message"]["parts"][0]
    assert data_part["kind"] == "data"
    assert "LEAK-ME" in data_part["data"]
