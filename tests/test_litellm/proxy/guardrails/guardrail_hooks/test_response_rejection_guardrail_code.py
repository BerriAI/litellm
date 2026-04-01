"""Tests for the response-rejection custom guardrail code (input_type response, block on refusal)."""

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.custom_code import (
    RESPONSE_REJECTION_GUARDRAIL_CODE, CustomCodeGuardrail)


@pytest.fixture
def response_rejection_guardrail():
    """Guardrail instance using the response-rejection custom code."""
    return CustomCodeGuardrail(
        guardrail_name="response_rejection",
        custom_code=RESPONSE_REJECTION_GUARDRAIL_CODE,
    )


@pytest.mark.asyncio
async def test_response_rejection_allows_request_input_type(response_rejection_guardrail):
    """Should allow when input_type is 'request' (no response check)."""
    result = await response_rejection_guardrail.apply_guardrail(
        inputs={"texts": ["some user message"]},
        request_data={},
        input_type="request",
    )
    assert result == {"texts": ["some user message"]}


@pytest.mark.asyncio
async def test_response_rejection_allows_helpful_response(response_rejection_guardrail):
    """Should allow when response text does not contain rejection phrases."""
    result = await response_rejection_guardrail.apply_guardrail(
        inputs={"texts": ["Here is how you can do that: step 1, step 2."]},
        request_data={},
        input_type="response",
    )
    assert result["texts"] == ["Here is how you can do that: step 1, step 2."]


@pytest.mark.asyncio
async def test_response_rejection_blocks_refusal_phrase(response_rejection_guardrail):
    """Should block when response contains a known rejection phrase."""
    with pytest.raises(HTTPException) as exc_info:
        await response_rejection_guardrail.apply_guardrail(
            inputs={"texts": ["That's not something I can help with."]},
            request_data={},
            input_type="response",
        )
    assert exc_info.value.status_code == 400
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    assert "error" in detail
    assert "rejected" in detail["error"].lower() or "reject" in detail["error"].lower()
    assert detail.get("guardrail") == "response_rejection"
    assert detail.get("detection_info", {}).get("matched_phrase") is not None


@pytest.mark.asyncio
async def test_response_rejection_blocks_case_insensitive(response_rejection_guardrail):
    """Should block on refusal phrase regardless of case."""
    with pytest.raises(HTTPException):
        await response_rejection_guardrail.apply_guardrail(
            inputs={"texts": ["I'M SORRY, I CAN'T do that."]},
            request_data={},
            input_type="response",
        )


@pytest.mark.asyncio
async def test_response_rejection_empty_texts_allowed(response_rejection_guardrail):
    """Should allow when texts is empty or missing."""
    result = await response_rejection_guardrail.apply_guardrail(
        inputs={},
        request_data={},
        input_type="response",
    )
    assert result == {}
