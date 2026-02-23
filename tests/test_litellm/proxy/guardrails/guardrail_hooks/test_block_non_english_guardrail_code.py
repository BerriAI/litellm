"""Tests for the block-non-English pre-built guardrail code."""

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.custom_code import (
    BLOCK_NON_ENGLISH_GUARDRAIL_CODE,
    CustomCodeGuardrail,
)


@pytest.fixture
def block_non_english_guardrail():
    """Guardrail instance using the block-non-English custom code."""
    return CustomCodeGuardrail(
        guardrail_name="block_non_english",
        custom_code=BLOCK_NON_ENGLISH_GUARDRAIL_CODE,
    )


@pytest.mark.asyncio
async def test_block_non_english_allows_request_input_type(
    block_non_english_guardrail,
):
    """Should allow when input_type is 'request' (no response check)."""
    result = await block_non_english_guardrail.apply_guardrail(
        inputs={"texts": ["some user message"]},
        request_data={},
        input_type="request",
    )
    assert result == {"texts": ["some user message"]}


@pytest.mark.asyncio
async def test_block_non_english_allows_english_response(block_non_english_guardrail):
    """Should allow when response text is English and append detection to metadata."""
    request_data = {"metadata": {}}
    result = await block_non_english_guardrail.apply_guardrail(
        inputs={"texts": ["This is a clear English sentence."]},
        request_data=request_data,
        input_type="response",
    )
    assert result["texts"] == ["This is a clear English sentence."]
    detections = request_data.get("metadata", {}).get("detections", [])
    assert len(detections) == 1
    assert detections[0]["detection_info"].get("is_english") is True
    assert detections[0]["detection_info"].get("language") == "en"


@pytest.mark.asyncio
async def test_block_non_english_blocks_non_english_response(
    block_non_english_guardrail,
):
    """Should block when response contains clear non-English text."""
    with pytest.raises(HTTPException) as exc_info:
        await block_non_english_guardrail.apply_guardrail(
            inputs={"texts": ["Esta es una oración en español."]},
            request_data={},
            input_type="response",
        )
    assert exc_info.value.status_code == 400
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    assert "not in English" in detail.get("error", "")
    assert detail.get("guardrail") == "block_non_english"
    assert "language" in detail.get("detection_info", {})


@pytest.mark.asyncio
async def test_block_non_english_empty_texts_allowed(block_non_english_guardrail):
    """Should allow when texts is empty or missing."""
    result = await block_non_english_guardrail.apply_guardrail(
        inputs={},
        request_data={},
        input_type="response",
    )
    assert result == {}
