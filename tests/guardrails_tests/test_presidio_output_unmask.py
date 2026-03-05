"""
Regression test for: fix(presidio): add output unmask path when output_parse_pii
is enabled.

Root cause
----------
apply_guardrail() always runs check_pii() (masking) regardless of whether it is
processing an input (request) or an output (response).  With output_parse_pii=True:

  - Input phase (input_type="request"):  PII is masked; tokens stored in
    request_data["pii_tokens"].
  - Output phase (input_type="response"): apply_guardrail() should RESTORE the
    original PII by replacing tokens with their original values.

Without the fix, apply_guardrail() calls check_pii() on the output as well —
which re-masks the already-masked response (or leaves it with <PERSON> tokens
unchanged because there is no fresh PII to detect), so the caller never sees the
original values.

Fix
---
apply_guardrail() now detects input_type=="response" && output_parse_pii and
takes an early-return path that calls _unmask_pii_text() for each text in
inputs["texts"], restoring the tokens using the pii_tokens dict from request_data.

Dependency
----------
Receiving pii_tokens in request_data requires the unified guardrail to forward
request_data to process_output_response() — which is what
fix/guardrail-request-data-passthrough (PR 1, BerriAI/litellm#22821) addresses.
Without PR 1, request_data will be empty and unmasking silently no-ops.  The two
fixes together form the complete end-to-end unmask pipeline.
"""
import os
import sys

import pytest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_guardrail(output_parse_pii: bool = True) -> _OPTIONAL_PresidioPIIMasking:
    return _OPTIONAL_PresidioPIIMasking(
        pii_entities_config={},
        output_parse_pii=output_parse_pii,
        presidio_analyzer_api_base="http://mock-analyzer",
        presidio_anonymizer_api_base="http://mock-anonymizer",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_unmasks_response_when_output_parse_pii_enabled() -> None:
    """With output_parse_pii=True and input_type='response', tokens are restored.

    Simulates the output phase of a round-trip:
      1. Input masked "John Smith" → "<PERSON>_abc123456789"
      2. LLM echoes the token back: "Hello, <PERSON>_abc123456789"
      3. apply_guardrail() with input_type="response" should restore the original
         name, returning "Hello, John Smith".
    """
    guardrail = _make_guardrail(output_parse_pii=True)

    token = "<PERSON>_abc123456789"
    pii_tokens = {token: "John Smith"}

    # request_data carries the token→original mapping from the input phase.
    request_data = {"pii_tokens": pii_tokens}

    inputs = {"texts": [f"Hello, {token}. How are you today?"]}

    result = await guardrail.apply_guardrail(
        inputs=inputs,  # type: ignore[arg-type]
        request_data=request_data,
        input_type="response",
    )

    assert result["texts"] == ["Hello, John Smith. How are you today?"], (
        "apply_guardrail() must restore PII tokens in response text"
    )


@pytest.mark.asyncio
async def test_apply_guardrail_response_noop_when_no_pii_tokens() -> None:
    """With output_parse_pii=True but empty pii_tokens, text is returned unchanged."""
    guardrail = _make_guardrail(output_parse_pii=True)

    original_text = "Hello, how are you?"
    inputs = {"texts": [original_text]}

    result = await guardrail.apply_guardrail(
        inputs=inputs,  # type: ignore[arg-type]
        request_data={},  # no pii_tokens
        input_type="response",
    )

    assert result["texts"] == [original_text], (
        "apply_guardrail() with empty pii_tokens should return text unchanged"
    )


@pytest.mark.asyncio
async def test_apply_guardrail_request_phase_still_masks() -> None:
    """With input_type='request', masking (check_pii) still runs as before."""
    guardrail = _make_guardrail(output_parse_pii=True)

    # Patch check_pii so we avoid a live Presidio instance.
    with patch.object(
        guardrail,
        "check_pii",
        new=AsyncMock(return_value="Hello, <PERSON>_masked"),
    ):
        result = await guardrail.apply_guardrail(
            inputs={"texts": ["Hello, John Smith"]},  # type: ignore[arg-type]
            request_data={},
            input_type="request",
        )

    assert result["texts"] == ["Hello, <PERSON>_masked"], (
        "apply_guardrail() with input_type='request' must still run check_pii masking"
    )


@pytest.mark.asyncio
async def test_apply_guardrail_response_calls_check_pii_when_output_parse_pii_disabled() -> None:
    """Without output_parse_pii, the response path falls through to check_pii."""
    guardrail = _make_guardrail(output_parse_pii=False)

    with patch.object(
        guardrail,
        "check_pii",
        new=AsyncMock(return_value="unchanged"),
    ) as mock_check_pii:
        await guardrail.apply_guardrail(
            inputs={"texts": ["some text"]},  # type: ignore[arg-type]
            request_data={},
            input_type="response",
        )

    mock_check_pii.assert_called_once(), (
        "check_pii should be called for response phase when output_parse_pii=False"
    )
