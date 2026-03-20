"""
Tests for POST /policy/templates/test endpoint logic.

Tests _test_guardrail_definitions and _compute_overall_action directly
without needing a running proxy.
"""

import pytest

from litellm.proxy.management_endpoints.policy_endpoints.endpoints import (
    GuardrailTestResultEntry,
    _compute_overall_action,
    _test_guardrail_definitions,
)


@pytest.mark.asyncio
async def test_pattern_based_guardrail_masks_pii():
    """A pattern-based guardrail should mask matching PII."""
    guardrail_defs = [
        {
            "guardrail_name": "test-ssn-masker",
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "patterns": [
                    {
                        "pattern_type": "prebuilt",
                        "pattern_name": "us_ssn",
                        "action": "MASK",
                    }
                ],
                "pattern_redaction_format": "[{pattern_name}_REDACTED]",
            },
            "guardrail_info": {"description": "Masks US SSNs"},
        }
    ]

    results = await _test_guardrail_definitions(
        guardrail_definitions=guardrail_defs,
        text="My SSN is 123-45-6789",
    )

    assert len(results) == 1
    assert results[0]["guardrail_name"] == "test-ssn-masker"
    assert results[0]["action"] == "masked"
    assert "123-45-6789" not in results[0]["output_text"]
    assert "REDACTED" in results[0]["output_text"]


@pytest.mark.asyncio
async def test_blocked_words_guardrail_blocks():
    """A blocked_words guardrail should block matching text."""
    guardrail_defs = [
        {
            "guardrail_name": "test-word-blocker",
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "blocked_words": [
                    {
                        "keyword": "forbidden_word",
                        "action": "BLOCK",
                        "description": "test block",
                    }
                ],
            },
            "guardrail_info": {"description": "Blocks forbidden words"},
        }
    ]

    results = await _test_guardrail_definitions(
        guardrail_definitions=guardrail_defs,
        text="This contains forbidden_word in it",
    )

    assert len(results) == 1
    assert results[0]["guardrail_name"] == "test-word-blocker"
    assert results[0]["action"] == "blocked"


@pytest.mark.asyncio
async def test_clean_text_passes():
    """Clean text should pass all guardrails."""
    guardrail_defs = [
        {
            "guardrail_name": "test-ssn-masker",
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "patterns": [
                    {
                        "pattern_type": "prebuilt",
                        "pattern_name": "us_ssn",
                        "action": "MASK",
                    }
                ],
            },
            "guardrail_info": {"description": "Masks US SSNs"},
        }
    ]

    results = await _test_guardrail_definitions(
        guardrail_definitions=guardrail_defs,
        text="Hello, this is a perfectly clean message.",
    )

    assert len(results) == 1
    assert results[0]["action"] == "passed"
    assert results[0]["output_text"] == "Hello, this is a perfectly clean message."


@pytest.mark.asyncio
async def test_unsupported_guardrail_type():
    """Non-litellm_content_filter types should return unsupported."""
    guardrail_defs = [
        {
            "guardrail_name": "test-mcp",
            "litellm_params": {
                "guardrail": "mcp_security",
                "mode": "pre_call",
            },
            "guardrail_info": {"description": "MCP guardrail"},
        }
    ]

    results = await _test_guardrail_definitions(
        guardrail_definitions=guardrail_defs,
        text="Any text",
    )

    assert len(results) == 1
    assert results[0]["action"] == "unsupported"
    assert "mcp_security" in results[0]["details"]


@pytest.mark.asyncio
async def test_multiple_guardrails_mixed_results():
    """Multiple guardrails with different outcomes."""
    guardrail_defs = [
        {
            "guardrail_name": "ssn-masker",
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "patterns": [
                    {
                        "pattern_type": "prebuilt",
                        "pattern_name": "us_ssn",
                        "action": "MASK",
                    }
                ],
                "pattern_redaction_format": "[{pattern_name}_REDACTED]",
            },
            "guardrail_info": {"description": "Masks SSNs"},
        },
        {
            "guardrail_name": "email-masker",
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "patterns": [
                    {
                        "pattern_type": "prebuilt",
                        "pattern_name": "email",
                        "action": "MASK",
                    }
                ],
                "pattern_redaction_format": "[{pattern_name}_REDACTED]",
            },
            "guardrail_info": {"description": "Masks emails"},
        },
    ]

    results = await _test_guardrail_definitions(
        guardrail_definitions=guardrail_defs,
        text="My SSN is 123-45-6789 but no email here",
    )

    assert len(results) == 2
    ssn_result = next(r for r in results if r["guardrail_name"] == "ssn-masker")
    email_result = next(r for r in results if r["guardrail_name"] == "email-masker")
    assert ssn_result["action"] == "masked"
    assert email_result["action"] == "passed"


def test_compute_overall_action_blocked_wins():
    results: list[GuardrailTestResultEntry] = [
        GuardrailTestResultEntry(guardrail_name="a", action="passed", output_text="", details=""),
        GuardrailTestResultEntry(guardrail_name="b", action="blocked", output_text="", details=""),
        GuardrailTestResultEntry(guardrail_name="c", action="masked", output_text="", details=""),
    ]
    assert _compute_overall_action(results) == "blocked"


def test_compute_overall_action_masked_wins_over_passed():
    results: list[GuardrailTestResultEntry] = [
        GuardrailTestResultEntry(guardrail_name="a", action="passed", output_text="", details=""),
        GuardrailTestResultEntry(guardrail_name="b", action="masked", output_text="", details=""),
    ]
    assert _compute_overall_action(results) == "masked"


def test_compute_overall_action_all_passed():
    results: list[GuardrailTestResultEntry] = [
        GuardrailTestResultEntry(guardrail_name="a", action="passed", output_text="", details=""),
        GuardrailTestResultEntry(guardrail_name="b", action="passed", output_text="", details=""),
    ]
    assert _compute_overall_action(results) == "passed"


def test_compute_overall_action_empty():
    assert _compute_overall_action([]) == "passed"
