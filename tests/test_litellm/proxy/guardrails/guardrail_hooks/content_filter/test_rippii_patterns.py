"""
Tests for rippii-derived PII regex patterns.

The source rippii config applies broad value regexes only near matching tags.
These tests verify the LiteLLM prebuilt patterns preserve that contextual behavior.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../"))

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
    PREBUILT_PATTERNS,
    get_compiled_pattern,
)
from litellm.types.guardrails import ContentFilterAction, ContentFilterPattern


RIPPII_DERIVED_PATTERNS = [
    "phone_number_international_contextual",
    "upi_vpa",
    "in_pan",
    "in_gstin",
    "payment_card_number_contextual",
    "bank_account_number_contextual",
    "cvv_contextual",
    "card_expiry_date_contextual",
    "card_expiry_month_contextual",
    "card_expiry_year_contextual",
    "otp_contextual",
    "authorization_value_contextual",
]


def _guardrail_for_patterns(pattern_names):
    return ContentFilterGuardrail(
        guardrail_name="test-rippii-derived-patterns",
        patterns=[
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name=pattern_name,
                action=ContentFilterAction.MASK,
            )
            for pattern_name in pattern_names
        ],
        pattern_redaction_format="[{pattern_name}_REDACTED]",
    )


async def _filter_text(guardrail, text):
    result = await guardrail.apply_guardrail(
        inputs={"texts": [text]},
        request_data={},
        input_type="request",
    )
    return result["texts"][0]


class TestRippiiDerivedPatternLoading:
    """Test rippii-derived patterns are available as prebuilt patterns."""

    def test_patterns_loaded(self):
        for pattern_name in RIPPII_DERIVED_PATTERNS:
            assert pattern_name in PREBUILT_PATTERNS
            assert get_compiled_pattern(pattern_name) is not None


class TestRippiiDerivedRawPatterns:
    """Test representative raw regex behavior for rippii-derived patterns."""

    def test_india_pan_and_gstin_patterns(self):
        assert get_compiled_pattern("in_pan").search("ABCDE1234F") is not None
        assert get_compiled_pattern("in_gstin").search("27ABCDE1234F1Z5") is not None

    def test_upi_vpa_does_not_match_email_domain(self):
        pattern = get_compiled_pattern("upi_vpa")
        assert pattern.search("alice@okhdfcbank") is not None
        assert pattern.search("alice@example.com") is None

    def test_contextual_financial_value_patterns(self):
        assert (
            get_compiled_pattern("bank_account_number_contextual").search(
                "123456789012"
            )
            is not None
        )
        assert get_compiled_pattern("cvv_contextual").search("123") is not None
        assert (
            get_compiled_pattern("card_expiry_date_contextual").search("05/2028")
            is not None
        )


class TestRippiiDerivedContextualMasking:
    """Test contextual masking preserves rippii tag-near-value semantics."""

    @pytest.mark.asyncio
    async def test_masks_india_identifier_and_payment_patterns_with_context(self):
        guardrail = _guardrail_for_patterns(
            [
                "upi_vpa",
                "in_pan",
                "in_gstin",
                "bank_account_number_contextual",
            ]
        )

        text = (
            "PAN ABCDE1234F, GSTIN 27ABCDE1234F1Z5, "
            "UPI alice@okhdfcbank, account number 123456789012"
        )
        result = await _filter_text(guardrail, text)

        assert "[IN_PAN_REDACTED]" in result
        assert "[IN_GSTIN_REDACTED]" in result
        assert "[UPI_VPA_REDACTED]" in result
        assert "[BANK_ACCOUNT_NUMBER_CONTEXTUAL_REDACTED]" in result
        assert "ABCDE1234F" not in result
        assert "27ABCDE1234F1Z5" not in result
        assert "alice@okhdfcbank" not in result
        assert "123456789012" not in result

    @pytest.mark.asyncio
    async def test_masks_card_security_and_expiry_patterns_with_context(self):
        guardrail = _guardrail_for_patterns(
            [
                "payment_card_number_contextual",
                "cvv_contextual",
                "card_expiry_date_contextual",
                "card_expiry_month_contextual",
                "card_expiry_year_contextual",
            ]
        )

        text = (
            "card 4111-1111-1111-1111, cvv 123, expiry 05/2028, "
            "expiry_month: 05, expiry_year: 2028"
        )
        result = await _filter_text(guardrail, text)

        assert "[PAYMENT_CARD_NUMBER_CONTEXTUAL_REDACTED]" in result
        assert "[CVV_CONTEXTUAL_REDACTED]" in result
        assert "[CARD_EXPIRY_DATE_CONTEXTUAL_REDACTED]" in result
        assert "[CARD_EXPIRY_MONTH_CONTEXTUAL_REDACTED]" in result
        assert "[CARD_EXPIRY_YEAR_CONTEXTUAL_REDACTED]" in result
        assert "4111-1111-1111-1111" not in result
        assert "cvv 123" not in result
        assert "05/2028" not in result

    @pytest.mark.asyncio
    async def test_masks_phone_otp_and_authorization_with_context(self):
        guardrail = _guardrail_for_patterns(
            [
                "phone_number_international_contextual",
                "otp_contextual",
                "authorization_value_contextual",
            ]
        )

        text = (
            "mobile +91 98765 43210, OTP 123456, "
            "Authorization: Bearer abcdef1234567890"
        )
        result = await _filter_text(guardrail, text)

        assert "[PHONE_NUMBER_INTERNATIONAL_CONTEXTUAL_REDACTED]" in result
        assert "[OTP_CONTEXTUAL_REDACTED]" in result
        assert "[AUTHORIZATION_VALUE_CONTEXTUAL_REDACTED]" in result
        assert "+91 98765 43210" not in result
        assert "123456" not in result
        assert "Bearer abcdef1234567890" not in result

    @pytest.mark.asyncio
    async def test_broad_values_without_context_pass_through(self):
        guardrail = _guardrail_for_patterns(
            [
                "bank_account_number_contextual",
                "cvv_contextual",
                "card_expiry_month_contextual",
                "card_expiry_year_contextual",
                "otp_contextual",
                "in_pan",
            ]
        )

        text = (
            "Order 123456789012 has shard 123, batch month 05, "
            "year 2028, code 123456, reference ABCDE1234F."
        )
        result = await _filter_text(guardrail, text)

        assert result == text
        assert "REDACTED" not in result
