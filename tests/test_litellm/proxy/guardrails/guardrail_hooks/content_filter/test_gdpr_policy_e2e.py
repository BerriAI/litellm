"""
End-to-end tests for GDPR Art. 32 EU PII Protection policy template
Tests the complete policy with various EU PII patterns
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../"))

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.types.guardrails import (
    ContentFilterAction,
    ContentFilterPattern,
)


class TestGDPRPolicyE2E:
    """End-to-end tests for GDPR policy template"""

    def setup_gdpr_guardrail(self):
        """
        Setup guardrail with all GDPR patterns (mimics the policy template)
        """
        patterns = [
            # National identifiers
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="fr_nir",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="eu_passport_generic",
                action=ContentFilterAction.MASK,
            ),
            # Financial data
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="eu_iban_enhanced",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="iban",
                action=ContentFilterAction.MASK,
            ),
            # Contact information
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="email",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="fr_phone",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="fr_postal_code",
                action=ContentFilterAction.MASK,
            ),
            # Business identifiers
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="eu_vat",
                action=ContentFilterAction.MASK,
            ),
        ]

        return ContentFilterGuardrail(
            guardrail_name="gdpr-eu-pii-protection",
            patterns=patterns,
        )

    @pytest.mark.asyncio
    async def test_french_nir_masked(self):
        """
        Test 1 - SHOULD MASK: French NIR/INSEE number is detected and masked
        """
        guardrail = self.setup_gdpr_guardrail()

        text = "The employee's NIR is 192057512345678 for tax purposes"
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])[0]

        assert "[FR_NIR_REDACTED]" in result
        assert "192057512345678" not in result

    @pytest.mark.asyncio
    async def test_eu_iban_masked(self):
        """
        Test 2 - SHOULD MASK: EU IBAN is detected and masked
        """
        guardrail = self.setup_gdpr_guardrail()

        text = "Wire transfer to account FR7630006000011234567890189"
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])[0]

        # Either pattern could match first
        assert "[EU_IBAN_ENHANCED_REDACTED]" in result or "[IBAN_REDACTED]" in result
        assert "FR7630006000011234567890189" not in result

    @pytest.mark.asyncio
    async def test_french_phone_masked(self):
        """
        Test 3 - SHOULD MASK: French phone number is detected and masked
        """
        guardrail = self.setup_gdpr_guardrail()

        text = "Call me at +33612345678 tomorrow"
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])[0]

        assert "[FR_PHONE_REDACTED]" in result
        assert "+33612345678" not in result

    @pytest.mark.asyncio
    async def test_eu_vat_masked(self):
        """
        Test 4 - SHOULD MASK: EU VAT number with keyword context is detected and masked
        """
        guardrail = self.setup_gdpr_guardrail()

        # Include VAT keyword for contextual matching (max 1 word gap)
        text = "Company VAT number: FR12345678901"
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])[0]

        assert "[EU_VAT_REDACTED]" in result
        assert "FR12345678901" not in result

    @pytest.mark.asyncio
    async def test_normal_text_passes(self):
        """
        Test 5 - SHOULD NOT MASK: Normal text without PII passes through
        """
        guardrail = self.setup_gdpr_guardrail()

        text = "This is a regular business communication about our meeting"
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])[0]

        # No redaction markers should be present
        assert "REDACTED" not in result
        assert result == text

    @pytest.mark.asyncio
    async def test_invalid_nir_passes(self):
        """
        Test 6 - SHOULD NOT MASK: Invalid NIR (month 13) is not detected
        """
        guardrail = self.setup_gdpr_guardrail()

        text = "The invalid number 192137512345678 is not a valid NIR"
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])[0]

        # Should not mask invalid NIR
        assert "192137512345678" in result
        assert "REDACTED" not in result

    @pytest.mark.asyncio
    async def test_invalid_phone_passes(self):
        """
        Test 7 - SHOULD NOT MASK: Invalid French phone (starts with 0) is not detected
        """
        guardrail = self.setup_gdpr_guardrail()

        text = "This number 0012345678 is not a valid French phone"
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])[0]

        # Should not mask invalid phone
        assert "0012345678" in result
        assert "REDACTED" not in result

    @pytest.mark.asyncio
    async def test_random_digits_without_context_passes(self):
        """
        Test 8 - SHOULD NOT MASK: Random 5-digit number without postal code context
        """
        guardrail = self.setup_gdpr_guardrail()

        text = "The order number is 12345 for tracking"
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])[0]

        # Should not mask 5-digit number without postal code context
        assert "12345" in result
        assert "REDACTED" not in result

    @pytest.mark.asyncio
    async def test_multiple_pii_types_masked(self):
        """
        Bonus test: Multiple PII types in same message are all masked
        """
        guardrail = self.setup_gdpr_guardrail()

        text = "Contact jean@example.com at +33612345678 with NIR 192057512345678"
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])[0]

        # All PII should be masked
        assert "EMAIL_REDACTED" in result
        assert "FR_PHONE_REDACTED" in result or "FR_NIR_REDACTED" in result
        assert "jean@example.com" not in result
        assert "+33612345678" not in result
        assert "192057512345678" not in result

    @pytest.mark.asyncio
    async def test_vat_number_without_keyword_context_passes(self):
        """
        Test 10 - SHOULD NOT MASK: VAT-like pattern without keyword context
        Contextual keyword guard prevents false positives
        """
        guardrail = self.setup_gdpr_guardrail()

        # Text with VAT-like format but no VAT keyword context
        text = "Product code FR12345678 for the shipment"
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])[0]

        # Should not mask without VAT keyword context
        assert "FR12345678" in result
        assert "REDACTED" not in result

    @pytest.mark.asyncio
    async def test_passport_number_without_keyword_context_passes(self):
        """
        Test 11 - SHOULD NOT MASK: Passport-like pattern without keyword context
        Contextual keyword guard prevents false positives
        """
        guardrail = self.setup_gdpr_guardrail()

        # Text with passport-like format but no passport keyword context
        text = "Reference number 12AB34567 for your order"
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])[0]

        # Should not mask without passport keyword context
        assert "12AB34567" in result
        assert "REDACTED" not in result
