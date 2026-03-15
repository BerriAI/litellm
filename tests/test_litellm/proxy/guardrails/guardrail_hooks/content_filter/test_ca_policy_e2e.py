"""
End-to-end tests for Canadian PII Protection (PIPEDA) policy template.

Tests the federal/provincial PII patterns (SIN, OHIP, driver's licence, passport,
immigration docs, bank account, postal code) — validates that PII-containing prompts
are detected/masked and that clean prompts pass through.
University of Toronto institutional identifiers are tested separately in test_uoft_policy_e2e.py.
"""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.types.guardrails import ContentFilterAction, ContentFilterPattern


class TestCanadianPIIPolicyE2E:
    """End-to-end tests for Canadian PII policy template"""

    def setup_canadian_guardrail(self):
        """
        Setup guardrail with all Canadian PII patterns (mimics the policy template)
        """
        patterns = [
            # Government identifiers
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="ca_sin",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="passport_canada",
                action=ContentFilterAction.MASK,
            ),
            # Health & drivers
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="ca_ohip",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="ca_on_drivers_licence",
                action=ContentFilterAction.MASK,
            ),
            # Immigration
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="ca_immigration_doc",
                action=ContentFilterAction.MASK,
            ),
            # Financial
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="ca_bank_account",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="credit_card",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="visa",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="mastercard",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="amex",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="iban",
                action=ContentFilterAction.MASK,
            ),
            # Contact info
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="email",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="us_phone",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="ca_postal_code",
                action=ContentFilterAction.MASK,
            ),
        ]

        return ContentFilterGuardrail(
            guardrail_name="canadian-pii-protection",
            patterns=patterns,
        )

    # =====================
    # SIN tests
    # =====================

    @pytest.mark.asyncio
    async def test_sin_dashed_masked(self):
        """SIN in dashed format is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "My SIN is 123-456-789, please update my tax records."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "[CA_SIN_REDACTED]" in output
        assert "123-456-789" not in output

    @pytest.mark.asyncio
    async def test_sin_spaced_masked(self):
        """SIN in spaced format is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "The employee's social insurance number is 987 654 321."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "[CA_SIN_REDACTED]" in output
        assert "987 654 321" not in output

    @pytest.mark.asyncio
    async def test_sin_question_passes(self):
        """Question about SIN without actual number passes through"""
        guardrail = self.setup_canadian_guardrail()

        text = "What is a Social Insurance Number and how do I apply for one?"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "REDACTED" not in output
        assert output == text

    # =====================
    # OHIP tests
    # =====================

    @pytest.mark.asyncio
    async def test_ohip_dashed_masked(self):
        """OHIP number with version code is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "My OHIP number is 1234-567-890-AB, can you verify my coverage?"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "[CA_OHIP_REDACTED]" in output
        assert "1234-567-890-AB" not in output

    @pytest.mark.asyncio
    async def test_ohip_compact_masked(self):
        """OHIP number in compact format is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "The health card number 9876543210XY needs to be updated in the system."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "[CA_OHIP_REDACTED]" in output
        assert "9876543210XY" not in output

    @pytest.mark.asyncio
    async def test_ohip_question_passes(self):
        """Question about OHIP without actual number passes through"""
        guardrail = self.setup_canadian_guardrail()

        text = "How do I renew my Ontario health card?"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "REDACTED" not in output
        assert output == text

    # =====================
    # Ontario Driver's Licence tests
    # =====================

    @pytest.mark.asyncio
    async def test_drivers_licence_masked(self):
        """Ontario driver's licence is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "My driver's licence number is A1234-56789-01234."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "[CA_ON_DRIVERS_LICENCE_REDACTED]" in output
        assert "A1234-56789-01234" not in output

    @pytest.mark.asyncio
    async def test_drivers_licences_plural_masked(self):
        """Ontario driver's licence with plural 'licenses' is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "My name is Jose Lujan and my drivers licenses is A1234-56789-01234."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "[CA_ON_DRIVERS_LICENCE_REDACTED]" in output
        assert "A1234-56789-01234" not in output

    @pytest.mark.asyncio
    async def test_drivers_licence_question_passes(self):
        """Question about driver's licence without actual number passes through"""
        guardrail = self.setup_canadian_guardrail()

        text = "How do I renew my Ontario driver's licence?"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "REDACTED" not in output
        assert output == text

    # =====================
    # Canadian Passport tests
    # =====================

    @pytest.mark.asyncio
    async def test_passport_masked(self):
        """Canadian passport number is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "My Canadian passport number is AB123456."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "[PASSPORT_CANADA_REDACTED]" in output
        assert "AB123456" not in output

    @pytest.mark.asyncio
    async def test_passport_question_passes(self):
        """Question about passports without actual number passes through"""
        guardrail = self.setup_canadian_guardrail()

        text = "How long does it take to renew a Canadian passport?"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "REDACTED" not in output
        assert output == text

    # =====================
    # Immigration Document tests
    # =====================

    @pytest.mark.asyncio
    async def test_imm_form_masked(self):
        """IRCC IMM form reference is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "Please reference immigration form IMM-5257 for the application."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "[CA_IMMIGRATION_DOC_REDACTED]" in output
        assert "IMM-5257" not in output

    @pytest.mark.asyncio
    async def test_study_permit_masked(self):
        """IRCC study permit number is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "My IRCC study permit number is T123456789."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "[CA_IMMIGRATION_DOC_REDACTED]" in output
        assert "T123456789" not in output

    @pytest.mark.asyncio
    async def test_immigration_question_passes(self):
        """Question about immigration without actual numbers passes through"""
        guardrail = self.setup_canadian_guardrail()

        text = "What documents do I need for a Canadian work permit application?"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "REDACTED" not in output
        assert output == text

    # =====================
    # Bank Account tests
    # =====================

    @pytest.mark.asyncio
    async def test_bank_account_masked(self):
        """Canadian bank account routing info is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "My bank account for direct deposit is 12345-003-1234567."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "[CA_BANK_ACCOUNT_REDACTED]" in output
        assert "12345-003-1234567" not in output

    @pytest.mark.asyncio
    async def test_visa_card_masked(self):
        """Visa card number is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "My Visa card number is 4111111111111111."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "REDACTED" in output
        assert "4111111111111111" not in output

    @pytest.mark.asyncio
    async def test_bank_question_passes(self):
        """Question about banking without actual numbers passes through"""
        guardrail = self.setup_canadian_guardrail()

        text = "How do I find my bank's transit and institution number?"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "REDACTED" not in output
        assert output == text

    # =====================
    # Phone Number tests
    # =====================

    @pytest.mark.asyncio
    async def test_phone_number_masked(self):
        """North American phone number is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "Call me at (416) 555-1234 to discuss."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "REDACTED" in output
        assert "(416) 555-1234" not in output

    # =====================
    # Postal Code tests
    # =====================

    @pytest.mark.asyncio
    async def test_postal_code_spaced_masked(self):
        """Canadian postal code in spaced format is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "Ship the package to my postal code M5V 2T6."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "[CA_POSTAL_CODE_REDACTED]" in output
        assert "M5V 2T6" not in output

    @pytest.mark.asyncio
    async def test_postal_code_compact_masked(self):
        """Canadian postal code in compact format is detected and masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "My mailing address postal code is K1A0B1."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "[CA_POSTAL_CODE_REDACTED]" in output
        assert "K1A0B1" not in output

    @pytest.mark.asyncio
    async def test_postal_code_question_passes(self):
        """Question about postal codes without actual code passes through"""
        guardrail = self.setup_canadian_guardrail()

        text = "What is the format of a Canadian postal code?"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "REDACTED" not in output
        assert output == text

    # =====================
    # Combined / edge case tests
    # =====================

    @pytest.mark.asyncio
    async def test_normal_text_passes(self):
        """Normal text without any PII passes through unchanged"""
        guardrail = self.setup_canadian_guardrail()

        text = "Please schedule a meeting for next Tuesday to discuss the project."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "REDACTED" not in output
        assert output == text

    @pytest.mark.asyncio
    async def test_multiple_pii_types_masked(self):
        """Multiple Canadian PII types in same message are all masked"""
        guardrail = self.setup_canadian_guardrail()

        text = (
            "Employee SIN 123-456-789, "
            "email jane@example.com, "
            "postal code M5V 2T6."
        )
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "123-456-789" not in output
        assert "jane@example.com" not in output
        assert "M5V 2T6" not in output
        assert "CA_SIN_REDACTED" in output
        assert "EMAIL_REDACTED" in output
        assert "CA_POSTAL_CODE_REDACTED" in output

    @pytest.mark.asyncio
    async def test_invalid_postal_code_first_letter_passes(self):
        """Postal code with invalid first letter (D, F, I, O, Q, U) is not masked"""
        guardrail = self.setup_canadian_guardrail()

        text = "The code D5V 2T6 is not a valid postal code."
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]

        assert "D5V 2T6" in output
        assert "POSTAL_CODE" not in output
