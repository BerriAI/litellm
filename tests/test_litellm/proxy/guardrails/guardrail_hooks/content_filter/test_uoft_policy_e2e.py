"""
End-to-end tests for University of Toronto identifier patterns (FIPPA compliance).

Tests the complete guardrail with UofT patterns — validates that
institutional identifiers are detected/masked and that clean prompts pass through.
"""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.types.guardrails import ContentFilterAction, ContentFilterPattern


class TestUofTPolicyE2E:
    """End-to-end tests for University of Toronto identifier patterns"""

    def setup_uoft_guardrail(self):
        """
        Setup guardrail with all UofT patterns (mimics the policy template sub-guardrail)
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="uoft_student_id",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="uoft_utorid",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="uoft_tcard",
                action=ContentFilterAction.MASK,
            ),
        ]
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-uoft-ids",
            patterns=patterns,
            pattern_redaction_format="[{pattern_name}_REDACTED]",
        )
        return guardrail

    # =====================
    # Student/Employee Number tests
    # =====================

    @pytest.mark.asyncio
    async def test_student_number_masked(self):
        guardrail = self.setup_uoft_guardrail()
        text = "My student number is 1012345678 for registration"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert "[UOFT_STUDENT_ID_REDACTED]" in output
        assert "1012345678" not in output

    @pytest.mark.asyncio
    async def test_employee_id_masked(self):
        guardrail = self.setup_uoft_guardrail()
        text = "Employee id 1099887766 needs access to the lab"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert "[UOFT_STUDENT_ID_REDACTED]" in output
        assert "1099887766" not in output

    @pytest.mark.asyncio
    async def test_uoft_student_id_in_context(self):
        guardrail = self.setup_uoft_guardrail()
        text = "University of Toronto student 1012345678 enrolled in CS"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert "[UOFT_STUDENT_ID_REDACTED]" in output
        assert "1012345678" not in output

    @pytest.mark.asyncio
    async def test_student_number_clean_passes(self):
        guardrail = self.setup_uoft_guardrail()
        text = "How do I find my U of T student number?"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert output == text

    # =====================
    # UTORid tests
    # =====================

    @pytest.mark.asyncio
    async def test_utorid_masked(self):
        guardrail = self.setup_uoft_guardrail()
        text = "My UTORid is smithj12 for ACORN login"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert "[UOFT_UTORID_REDACTED]" in output
        assert "smithj12" not in result

    @pytest.mark.asyncio
    async def test_utorid_quercus_context_masked(self):
        guardrail = self.setup_uoft_guardrail()
        text = "Quercus login: kcheng42"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert "[UOFT_UTORID_REDACTED]" in output
        assert "kcheng42" not in output

    @pytest.mark.asyncio
    async def test_utorid_acorn_context_masked(self):
        guardrail = self.setup_uoft_guardrail()
        text = "Log in to ACORN with li5"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert "[UOFT_UTORID_REDACTED]" in output
        assert "li5" not in output

    @pytest.mark.asyncio
    async def test_utorid_clean_passes(self):
        guardrail = self.setup_uoft_guardrail()
        text = "How do I reset my UTORid password?"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert output == text

    # =====================
    # TCard tests
    # =====================

    @pytest.mark.asyncio
    async def test_tcard_masked(self):
        guardrail = self.setup_uoft_guardrail()
        text = "My TCard number is 1234567890123456 for library access"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert "[UOFT_TCARD_REDACTED]" in output
        assert "1234567890123456" not in output

    @pytest.mark.asyncio
    async def test_campus_card_masked(self):
        guardrail = self.setup_uoft_guardrail()
        text = "Campus card 9876543210987654 needs reactivation"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert "[UOFT_TCARD_REDACTED]" in output
        assert "9876543210987654" not in output

    @pytest.mark.asyncio
    async def test_student_card_masked(self):
        guardrail = self.setup_uoft_guardrail()
        text = "Lost my student card 1111222233334444 yesterday"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert "[UOFT_TCARD_REDACTED]" in output
        assert "1111222233334444" not in output

    @pytest.mark.asyncio
    async def test_tcard_clean_passes(self):
        guardrail = self.setup_uoft_guardrail()
        text = "Where can I get a replacement TCard on campus?"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert output == text

    # =====================
    # No false positives without context
    # =====================

    @pytest.mark.asyncio
    async def test_generic_10digit_no_context_passes(self):
        """10-digit number starting with 10 but no keyword context should pass"""
        guardrail = self.setup_uoft_guardrail()
        text = "The order total is 1012345678 units"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert output == text

    @pytest.mark.asyncio
    async def test_generic_short_word_no_context_passes(self):
        """Generic short word with digits should not be masked without UTORid context"""
        guardrail = self.setup_uoft_guardrail()
        text = "The variable abc123 is used in the code"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert output == text

    @pytest.mark.asyncio
    async def test_generic_16digit_no_context_passes(self):
        """16-digit number without TCard context should not match TCard pattern"""
        guardrail = self.setup_uoft_guardrail()
        text = "Reference number 1234567890123456 for the shipment"
        result = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output = result.get("texts", [])[0]
        assert output == text
