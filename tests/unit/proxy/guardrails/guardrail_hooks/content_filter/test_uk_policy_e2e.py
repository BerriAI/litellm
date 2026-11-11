import json
from itertools import chain
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.types.guardrails import ContentFilterAction, ContentFilterPattern

REPO_ROOT: Final = Path(__file__).resolve().parents[6]
UK_TEMPLATE_PATTERN_NAMES: Final = (
    "uk_nino",
    "uk_utr",
    "passport_uk",
    "uk_nhs_number",
    "uk_driving_licence",
    "uk_sort_code",
    "uk_bank_account",
    "credit_card",
    "visa",
    "mastercard",
    "amex",
    "iban",
    "email",
    "uk_phone",
    "uk_postcode",
    "uk_upn",
    "uk_uln",
)

UK_PII_CASES: Final = (
    ("National Insurance Number: AB 12 34 56 C", "AB 12 34 56 C", "[UK_NINO_REDACTED]"),
    ("Unique Taxpayer Reference: 12345 67890", "12345 67890", "[UK_UTR_REDACTED]"),
    ("UK passport number: 123456789", "123456789", "[PASSPORT_UK_REDACTED]"),
    ("NHS number: 943 476 5919", "943 476 5919", "[UK_NHS_NUMBER_REDACTED]"),
    ("Driving licence number: ABCDE123456AB1CD", "ABCDE123456AB1CD", "[UK_DRIVING_LICENCE_REDACTED]"),
    ("Sort code: 12-34-56", "12-34-56", "[UK_SORT_CODE_REDACTED]"),
    ("Bank account number: 12345678", "12345678", "[UK_BANK_ACCOUNT_REDACTED]"),
    ("Credit card number: 4111111111111111", "4111111111111111", "[CREDIT_CARD_REDACTED]"),
    ("Visa card number: 4111111111111111", "4111111111111111", "[CREDIT_CARD_REDACTED]"),
    ("Mastercard number: 5555555555554444", "5555555555554444", "[CREDIT_CARD_REDACTED]"),
    ("Amex card number: 378282246310005", "378282246310005", "[CREDIT_CARD_REDACTED]"),
    ("IBAN: GB82WEST12345698765432", "GB82WEST12345698765432", "[IBAN_REDACTED]"),
    ("Email: person@example.com", "person@example.com", "[EMAIL_REDACTED]"),
    ("Mobile number: 07911 123456", "07911 123456", "[UK_PHONE_REDACTED]"),
    ("Postcode: SW1A 1AA", "SW1A 1AA", "[UK_POSTCODE_REDACTED]"),
    ("Unique pupil number: A123456789012", "A123456789012", "[UK_UPN_REDACTED]"),
    ("Unique learner number: 1234567890", "1234567890", "[UK_ULN_REDACTED]"),
)


def test_full_pipeline_covers_all_template_patterns() -> None:
    policy_templates: Final = json.loads((REPO_ROOT / "policy_templates.json").read_text())
    uk_template: Final = next(
        template for template in policy_templates if template["id"] == "uk-pii-protection"
    )
    guardrail_patterns: Final = tuple(
        guardrail["litellm_params"]["patterns"]
        for guardrail in uk_template["guardrailDefinitions"]
    )
    template_pattern_names: Final = tuple(
        pattern["pattern_name"] for pattern in chain.from_iterable(guardrail_patterns)
    )

    assert template_pattern_names == UK_TEMPLATE_PATTERN_NAMES


class TestUKPIIPolicyE2E:
    def setup_uk_guardrail(self) -> ContentFilterGuardrail:
        patterns: Final = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name=pattern_name,
                action=ContentFilterAction.MASK,
            )
            for pattern_name in UK_TEMPLATE_PATTERN_NAMES
        ]

        return ContentFilterGuardrail(
            guardrail_name="uk-pii-protection",
            patterns=patterns,
            pattern_redaction_format="[{pattern_name}_REDACTED]",
        )

    @pytest.mark.parametrize(
        "text, sensitive_value, redaction",
        UK_PII_CASES,
        ids=tuple(f"case-{index}" for index in range(len(UK_PII_CASES))),
    )
    @pytest.mark.asyncio
    async def test_template_pii_is_masked(self, text: str, sensitive_value: str, redaction: str) -> None:
        guardrail: Final = self.setup_uk_guardrail()
        result: Final = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output: Final = result["texts"][0]

        assert redaction in output
        assert sensitive_value not in output

    @pytest.mark.asyncio
    async def test_clean_prompt_passes_unchanged(self) -> None:
        guardrail: Final = self.setup_uk_guardrail()
        text: Final = "How do I update my address with the local council?"
        result: Final = await guardrail.apply_guardrail(
            inputs={"texts": [text]},
            request_data={},
            input_type="request",
        )
        output: Final = result["texts"][0]

        assert output == text
