import json
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.types.guardrails import ContentFilterAction, ContentFilterPattern

REPO_ROOT: Final = Path(__file__).resolve().parents[6]
UK_PATTERN_CASES: Final = (
    ("uk_nino", "national insurance number", "AB 12 34 56 C", "BG123456A"),
    ("uk_nhs_number", "NHS number", "943 476 5919", "94-3476-5919"),
    ("uk_driving_licence", "driving licence number", "ABCDE123456AB1CD", "ABCDE12345AB1CD"),
    ("uk_utr", "unique taxpayer reference", "12345 67890", "1234 567890"),
    ("uk_sort_code", "sort code", "12-34-56", "12 345 6"),
    ("uk_bank_account", "bank account number", "12345678", "1234567"),
    ("uk_postcode", "postcode", "SW1A 1AA", "QQ1 1AA"),
    ("uk_phone", "mobile number", "07911 123456", "04911 123456"),
    ("uk_upn", "unique pupil number", "A123456789012", "A12345678901"),
    ("uk_uln", "unique learner number", "1234567890", "123456789"),
)
UK_PATTERN_MATCH_CASES: Final = tuple(case[:3] for case in UK_PATTERN_CASES)


def _guardrail_for(pattern_name: str) -> ContentFilterGuardrail:
    return ContentFilterGuardrail(
        guardrail_name=f"test-{pattern_name}",
        patterns=[
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name=pattern_name,
                action=ContentFilterAction.MASK,
            )
        ],
        pattern_redaction_format="[{pattern_name}_REDACTED]",
    )


@pytest.mark.parametrize(
    "pattern_name, keyword, valid_value",
    UK_PATTERN_MATCH_CASES,
)
@pytest.mark.asyncio
async def test_uk_pii_value_with_keyword_is_masked(
    pattern_name: str,
    keyword: str,
    valid_value: str,
) -> None:
    guardrail: Final = _guardrail_for(pattern_name)
    result: Final = await guardrail.apply_guardrail(
        inputs={"texts": [f"{keyword}: {valid_value}"]},
        request_data={},
        input_type="request",
    )
    output: Final = result["texts"][0]

    assert f"[{pattern_name.upper()}_REDACTED]" in output
    assert valid_value not in output


@pytest.mark.parametrize(
    "pattern_name, keyword, valid_value",
    UK_PATTERN_MATCH_CASES,
)
@pytest.mark.asyncio
async def test_uk_pii_value_without_keyword_passes_unchanged(
    pattern_name: str,
    keyword: str,
    valid_value: str,
) -> None:
    guardrail: Final = _guardrail_for(pattern_name)
    result: Final = await guardrail.apply_guardrail(
        inputs={"texts": [valid_value]},
        request_data={},
        input_type="request",
    )
    output: Final = result["texts"][0]

    assert output == valid_value
    assert keyword.lower() not in output.lower()


@pytest.mark.parametrize(
    "pattern_name, keyword, valid_value, invalid_value",
    UK_PATTERN_CASES,
)
@pytest.mark.asyncio
async def test_invalid_uk_pii_format_passes_unchanged(
    pattern_name: str,
    keyword: str,
    valid_value: str,
    invalid_value: str,
) -> None:
    guardrail: Final = _guardrail_for(pattern_name)
    text: Final = f"{keyword}: {invalid_value}"
    result: Final = await guardrail.apply_guardrail(
        inputs={"texts": [text]},
        request_data={},
        input_type="request",
    )
    output: Final = result["texts"][0]

    assert output == text
    assert valid_value not in output


def test_uk_policy_templates_reference_available_prebuilt_patterns() -> None:
    patterns_file: Final = REPO_ROOT / "litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/patterns.json"
    pattern_names: Final = {
        pattern["name"] for pattern in json.loads(patterns_file.read_text())["patterns"]
    }
    policy_files: Final = (
        REPO_ROOT / "policy_templates.json",
        REPO_ROOT / "litellm/policy_templates_backup.json",
    )
    policy_templates: Final = tuple(
        json.loads(policy_file.read_text()) for policy_file in policy_files
    )
    uk_templates: Final = tuple(
        next(template for template in templates if template["id"] == "uk-pii-protection")
        for templates in policy_templates
    )
    backup_ids: Final = {template["id"] for template in policy_templates[1]}

    assert "canadian-pii-protection" in backup_ids
    for template in uk_templates:
        for guardrail in template["guardrailDefinitions"]:
            for pattern in guardrail["litellm_params"]["patterns"]:
                if pattern["pattern_type"] == "prebuilt":
                    assert pattern["pattern_name"] in pattern_names
