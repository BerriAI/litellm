"""The Taiwan policy template only selects zh-TW Presidio entities. Rule text stays in Presidio."""

import json
from pathlib import Path
from typing import Final

from litellm.types.guardrails import LitellmParams, PiiAction

TEMPLATE_ID: Final = "advanced-tw-pii-protection"
DESCRIPTION: Final = "Selects TW_NATIONAL_ID, TW_PHONE_NUMBER, TW_ROC_DATE, and CREDIT_CARD on Presidio language zh-TW."
CHECKED: Final[tuple[tuple[str, str], ...]] = (
    ("tw-national-id", "TW_NATIONAL_ID"),
    ("tw-phone-number", "TW_PHONE_NUMBER"),
    ("tw-roc-date", "TW_ROC_DATE"),
    ("tw-credit-card", "CREDIT_CARD"),
)
_REPO_ROOT: Final = Path(__file__).resolve().parents[3]


def _template(relative: str) -> dict[str, object]:
    raw: Final = json.loads((_REPO_ROOT / relative).read_text())
    assert isinstance(raw, list)
    found: Final = tuple(item for item in raw if isinstance(item, dict) and item.get("id") == TEMPLATE_ID)
    assert len(found) == 1
    match: Final = found[0]
    assert isinstance(match, dict)
    return match


def test_published_and_backup_taiwan_templates_match() -> None:
    published: Final = _template("policy_templates.json")
    backup: Final = _template("litellm/policy_templates_backup.json")
    assert published == backup


def _assert_checked_guardrail(definition: object, name: str, entity: str) -> None:
    assert isinstance(definition, dict)
    assert definition["guardrail_name"] == name
    info: Final = definition["guardrail_info"]
    assert isinstance(info, dict)
    guardrail_description: Final = info["description"]
    assert isinstance(guardrail_description, str)
    assert guardrail_description == f"Masks {entity}"
    assert guardrail_description.isascii()

    params: Final = LitellmParams.model_validate(definition["litellm_params"])
    assert params.guardrail == "presidio"
    assert params.mode == "pre_call"
    assert params.presidio_language == "zh-TW"
    config: Final = params.pii_entities_config
    assert config is not None
    selected: Final = {str(key): action for key, action in config.items()}
    assert selected == {entity: PiiAction.MASK}
    assert "TW_UBN" not in selected


def test_taiwan_template_checks_enabled_zh_tw_entities_only() -> None:
    template: Final = _template("policy_templates.json")
    names: Final = tuple(name for name, _entity in CHECKED)
    summary: Final = template["description"]
    assert isinstance(summary, str)
    assert template["title"] == "Advanced PII Protection (Taiwan)"
    assert summary == DESCRIPTION
    assert summary.isascii()
    assert template["tags"] == ["PII Protection", "Taiwan"]
    assert template["guardrails"] == list(names)

    template_data: Final = template["templateData"]
    assert isinstance(template_data, dict)
    assert template_data["policy_name"] == "advanced-pii-protection-taiwan"
    assert template_data["description"] == DESCRIPTION
    assert template_data["guardrails_add"] == list(names)
    assert template_data["guardrails_remove"] == []

    definitions: Final = template["guardrailDefinitions"]
    assert isinstance(definitions, list)
    assert len(definitions) == len(CHECKED)
    for definition, (name, entity) in zip(definitions, CHECKED, strict=True):
        _assert_checked_guardrail(definition, name, entity)
