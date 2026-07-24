from typing import List, Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class ATRGuardrailConfigModelOptionalParams(BaseModel):
    severity_threshold: Optional[str] = Field(
        default="high",
        description=(
            "Minimum ATR rule severity to block: 'critical', 'high', "
            "'medium', or 'low'. Matches below this threshold are not "
            "blocked. Defaults to 'high'."
        ),
    )
    include_tags: Optional[List[str]] = Field(
        default=None,
        description=(
            "If set, only rules whose tags contain any of the listed "
            "values (e.g. 'prompt_injection', 'tool_poisoning') are "
            "applied. When None, all loaded rules are applied."
        ),
    )


class ATRGuardrailConfigModel(
    GuardrailConfigModel[ATRGuardrailConfigModelOptionalParams]
):
    rules_path: Optional[str] = Field(
        default=None,
        description=(
            "Filesystem path to an ATR rules directory. If omitted, "
            "the rules bundled with pyatr (./rules sibling directory) "
            "are loaded. Also checks ATR_RULES_PATH environment variable."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "ATR (Agent Threat Rules)"
