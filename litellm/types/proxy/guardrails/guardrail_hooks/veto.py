from typing import Any, List, Optional, cast

from pydantic import Field

from .base import GuardrailConfigModel

VETO_CATEGORY_OPTIONS = ["pii", "secrets", "injection"]


class VetoGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "Veto API key (prefix 'vt_'). If not provided, the "
            "VETO_API_KEY environment variable is used."
        ),
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Veto gateway base URL. Defaults to https://api.vetocheck.com. "
            "Falls back to the VETO_API_BASE env var."
        ),
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description=(
            "Detector categories to run on each request. Select one or more "
            "of pii, secrets, injection; if none are selected the guardrail "
            "runs all three."
        ),
        json_schema_extra=cast(
            Any,
            {
                "ui_type": "multiselect",
                "options": VETO_CATEGORY_OPTIONS,
            },
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Veto"
