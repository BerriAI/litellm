from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .base import GuardrailConfigModel

# TypedDicts for EnkryptAI API Response Structure


class EnkryptAIPolicyViolationDetail(TypedDict, total=False):
    """Details for policy violation detection."""

    violating_policy: str
    explanation: str


class EnkryptAIPIIDetail(TypedDict, total=False):
    """Details for PII detection."""

    pii: dict[str, Any]


class EnkryptAIToxicityDetail(TypedDict, total=False):
    """Details for toxicity detection.

    Contains scores for different types of toxicity:
    - toxic
    - severe_toxic
    - obscene
    - threat
    - insult
    - identity_hate
    """

    toxic: float | None
    severe_toxic: float | None
    obscene: float | None
    threat: float | None
    insult: float | None
    identity_hate: float | None


class EnkryptAIKeywordDetail(TypedDict, total=False):
    """Details for keyword detection."""

    detected_keywords: list[str]


class EnkryptAIBiasDetail(TypedDict, total=False):
    """Details for bias detection."""

    bias_detected: bool


class EnkryptAIResponseSummary(TypedDict, total=False):
    """Summary of detected violations in EnkryptAI response.

    Each key represents a type of violation:
    - toxicity: List (non-empty if detected)
    - policy_violation: 0 or 1
    - pii: 0 or 1
    - keyword_detected: 0 or 1
    - bias: 0 or 1
    - prompt_injection: 0 or 1
    - jailbreak: 0 or 1
    """

    toxicity: list[str] | None
    policy_violation: int | None
    pii: int | None
    keyword_detected: int | None
    bias: int | None
    prompt_injection: int | None
    jailbreak: int | None


class EnkryptAIResponseDetails(TypedDict, total=False):
    """Detailed information about detected violations."""

    policy_violation: EnkryptAIPolicyViolationDetail | None
    pii: EnkryptAIPIIDetail | None
    toxicity: EnkryptAIToxicityDetail | None
    keyword_detected: EnkryptAIKeywordDetail | None
    bias: EnkryptAIBiasDetail | None
    prompt_injection: dict[str, Any] | None
    jailbreak: dict[str, Any] | None


class EnkryptAIResponse(TypedDict, total=False):
    """Complete response from EnkryptAI Guardrails API."""

    summary: EnkryptAIResponseSummary
    details: EnkryptAIResponseDetails


class EnkryptAIProcessedResult(TypedDict):
    """Processed result from EnkryptAI guardrail response."""

    attacks_detected: list[str]
    attack_details: dict[
        str,
        EnkryptAIPolicyViolationDetail
        | EnkryptAIPIIDetail
        | EnkryptAIToxicityDetail
        | EnkryptAIKeywordDetail
        | EnkryptAIBiasDetail
        | dict[str, Any],
    ]


# Pydantic Config Model
class EnkryptAIGuardrailConfigs(BaseModel):
    """Configuration parameters for the EnkryptAI guardrail"""

    api_key: str | None = Field(
        default=None,
        description="The EnkryptAI API key. Reads from ENKRYPTAI_API_KEY env var if None.",
    )
    api_base: str | None = Field(
        default=None,
        description="The EnkryptAI API base URL. Defaults to https://api.enkryptai.com. Also checks if the ENKRYPTAI_API_KEY env var is set.",
    )
    policy_name: str | None = Field(
        default=None,
        description="The EnkryptAI policy name to use. Sent via x-enkrypt-policy header.",
    )
    deployment_name: str | None = Field(
        default=None,
        description="The EnkryptAI deployment name to use. Sent via X-Enkrypt-Deployment header.",
    )
    detectors: dict | None = Field(
        default=None,
        description="Dictionary of detector configurations (e.g., {'nsfw': {'enabled': True}, 'toxicity': {'enabled': True}}).",
    )
    block_on_violation: bool | None = Field(
        default=True,
        description="Whether to block requests when violations are detected. Defaults to True.",
    )


class EnkryptAIGuardrailConfigModel(GuardrailConfigModel, EnkryptAIGuardrailConfigs):
    @staticmethod
    def ui_friendly_name() -> str:
        return "EnkryptAI"
