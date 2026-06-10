from typing import Dict, List, Optional

from pydantic import Field
from typing_extensions import TypedDict

from .base import GuardrailConfigModel


class HighflameGuardInput(TypedDict):
    text: str


class HighflameGuardRequest(TypedDict):
    input: HighflameGuardInput
    config: Optional[Dict]
    metadata: Optional[Dict]


class HighflamePromptInjectionCategories(TypedDict):
    prompt_injection: bool
    jailbreak: bool


class HighflamePromptInjectionCategoryScores(TypedDict):
    prompt_injection: float
    jailbreak: float


class HighflamePromptInjectionResults(TypedDict):
    categories: HighflamePromptInjectionCategories
    category_scores: HighflamePromptInjectionCategoryScores
    reject_prompt: str


class HighflamePromptInjectionAssessment(TypedDict):
    results: HighflamePromptInjectionResults
    request_reject: bool


class HighflameTrustSafetyCategories(TypedDict):
    violence: bool
    weapons: bool
    hate_speech: bool
    crime: bool
    sexual: bool
    profanity: bool


class HighflameTrustSafetyCategoryScores(TypedDict):
    violence: float
    weapons: float
    hate_speech: float
    crime: float
    sexual: float
    profanity: float


class HighflameTrustSafetyResults(TypedDict):
    categories: HighflameTrustSafetyCategories
    category_scores: HighflameTrustSafetyCategoryScores


class HighflameTrustSafetyAssessment(TypedDict):
    results: HighflameTrustSafetyResults
    request_reject: bool


class HighflameLanguageDetectionResults(TypedDict):
    lang: str
    prob: float


class HighflameLanguageDetectionAssessment(TypedDict):
    results: HighflameLanguageDetectionResults
    request_reject: bool


class HighflameGuardResponse(TypedDict):
    assessments: List[
        Dict[
            str,
            HighflamePromptInjectionAssessment
            | HighflameTrustSafetyAssessment
            | HighflameLanguageDetectionAssessment,
        ]
    ]


class HighflameGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the Highflame guardrail. See https://docs.highflame.ai/"""

    guard_name: Optional[str] = Field(
        default="highflame_guard",
        description="Name of the Highflame guard to use",
    )
    api_version: Optional[str] = Field(
        default="v1", description="API version for Highflame service"
    )
    metadata: Optional[Dict] = Field(
        default=None, description="Additional metadata to send with requests"
    )
    application: Optional[str] = Field(
        default=None, description="Application name for Highflame service"
    )
    config: Optional[Dict] = Field(
        default=None, description="Configuration parameters for Highflame service"
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Highflame Guardrails"
