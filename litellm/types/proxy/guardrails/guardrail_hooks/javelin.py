from typing import Dict, List, Optional

from pydantic import Field
from typing_extensions import TypedDict

from .base import GuardrailConfigModel


class JavelinGuardInput(TypedDict):
    text: str


class JavelinGuardRequest(TypedDict):
    input: JavelinGuardInput
    config: Optional[Dict]
    metadata: Optional[Dict]


class JavelinPromptInjectionCategories(TypedDict):
    prompt_injection: bool
    jailbreak: bool


class JavelinPromptInjectionCategoryScores(TypedDict):
    prompt_injection: float
    jailbreak: float


class JavelinPromptInjectionResults(TypedDict):
    categories: JavelinPromptInjectionCategories
    category_scores: JavelinPromptInjectionCategoryScores
    reject_prompt: str


class JavelinPromptInjectionAssessment(TypedDict):
    results: JavelinPromptInjectionResults
    request_reject: bool


class JavelinTrustSafetyCategories(TypedDict):
    violence: bool
    weapons: bool
    hate_speech: bool
    crime: bool
    sexual: bool
    profanity: bool


class JavelinTrustSafetyCategoryScores(TypedDict):
    violence: float
    weapons: float
    hate_speech: float
    crime: float
    sexual: float
    profanity: float


class JavelinTrustSafetyResults(TypedDict):
    categories: JavelinTrustSafetyCategories
    category_scores: JavelinTrustSafetyCategoryScores


class JavelinTrustSafetyAssessment(TypedDict):
    results: JavelinTrustSafetyResults
    request_reject: bool


class JavelinLanguageDetectionResults(TypedDict):
    lang: str
    prob: float


class JavelinLanguageDetectionAssessment(TypedDict):
    results: JavelinLanguageDetectionResults
    request_reject: bool


class JavelinGuardResponse(TypedDict):
    assessments: List[
        Dict[
            str,
            JavelinPromptInjectionAssessment
            | JavelinTrustSafetyAssessment
            | JavelinLanguageDetectionAssessment,
        ]
    ]


class JavelinGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the Javelin guardrail"""

    guard_name: Optional[str] = Field(
        default=None, description="Name of the Javelin guard to use"
    )
    api_version: Optional[str] = Field(
        default="v1", description="API version for Javelin service"
    )
    metadata: Optional[Dict] = Field(
        default=None, description="Additional metadata to send with requests"
    )
    application: Optional[str] = Field(
        default=None, description="Application name for Javelin service"
    )
    config: Optional[Dict] = Field(
        default=None, description="Configuration parameters for Javelin service"
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Javelin Guardrails"
