from pydantic import Field
from typing_extensions import TypedDict

from .base import GuardrailConfigModel


class JavelinGuardInput(TypedDict):
    text: str


class JavelinGuardRequest(TypedDict):
    input: JavelinGuardInput
    config: dict | None
    metadata: dict | None


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
    assessments: list[
        dict[
            str,
            JavelinPromptInjectionAssessment | JavelinTrustSafetyAssessment | JavelinLanguageDetectionAssessment,
        ]
    ]


class JavelinGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the Javelin guardrail"""

    guard_name: str | None = Field(default=None, description="Name of the Javelin guard to use")
    api_version: str | None = Field(default="v1", description="API version for Javelin service")
    metadata: dict | None = Field(default=None, description="Additional metadata to send with requests")
    application: str | None = Field(default=None, description="Application name for Javelin service")
    config: dict | None = Field(default=None, description="Configuration parameters for Javelin service")

    @staticmethod
    def ui_friendly_name() -> str:
        return "Javelin Guardrails"
