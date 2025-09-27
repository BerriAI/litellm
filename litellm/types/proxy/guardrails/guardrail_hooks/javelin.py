from typing import Dict, List, Optional

from typing_extensions import TypedDict


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
