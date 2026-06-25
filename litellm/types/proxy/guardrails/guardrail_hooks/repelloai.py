from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .base import GuardrailConfigModel


class RepelloAIGuardrailConfigModel(GuardrailConfigModel[BaseModel]):
    """Config model for the RepelloAI Argus guardrail."""

    api_key: Optional[str] = Field(
        default=None,
        description="API key for the RepelloAI Argus service. Falls back to ARGUS_API_KEY or REPELLOAI_API_KEY.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for the RepelloAI Argus API. Defaults to https://argusapi.repello.ai/sdk/v1",
    )
    asset_id: Optional[str] = Field(
        default=None,
        description="Repello asset ID whose dashboard policies are enforced. Required; the guardrail raises at init if it is missing.",
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description="What to do when the RepelloAI Argus API is unreachable. 'fail_closed' = block (default), 'fail_open' = allow.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "RepelloAI Argus"


class RepelloAIScanData(TypedDict, total=False):
    """The text payload sent to the RepelloAI Argus analyze endpoints.
    Only one of 'prompt' or 'response' is set per request.
    """

    prompt: Optional[str]
    response: Optional[str]


class RepelloAIAnalyzeRequest(TypedDict, total=False):
    """Request body for POST {api_base}/analyze/{prompt|response}."""

    asset_id: str
    scan_data: RepelloAIScanData


class RepelloAIViolatedPolicy(TypedDict, total=False):
    policy_name: Optional[str]
    policy_id: Optional[str]
    action_taken: Optional[str]
    scope: Optional[str]
    details: Optional[dict[str, object]]
    masked_result: Optional[str]


class RepelloAIAnalyzeResponse(TypedDict, total=False):
    """Response body returned by the RepelloAI Argus analyze endpoints."""

    verdict: Optional[str]  # "blocked" | "flagged" | "passed"
    request_id: Optional[str]
    policies_violated: Optional[List[RepelloAIViolatedPolicy]]
    policies_applied: Optional[List[dict[str, object]]]
