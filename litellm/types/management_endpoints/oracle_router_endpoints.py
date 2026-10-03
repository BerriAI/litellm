"""Types for the ORACLE router management endpoints."""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field


class OracleRouterFeedbackRequest(BaseModel):
    """Report that a program finished. ``score`` is optional when the router's verifier grades it."""

    program_id: str = Field(min_length=1)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    cost: float | None = Field(
        default=None, ge=0.0, description="Overrides the spend LiteLLM accumulated for the program"
    )
    payload: Mapping[str, object] = Field(default_factory=dict, description="Passed to the verifier unchanged")


class OracleRouterFeedbackResponse(BaseModel):
    program_id: str
    router_name: str
    model: str
    pending_verifications: int
    deferred: bool = Field(
        default=False,
        description="No response of the program has been observed yet; its next one completes it with this feedback",
    )


class OracleRouterStateResponse(BaseModel):
    routers: Sequence[Mapping[str, object]]
