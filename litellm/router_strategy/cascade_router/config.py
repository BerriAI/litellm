"""Configuration for the cascade router."""

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from litellm.types.llms.base import LiteLLMBaseModel


class CascadeRouterConfig(LiteLLMBaseModel):
    """Cheap deployment first; verify its answer; escalate to the strong deployment when it falls short.

    `cheap_model`, `strong_model` and `verifier_model` are model group names registered on the same
    Router (every inner call is routed by the Router: load balancing, cooldowns, retries). The
    verifier, when set, is asked one question about the cheap answer; a decision model answers it
    through `Router.adecisions`, a chat model through a one-word YES/NO prompt.
    """

    model_config = ConfigDict(frozen=True)

    cheap_model: str = Field(description="Model group answered first on every request.")
    strong_model: str = Field(description="Model group called when the cheap answer does not pass.")
    verifier_model: str | None = Field(
        default=None,
        description=(
            "Model group that judges the cheap answer after the heuristics pass. Unset means heuristics only."
        ),
    )
    verifier_kind: Literal["decision", "chat"] = Field(
        default="decision",
        description=(
            "How the verifier is asked: 'decision' sends a choice question through litellm.adecisions "
            "(decision models such as Laya, LM Studio or Azure decisions); 'chat' asks a chat model for YES or NO."
        ),
    )
    verifier_question: str = Field(
        default=(
            "Does the answer actually answer the user's request: correct, complete enough, on topic, and not a refusal?"
        ),
        description="The question put to the verifier about the cheap answer.",
    )
    verifier_min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="For decision verifiers: a 'yes' below this confidence escalates anyway.",
    )
    min_chars: int = Field(
        default=20, ge=0, description="A cheap answer shorter than this escalates without a verifier call."
    )
    escalate_on_refusal: bool = Field(default=True, description="Escalate when the cheap answer reads as a refusal.")
    escalate_on_truncation: bool = Field(
        default=True,
        description=(
            "Escalate when the cheap answer stopped on the model's token limit (finish_reason=length). "
            "A request that set its own max_tokens is never escalated for truncation: the strong model "
            "would be cut at the same point."
        ),
    )
    on_verifier_error: Literal["escalate", "accept"] = Field(
        default="escalate",
        description="What to do when the verifier call fails or returns nothing usable.",
    )
    on_strong_error: Literal["raise", "cheap"] = Field(
        default="raise",
        description=(
            "What to do when the escalation call to strong_model fails: 'raise' fails the request with the "
            "strong model's error; 'cheap' returns the cheap answer that was judged inadequate, with cause "
            "cascade_strong_error, so an outage of the strong tier degrades the alias instead of failing it."
        ),
    )

    @model_validator(mode="after")
    def _distinct_models(self) -> "CascadeRouterConfig":
        if self.cheap_model == self.strong_model:
            raise ValueError("cascade_router_config: cheap_model and strong_model must differ")
        return self
