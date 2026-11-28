"""Types for the Epistemic Guard post-call tool-receipt verification guardrail."""

from typing import Literal, TypedDict

from pydantic import Field

from .base import GuardrailConfigModel

EpistemicGuardActionTaken = Literal["block", "rewrite", "allow", "log_only"]


class EpistemicGuardDetection(TypedDict, total=False):
    """Detection output for a single post-call narration check."""

    type: Literal["ungrounded_action_claim"]
    unsupported_claims: list[str]
    clauses_checked: int
    device: str
    latency_ms: float
    action_taken: EpistemicGuardActionTaken


class EpistemicGuardConfigModel(GuardrailConfigModel):
    """Configuration for the Epistemic Guard guardrail."""

    on_fail: Literal["block", "rewrite", "log_only"] = Field(
        default="block",
        description=(
            "'block' raises HTTP 400 on ungrounded action claims; "
            "'rewrite' replaces the response with a retry prompt; "
            "'log_only' records detections without blocking."
        ),
    )
    verify_output_values: bool = Field(
        default=False,
        description=(
            "When True, also verify that concrete values reported by the assistant "
            "match the corresponding tool output receipts."
        ),
    )
    allow_creative_context: bool = Field(
        default=False,
        description=(
            "When True, exempts creative/grammar-explanation prompts with no tool receipts "
            "from verification. Defaults to False so untrusted user text cannot disable enforcement."
        ),
    )
    device: str | None = Field(
        default=None,
        description=(
            "Optional inference device passed to epistemic_guard.ReceiptSupervisor "
            "(e.g. 'CPU', 'NPU', or None for auto)."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Epistemic Guard"
