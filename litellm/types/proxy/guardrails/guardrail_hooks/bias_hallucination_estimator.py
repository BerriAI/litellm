from typing import Literal

from pydantic import BaseModel, Field

from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class BiasAnalysis(BaseModel):
    """Model representing the result of a bias analysis."""

    bias_detected: bool = Field(default=False, description="Indicates if bias was detected in the text")
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score representing the level of bias detected",
    )
    patterns_found: list[str] = Field(
        default_factory=list,
        description="List of patterns or phrases that indicate bias in the text",
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Examples of text segments that were identified as biased",
    )
    reasoning: str = Field(
        default="",
        description="Explanation of the reasoning behind the bias detection and scoring",
    )


class HallucinationAnalysis(BaseModel):
    """Model representing the result of a hallucination analysis."""

    hallucination_detected: bool = Field(
        default=False,
        description="Indicates if hallucination risk was detected in the text",
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score representing the level of hallucination risk detected",
    )
    patterns_found: list[str] = Field(
        default_factory=list,
        description="List of patterns or phrases that indicate hallucination risk",
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Examples of text segments that were identified as risky",
    )
    unsourced_claims: list[str] = Field(default_factory=list)
    fabricated_specificity: list[str] = Field(default_factory=list)
    missing_citations: list[str] = Field(default_factory=list)
    reasoning: str = Field(
        default="",
        description="Explanation of the reasoning behind the hallucination risk score",
    )


class UncertaintyAnalysis(BaseModel):
    """Model representing the result of an uncertainty analysis based on logprobs."""

    uncertainty_detected: bool = Field(
        default=False,
        description="Indicates if high uncertainty was detected in the text based on logprobs",
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score representing the level of uncertainty detected based on logprobs",
    )
    patterns_found: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")


class RiskScore(BaseModel):
    """Model representing the overall risk score combining bias and hallucination."""

    overall_risk_percentage: int = Field(default=0, ge=0, le=100, description="Overall risk percentage (0-100)")
    bias_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_score: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    detected_issues: list[str] = Field(default_factory=list)
    recommendation: Literal["pass", "flag", "block"] = Field(default="pass")


class BiasHallucinationEstimatorConfigModel(GuardrailConfigModel):  # pyright: ignore[reportMissingTypeArgument]
    """Configuration schema for the native bias and hallucination estimator."""

    bias_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    hallucination_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_flag_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    risk_block_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    block_on_high_risk: bool = True
    log_only: bool = False
    check_request: bool = False
    check_response: bool = True
    violation_message: str | None = None
    bias_weight: float = Field(default=0.4, ge=0.0)
    hallucination_weight: float = Field(default=0.6, ge=0.0)

    @staticmethod
    def ui_friendly_name() -> str:
        return "LiteLLM Bias & Hallucination Estimator"
