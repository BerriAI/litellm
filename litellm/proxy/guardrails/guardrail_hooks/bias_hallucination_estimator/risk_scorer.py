from __future__ import annotations

from typing import List, Literal, Optional

from litellm.types.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator import (
    BiasAnalysis,
    HallucinationAnalysis,
    RiskScore,
    UncertaintyAnalysis,
)


class RiskScorer:
    def __init__(
        self,
        *,
        use_logprobs: bool = False,
        bias_weight: float = 0.4,
        hallucination_weight: float = 0.6,
        uncertainty_weight: float = 0.0,
        bias_threshold: float = 0.5,
        hallucination_threshold: float = 0.5,
        flag_threshold: float = 0.25,
        block_threshold: float = 0.5,
    ) -> None:
        self.use_logprobs = use_logprobs
        self.bias_weight = bias_weight
        self.hallucination_weight = hallucination_weight
        self.uncertainty_weight = uncertainty_weight if use_logprobs else 0.0
        self.bias_threshold = bias_threshold
        self.hallucination_threshold = hallucination_threshold
        self.flag_threshold = flag_threshold
        self.block_threshold = block_threshold

    def compute_risk(
        self,
        bias_analysis: BiasAnalysis,
        hallucination_analysis: HallucinationAnalysis,
        uncertainty_analysis: Optional[UncertaintyAnalysis] = None,
    ) -> RiskScore:
        uncertainty_score = uncertainty_analysis.score if uncertainty_analysis else 0.0
        overall_risk = self._weighted_score(
            bias_score=bias_analysis.score,
            hallucination_score=hallucination_analysis.score,
            uncertainty_score=uncertainty_score,
        )
        detected_issues = self.determine_detected_issues(
            bias_analysis=bias_analysis,
            hallucination_analysis=hallucination_analysis,
            uncertainty_analysis=uncertainty_analysis,
        )

        return RiskScore(
            overall_risk_percentage=round(overall_risk * 100),
            bias_score=bias_analysis.score,
            hallucination_score=hallucination_analysis.score,
            uncertainty_score=uncertainty_score,
            detected_issues=detected_issues,
            recommendation=self.determine_recommendation(
                overall_risk=overall_risk,
                bias_score=bias_analysis.score,
                hallucination_score=hallucination_analysis.score,
                uncertainty_score=uncertainty_score,
            ),
        )

    def _weighted_score(
        self,
        *,
        bias_score: float,
        hallucination_score: float,
        uncertainty_score: float,
    ) -> float:
        total_weight = self.bias_weight + self.hallucination_weight + self.uncertainty_weight
        if total_weight <= 0:
            return 0.0
        return min(
            (
                bias_score * self.bias_weight
                + hallucination_score * self.hallucination_weight
                + uncertainty_score * self.uncertainty_weight
            )
            / total_weight,
            1.0,
        )

    def determine_recommendation(
        self,
        *,
        overall_risk: float,
        bias_score: float,
        hallucination_score: float,
        uncertainty_score: float,
    ) -> Literal["pass", "flag", "block"]:
        if (
            overall_risk >= self.block_threshold
            or bias_score >= self.bias_threshold
            or hallucination_score >= self.hallucination_threshold
            or (self.use_logprobs and self.uncertainty_weight > 0 and uncertainty_score >= self.block_threshold)
        ):
            return "block"
        if overall_risk >= self.flag_threshold:
            return "flag"
        return "pass"

    @staticmethod
    def determine_detected_issues(
        *,
        bias_analysis: BiasAnalysis,
        hallucination_analysis: HallucinationAnalysis,
        uncertainty_analysis: Optional[UncertaintyAnalysis],
    ) -> List[str]:
        issues: List[str] = []
        issues.extend(f"bias:{pattern}" for pattern in bias_analysis.patterns_found)
        issues.extend(f"hallucination:{pattern}" for pattern in hallucination_analysis.patterns_found)
        if uncertainty_analysis and uncertainty_analysis.uncertainty_detected:
            issues.extend(f"uncertainty:{pattern}" for pattern in uncertainty_analysis.patterns_found)
        return issues
