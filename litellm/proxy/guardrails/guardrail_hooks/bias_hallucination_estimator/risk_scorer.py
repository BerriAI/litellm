from __future__ import annotations

from typing import Literal

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
        bias_weight: float = 0.4,
        hallucination_weight: float = 0.6,
        bias_threshold: float = 0.5,
        hallucination_threshold: float = 0.5,
        flag_threshold: float = 0.25,
        block_threshold: float = 0.5,
    ) -> None:
        self.bias_weight = bias_weight
        self.hallucination_weight = hallucination_weight
        self.bias_threshold = bias_threshold
        self.hallucination_threshold = hallucination_threshold
        self.flag_threshold = flag_threshold
        self.block_threshold = block_threshold

    def compute_risk(
        self,
        bias_analysis: BiasAnalysis,
        hallucination_analysis: HallucinationAnalysis,
        uncertainty_analysis: UncertaintyAnalysis | None = None,
    ) -> RiskScore:
        overall_risk = self._weighted_score(
            bias_score=bias_analysis.score,
            hallucination_score=hallucination_analysis.score,
        )
        return RiskScore(
            overall_risk_percentage=round(overall_risk * 100),
            bias_score=bias_analysis.score,
            hallucination_score=hallucination_analysis.score,
            detected_issues=self._detected_issues(
                bias_analysis=bias_analysis,
                hallucination_analysis=hallucination_analysis,
                uncertainty_analysis=uncertainty_analysis,
            ),
            recommendation=self.determine_recommendation(
                overall_risk=overall_risk,
                bias_score=bias_analysis.score,
                hallucination_score=hallucination_analysis.score,
            ),
        )

    def _weighted_score(
        self,
        *,
        bias_score: float,
        hallucination_score: float,
    ) -> float:
        total_weight = self.bias_weight + self.hallucination_weight
        if total_weight <= 0:
            return 0.0
        return min(
            (
                bias_score * self.bias_weight
                + hallucination_score * self.hallucination_weight
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
    ) -> Literal["pass", "flag", "block"]:
        if (
            overall_risk >= self.block_threshold
            or bias_score >= self.bias_threshold
            or hallucination_score >= self.hallucination_threshold
        ):
            return "block"
        if overall_risk >= self.flag_threshold:
            return "flag"
        return "pass"

    @staticmethod
    def _detected_issues(
        *,
        bias_analysis: BiasAnalysis,
        hallucination_analysis: HallucinationAnalysis,
        uncertainty_analysis: UncertaintyAnalysis | None,
    ) -> list[str]:
        uncertainty_issues: tuple[str, ...] = (
            tuple(f"uncertainty:{p}" for p in uncertainty_analysis.patterns_found)
            if uncertainty_analysis and uncertainty_analysis.uncertainty_detected
            else ()
        )
        return list(
            tuple(f"bias:{p}" for p in bias_analysis.patterns_found)
            + tuple(f"hallucination:{p}" for p in hallucination_analysis.patterns_found)
            + uncertainty_issues
        )
