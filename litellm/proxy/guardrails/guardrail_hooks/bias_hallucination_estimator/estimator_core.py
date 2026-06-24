from __future__ import annotations

from litellm.types.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator import (
    BiasAnalysis,
    HallucinationAnalysis,
)

from .patterns import (
    BIAS_PATTERNS,
    CITATION_GAP_PATTERNS,
    FABRICATED_SPECIFICITY_PATTERNS,
    SOURCE_INDICATOR_PATTERN,
    UNSOURCED_STATISTIC_PATTERN,
    PatternRule,
)
from .utils import clip_example, split_sentences, unique_preserve_order


class BiasDetector:
    def detect(self, text: str) -> BiasAnalysis:
        return self.detect_bias(text)

    def detect_bias(self, text: str) -> BiasAnalysis:
        matches = self._match_patterns(text, BIAS_PATTERNS)
        score = min(sum(match[2] for match in matches), 1.0)
        patterns_found = unique_preserve_order(match[0] for match in matches)
        examples = unique_preserve_order(match[1] for match in matches)

        return BiasAnalysis(
            bias_detected=bool(matches),
            score=round(score, 3),
            patterns_found=list(patterns_found),
            examples=list(examples),
            reasoning=self._build_reasoning(patterns_found),
        )

    @staticmethod
    def _match_patterns(
        text: str, patterns: tuple[PatternRule, ...]
    ) -> tuple[tuple[str, str, float], ...]:
        return tuple(
            (rule.name, clip_example(match.group(0)), rule.score)
            for rule in patterns
            for match in rule.pattern.finditer(text)
        )

    @staticmethod
    def _build_reasoning(patterns_found: tuple[str, ...]) -> str:
        if not patterns_found:
            return "No bias indicators found."
        return f"Detected bias indicators: {', '.join(patterns_found)}."


class HallucinationDetector:
    def detect(self, text: str) -> HallucinationAnalysis:
        return self.detect_hallucination(text)

    def detect_hallucination(self, text: str) -> HallucinationAnalysis:
        sentences = split_sentences(text)
        unsourced_claims = self._find_unsourced_statistics(sentences)
        missing_citations = self._find_rule_matches(text, CITATION_GAP_PATTERNS)
        fabricated_specificity = self._find_rule_matches(
            text, FABRICATED_SPECIFICITY_PATTERNS
        )
        patterns_found = self._patterns_found(
            has_unsourced_claims=bool(unsourced_claims),
            has_missing_citations=bool(missing_citations),
            has_fabricated_specificity=bool(fabricated_specificity),
        )
        score = self._score(
            unsourced_claims=unsourced_claims,
            missing_citations=missing_citations,
            fabricated_specificity=fabricated_specificity,
        )
        examples = unique_preserve_order(
            tuple(unsourced_claims)
            + tuple(missing_citations)
            + tuple(fabricated_specificity)
        )

        return HallucinationAnalysis(
            hallucination_detected=score > 0,
            score=round(score, 3),
            patterns_found=list(patterns_found),
            examples=list(examples),
            unsourced_claims=list(unsourced_claims),
            fabricated_specificity=list(fabricated_specificity),
            missing_citations=list(missing_citations),
            reasoning=self._build_reasoning(patterns_found),
        )

    @staticmethod
    def _find_unsourced_statistics(sentences: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            clip_example(sentence)
            for sentence in sentences
            if UNSOURCED_STATISTIC_PATTERN.pattern.search(sentence)
            and SOURCE_INDICATOR_PATTERN.search(sentence) is None
        )

    @staticmethod
    def _find_rule_matches(
        text: str, rules: tuple[PatternRule, ...]
    ) -> tuple[str, ...]:
        return unique_preserve_order(
            clip_example(match.group(0))
            for rule in rules
            for match in rule.pattern.finditer(text)
        )

    @staticmethod
    def _patterns_found(
        *,
        has_unsourced_claims: bool,
        has_missing_citations: bool,
        has_fabricated_specificity: bool,
    ) -> tuple[str, ...]:
        return tuple(
            pattern_name
            for pattern_name, detected in (
                ("unsourced_statistics", has_unsourced_claims),
                ("missing_citations", has_missing_citations),
                ("fabricated_specificity", has_fabricated_specificity),
            )
            if detected
        )

    @staticmethod
    def _score(
        *,
        unsourced_claims: tuple[str, ...],
        missing_citations: tuple[str, ...],
        fabricated_specificity: tuple[str, ...],
    ) -> float:
        unsourced_score = min(len(unsourced_claims) * 0.32, 0.64)
        citation_score = min(len(missing_citations) * 0.3, 0.6)
        specificity_score = min(len(fabricated_specificity) * 0.22, 0.44)
        return min(unsourced_score + citation_score + specificity_score, 1.0)

    @staticmethod
    def _build_reasoning(patterns_found: tuple[str, ...]) -> str:
        if not patterns_found:
            return "No hallucination risk indicators found."
        return f"Detected hallucination risk indicators: {', '.join(patterns_found)}."
