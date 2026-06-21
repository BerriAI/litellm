from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from litellm._logging import verbose_logger

from .data_sources import DataSource, DataSourceResult


@dataclass
class GroundingResult:
    claim: str
    is_grounded: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    supporting_docs: List[Dict[str, Any]] = field(default_factory=list) 
    sources_searched: List[str] = field(default_factory=list)
    missing_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GroundingChecker:
    def __init__(
        self,
        data_sources: List[DataSource],
        confidence_threshold: float = 0.6,
        timeout_per_source: float = 5.0,
    ) -> None:
        self.data_sources = sorted(data_sources, key=lambda x: -x.priority)
        self.confidence_threshold = confidence_threshold
        self.timeout_per_source = timeout_per_source

    async def check_claim_grounding(self, claim: str) -> GroundingResult:
        if not claim or not claim.strip():
            return GroundingResult(claim=claim, reasoning="Claim is empty")

        claim_elements = self._extract_verifiable_elements(claim)
        if not any(claim_elements.values()):
            return GroundingResult(claim=claim, reasoning="No verifiable elements found in claim")

        enabled_sources = [ds for ds in self.data_sources if ds.enabled]
        if not enabled_sources:
            return GroundingResult(claim=claim, reasoning="No enabled data sources available")

        raw_results = await asyncio.gather(
            *[self._search_source_safe(source, claim) for source in enabled_sources],
            return_exceptions=True,
        )

        supporting_docs: List[Dict[str, Any]] = []
        sources_searched: List[str] = [ds.name for ds in enabled_sources]
        missing_sources: List[str] = []
        max_confidence = 0.0

        for source, result in zip(enabled_sources, raw_results):
            if isinstance(result, BaseException):
                verbose_logger.warning(f"Data source {source.name} failed: {result}")
                missing_sources.append(source.name)
                continue
            for item in result:
                boosted = self._boost_confidence(item, claim_elements)
                supporting_docs.append(
                    {
                        "text": item.text[:200],
                        "source": item.source,
                        "confidence": boosted,
                        "match_score": item.confidence,
                    }
                )
                max_confidence = max(max_confidence, boosted)

        is_grounded = max_confidence >= self.confidence_threshold
        if is_grounded:
            reasoning = f"Claim verified with {max_confidence:.1%} confidence across {len(supporting_docs)} sources"
        elif supporting_docs:
            reasoning = f"Partial match found but confidence ({max_confidence:.1%}) below threshold ({self.confidence_threshold:.1%})"
        else:
            reasoning = f"No matching data found in {len(enabled_sources)} sources"

        return GroundingResult(
            claim=claim,
            is_grounded=is_grounded,
            confidence=max_confidence,
            supporting_docs=supporting_docs,
            sources_searched=sources_searched,
            missing_sources=missing_sources,
            reasoning=reasoning,
        )

    async def verify_multiple_claims(self, claims: List[str]) -> List[GroundingResult]:
        return list(await asyncio.gather(*[self.check_claim_grounding(c) for c in claims]))

    async def _search_source_safe(self, source: DataSource, claim: str) -> List[DataSourceResult]:
        try:
            return await asyncio.wait_for(source.search(claim, limit=3), timeout=self.timeout_per_source)
        except asyncio.TimeoutError:
            verbose_logger.warning(f"Timeout searching {source.name} for claim: {claim}")
            return []
        except Exception as e:
            verbose_logger.warning(f"Error searching {source.name}: {e}")
            return []

    @staticmethod
    def _extract_verifiable_elements(claim: str) -> Dict[str, List[str]]:
        stop_words = frozenset(
            (
                "the", "a", "an", "and", "or", "but", "in", "is", "are", "was",
                "be", "been", "being", "have", "has", "had", "do", "does", "did",
            )
        )
        numbers = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\s?%?|\b\d{4}\b", claim)
        dates = re.findall(r"\b(?:\d{1,2}/\d{1,2}/\d{4}|\d{4})\b", claim)
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim)
        keywords = [w for w in re.findall(r"\b[a-z]+\b", claim.lower()) if w not in stop_words and len(w) > 3]
        return {
            "numbers": numbers[:5],
            "dates": dates[:3],
            "entities": entities[:5],
            "keywords": keywords[:8],
        }

    @staticmethod
    def _boost_confidence(result: DataSourceResult, claim_elements: Dict[str, List[str]]) -> float:
        boost = 0.0
        result_numbers = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\s?%?|\b\d{4}\b", result.text))
        if set(claim_elements.get("numbers", [])) & result_numbers:
            boost += 0.1
        result_entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", result.text))
        if set(claim_elements.get("entities", [])) & result_entities:
            boost += 0.15
        result_words = set(re.findall(r"\b[a-z]+\b", result.text.lower()))
        claim_keywords = set(claim_elements.get("keywords", []))
        if claim_keywords and result_words:
            boost += min(0.2, len(claim_keywords & result_words) / len(claim_keywords) * 0.25)
        return round(min(1.0, result.confidence + boost), 2)
