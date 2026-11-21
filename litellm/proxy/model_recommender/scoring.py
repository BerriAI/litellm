from collections.abc import Mapping
from math import log
from typing import Final

from .benchmarks import BenchmarkScore, intelligence_score
from .models import Candidate, Recommendation, RecommendationRequest, Requirements, Signal, Signals, Speed


def matches(candidate: Candidate, requirements: Requirements) -> bool:
    return (
        candidate.mode == "chat"
        and (not requirements.reasoning or candidate.supports_reasoning)
        and (not requirements.vision or candidate.supports_vision)
        and (not requirements.function_calling or candidate.supports_function_calling)
        and (
            requirements.min_context_tokens is None
            or (
                candidate.max_input_tokens is not None and candidate.max_input_tokens >= requirements.min_context_tokens
            )
        )
    )


def blended_cost(candidate: Candidate) -> float | None:
    if candidate.input_cost_per_token is None or candidate.output_cost_per_token is None:
        return None
    return (3 * candidate.input_cost_per_token + candidate.output_cost_per_token) * 250_000


def normalize(value: float | None, values: tuple[float | None, ...], invert: bool = False) -> Signal:
    known: Final = tuple(item for item in values if item is not None)
    if value is None or not known:
        return Signal(raw=value, normalized=0.5, unknown=True)
    low: Final = min(known)
    high: Final = max(known)
    scaled: Final = (value - low) / (high - low) if high != low else 0.5
    return Signal(raw=value, normalized=1 - scaled if invert else scaled, unknown=False)


def cost_signal(value: float | None, values: tuple[float | None, ...]) -> Signal:
    positive: Final = tuple(item for item in values if item is not None and item > 0)
    floor: Final = min(positive) / 10 if positive else 1.0
    transformed: Final = tuple(log(max(item, floor)) if item is not None else None for item in values)
    scaled: Final = normalize(log(max(value, floor)) if value is not None else None, transformed, invert=True)
    return scaled.model_copy(update={"raw": value})


def reasons(signals: Signals, count: int) -> tuple[str, ...]:
    return (
        (
            f"Cheapest of {count} matching models"
            if signals.cost.normalized == 1
            else "Pricing unavailable"
            if signals.cost.unknown
            else f"${signals.cost.raw:.2f} per 1M blended tokens"
        ),
        (
            "Benchmark coverage unavailable for selected focus"
            if signals.intelligence.unknown
            else f"Selected benchmark average: {signals.intelligence.raw:.0%}"
        ),
        (
            "No recent gateway speed data"
            if signals.speed.unknown
            else f"Median gateway latency: {signals.speed.raw:.2f}s"
        ),
    )


def rank(
    candidates: tuple[Candidate, ...],
    request: RecommendationRequest,
    benchmarks: tuple[BenchmarkScore, ...],
    speeds: Mapping[str, Speed],
) -> tuple[Recommendation, ...]:
    costs: Final = tuple(blended_cost(candidate) for candidate in candidates)
    intelligence: Final = tuple(intelligence_score(c.underlying_models, request.focus, benchmarks) for c in candidates)
    latency: Final = tuple(
        speeds[c.model_group].latency_seconds if c.model_group in speeds else None for c in candidates
    )
    return tuple(
        sorted(
            (
                recommendation(c, index, candidates, request, costs, intelligence, latency, speeds)
                for index, c in enumerate(candidates)
            ),
            key=lambda result: (-result.score, result.model_group),
        )
    )[: request.limit]


def recommendation(
    candidate: Candidate,
    index: int,
    candidates: tuple[Candidate, ...],
    request: RecommendationRequest,
    costs: tuple[float | None, ...],
    intelligence: tuple[float | None, ...],
    latency: tuple[float | None, ...],
    speeds: Mapping[str, Speed],
) -> Recommendation:
    benchmark: Final = intelligence[index]
    signals: Final = Signals(
        intelligence=Signal(
            raw=benchmark,
            normalized=benchmark if benchmark is not None else 0.5,
            unknown=benchmark is None,
        ),
        speed=normalize(latency[index], latency, invert=True),
        cost=cost_signal(costs[index], costs),
    )
    weights: Final = request.priorities
    total: Final = weights.intelligence + weights.speed + weights.cost
    weighted: Final = (
        weights.intelligence * signals.intelligence.normalized
        + weights.speed * signals.speed.normalized
        + weights.cost * signals.cost.normalized
    )
    speed: Final = speeds.get(candidate.model_group)
    return Recommendation(
        model_group=candidate.model_group,
        on_gateway=candidate.on_gateway,
        providers=candidate.providers,
        score=round(100 * weighted / total, 2) if total else 50,
        signals=signals,
        latency_seconds=latency[index],
        output_tokens_per_second=speed.output_tokens_per_second if speed else None,
        capabilities=Requirements(
            reasoning=candidate.supports_reasoning,
            vision=candidate.supports_vision,
            function_calling=candidate.supports_function_calling,
        ),
        context_window=candidate.max_input_tokens,
        input_cost_per_token=candidate.input_cost_per_token,
        output_cost_per_token=candidate.output_cost_per_token,
        reasons=reasons(signals, len(candidates)),
    )
