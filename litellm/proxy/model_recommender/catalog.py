from collections.abc import Mapping
from itertools import chain
from typing import Final

from pydantic import ValidationError

from .benchmarks import normalized_id
from .models import Candidate, FrozenModel

PROVIDERS: Final = ("openai", "anthropic", "gemini", "groq", "deepseek", "mistral", "xai", "cerebras")


class CatalogEntry(FrozenModel):
    mode: str | None = None
    litellm_provider: str | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    max_input_tokens: float | None = None
    supports_reasoning: bool | None = None
    supports_vision: bool | None = None
    supports_function_calling: bool | None = None


def catalog_candidate(name: str, raw: object) -> Candidate | None:
    try:
        entry: Final = CatalogEntry.model_validate(raw)
    except ValidationError:
        return None
    if entry.mode != "chat" or entry.litellm_provider is None or entry.litellm_provider not in PROVIDERS:
        return None
    if any(price is None or price < 0 for price in (entry.input_cost_per_token, entry.output_cost_per_token)):
        return None
    return Candidate(
        model_group=name if "/" in name else f"{entry.litellm_provider}/{name}",
        on_gateway=False,
        providers=(entry.litellm_provider,),
        underlying_models=(name,),
        mode="chat",
        input_cost_per_token=entry.input_cost_per_token,
        output_cost_per_token=entry.output_cost_per_token,
        max_input_tokens=entry.max_input_tokens,
        supports_reasoning=bool(entry.supports_reasoning),
        supports_vision=bool(entry.supports_vision),
        supports_function_calling=bool(entry.supports_function_calling),
    )


def external_candidates(catalog: Mapping[str, object], gateway: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
    configured: Final = frozenset(
        map(normalized_id, chain.from_iterable((c.model_group, *c.underlying_models) for c in gateway))
    )
    parsed: Final = tuple(
        candidate
        for name, raw in sorted(catalog.items(), reverse=True)
        if normalized_id(name) not in configured and (candidate := catalog_candidate(name, raw)) is not None
    )
    unique: Final = tuple({normalized_id(c.model_group): c for c in parsed}.values())
    return tuple(
        chain.from_iterable(tuple(c for c in unique if c.providers == (provider,))[:40] for provider in PROVIDERS)
    )
