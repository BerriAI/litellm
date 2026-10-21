import fnmatch
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

_GLOB_CHARS: Final = frozenset("*?[")
_GLOB_TOKEN: Final = re.compile(
    r"\[!\][^\]]*\]|\[\][^\]]*\]|\[!(?!\])[^\]]*\]|\[(?![!\]])[^\]]*\]|.",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class CostDiscountKey:
    provider: str
    model_pattern: str | None


def parse_cost_discount_key(key: str) -> CostDiscountKey:
    provider, separator, pattern = key.partition("/")
    return CostDiscountKey(provider=provider, model_pattern=pattern if separator else None)


def _literal_length(pattern: str) -> int:
    return sum(1 for token in _GLOB_TOKEN.findall(pattern) if len(token) == 1 and token not in "*?")


def _model_names(model: str, provider_prefix: str) -> tuple[str, ...]:
    prefix_run: Final = re.match(f"(?:{re.escape(provider_prefix)})*", model)
    depth: Final = len(prefix_run.group()) // len(provider_prefix) if prefix_run else 0
    if depth == 0:
        return (model,)
    return tuple(model[len(provider_prefix) * level :] for level in range(1, depth + 1))


def resolve_cost_discount(
    cost_discount_config: Mapping[str, float],
    custom_llm_provider: str | None,
    model: str | None,
    region_name: str | None = None,
) -> float | None:
    if not custom_llm_provider:
        return None

    provider_prefix: Final = f"{custom_llm_provider}/"
    region_prefix: Final = f"{provider_prefix}{region_name}/" if region_name else ""
    model_names: Final = (
        ()
        if model is None
        else _model_names(model, provider_prefix)
        + (
            _model_names(f"{provider_prefix}{model[len(region_prefix) :]}", provider_prefix)
            if region_prefix and model.startswith(region_prefix)
            else ()
        )
    )
    patterns: Final = tuple(
        parsed.model_pattern
        for parsed in (parse_cost_discount_key(key) for key in cost_discount_config)
        if parsed.provider == custom_llm_provider and parsed.model_pattern is not None
    )

    exact_patterns: Final = frozenset(pattern for pattern in patterns if _GLOB_CHARS.isdisjoint(pattern))
    exact: Final = next((name for name in model_names if name in exact_patterns), None)
    if exact is not None:
        return cost_discount_config[f"{custom_llm_provider}/{exact}"]
    prefixed_markers: Final = (provider_prefix,) if not region_name else (provider_prefix, f"{region_name}/")
    matches: Final = tuple(
        pattern
        for pattern in patterns
        if not _GLOB_CHARS.isdisjoint(pattern)
        and any(
            fnmatch.fnmatchcase(name, pattern)
            for name in model_names
            if "/" in pattern or not name.startswith(prefixed_markers)
        )
    )
    if matches:
        best_match: Final = max(matches, key=_literal_length)
        return cost_discount_config[f"{custom_llm_provider}/{best_match}"]

    return cost_discount_config.get(custom_llm_provider)
