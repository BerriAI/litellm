"""Sync the openrouter and vercel_ai_gateway entries of model_prices_and_context_window.json with the live catalogs.

Pulls ``GET https://openrouter.ai/api/v1/models`` and ``GET https://ai-gateway.vercel.sh/v1/models``, maps the
catalog fields onto registry fields, and diffs the result against the registry. Dry run (the default) prints the
diff summary and the generated PR body; ``--write`` applies the changes to the root cost map and its ``litellm/``
backup copy.

Policy:
- Both catalogs price per token as decimal strings; values are normalized to six significant digits.
- Vercel long-context tiers map to the registry's ``*_above_<N>k_tokens`` keys, which litellm applies once the
  prompt exceeds N thousand tokens. A row whose tier boundaries are not whole thousands is skipped with a warning.
- A Vercel price flagged ``varies_by_provider`` is only a headline: it seeds a new entry, marked
  ``price_varies_by_provider``, and keeps a marked entry in sync, but never overwrites the price of an entry without
  the mark; that difference is reported as a warning. A human who curates one provider's own price drops the mark.
- Image and audio output are priced from the catalog's per-token ``image_output`` and ``audio_output`` prices. A row
  whose non-text output the catalog does not price per token is skipped.
- A new entry inherits the traits no catalog expresses (adaptive thinking, sampling params, cache minimums, system
  messages) from the same model's root registry entry, found by the bare model name in any mode or else its
  longest dash-prefix with the same mode, so the family-wide invariants the test suite enforces hold for the route.
- An existing entry only gains or changes the fields the catalog expresses. Nothing is ever removed and a
  capability flag the catalog does not claim stays as curated. ``max_output_tokens`` and ``max_tokens`` move as a
  pair and only when the catalog states an output ceiling; a curated output cap equal to the entry's own context
  window is a copy of that window, not a ceiling, so the catalog's ceiling replaces it unless that would shrink it
  more than 10x.
- A limit that would shrink, a price that would cross zero, a price that would move more than 10x either way, and
  a capability flag curated as false are held back as warnings for a human instead of applied.
- Router models and rows without a usable prompt and completion price are skipped.
- A registry entry absent from its catalog is left untouched; retiring a model stays a human call.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

COST_MAP_RELPATHS: Final = (
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
)
OPENROUTER_MODELS_URL: Final = "https://openrouter.ai/api/v1/models"
VERCEL_MODELS_URL: Final = "https://ai-gateway.vercel.sh/v1/models"
VERCEL_TYPE_TO_MODE: Final = MappingProxyType({"language": "chat", "embedding": "embedding"})
LIMIT_PAIR: Final = ("max_output_tokens", "max_tokens")
SWING_LIMIT: Final = 10
INHERITED_TRAITS: Final = frozenset(
    {
        "prompt_cache_min_tokens",
        "supports_adaptive_thinking",
        "supports_sampling_params",
        "supports_system_messages",
        "thinking_always_on",
    }
)
PR_BODY_SECTION_LIMIT: Final = 30
GITHUB_BODY_LIMIT: Final = 65_536

Provider = Literal["openrouter", "vercel_ai_gateway"]
RegistryEntry = dict[str, object]
CostMap = dict[str, object]
Prices = Mapping[str, float]

NO_PRICES: Final[Prices] = MappingProxyType({})
NO_TRAITS: Final[Mapping[str, object]] = MappingProxyType({})


class SyncError(RuntimeError):
    pass


class OpenRouterPricing(BaseModel):
    prompt: str
    completion: str
    input_cache_read: str | None = None
    input_cache_write: str | None = None
    internal_reasoning: str | None = None
    image_output: str | None = None
    audio: str | None = None
    audio_output: str | None = None


class OpenRouterArchitecture(BaseModel):
    input_modalities: tuple[str, ...] | None = None
    output_modalities: tuple[str, ...] | None = None


class OpenRouterTopProvider(BaseModel):
    max_completion_tokens: int | None = None


class OpenRouterModel(BaseModel):
    id: str
    context_length: int | None = None
    architecture: OpenRouterArchitecture | None = None
    top_provider: OpenRouterTopProvider = OpenRouterTopProvider()
    pricing: OpenRouterPricing
    supported_parameters: tuple[str, ...] | None = None


class VercelTier(BaseModel):
    cost: str
    min: int | None = None
    max: int | None = None


class VercelPricing(BaseModel):
    input: str | None = None
    output: str | None = None
    input_cache_read: str | None = None
    input_cache_write: str | None = None
    input_tiers: tuple[VercelTier, ...] | None = None
    output_tiers: tuple[VercelTier, ...] | None = None
    input_cache_read_tiers: tuple[VercelTier, ...] | None = None
    input_cache_write_tiers: tuple[VercelTier, ...] | None = None
    audio_input_token_cost: str | None = None
    audio_output_token_cost: str | None = None
    varies_by_provider: bool = False


class VercelModalities(BaseModel):
    input: tuple[str, ...] | None = None
    output: tuple[str, ...] | None = None


class VercelModel(BaseModel):
    id: str
    type: str
    context_window: int | None = None
    max_tokens: int | None = None
    modalities: VercelModalities | None = None
    pricing: VercelPricing = VercelPricing()
    supported_parameters: tuple[str, ...] | None = None
    deprecated_at: int | None = None


OPENROUTER_ADAPTER: Final = TypeAdapter(list[OpenRouterModel])
VERCEL_ADAPTER: Final = TypeAdapter(list[VercelModel])


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    key: str
    provider: Provider
    mode: str
    source: str
    fields: Mapping[str, object]
    indicative_prices: bool = False


@dataclass(frozen=True, slots=True)
class Skipped:
    reason: str
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class Unmappable:
    problem: str


@dataclass(frozen=True, slots=True)
class Catalog:
    provider: Provider
    entries: tuple[CatalogEntry, ...]
    skipped: Mapping[str, int]
    warnings: tuple[str, ...]


def per_token(price: float) -> float:
    return float(f"{price:.6g}")


def _token_price(raw: str | None) -> float | None:
    if raw is None:
        return None
    value: Final = float(raw)
    return per_token(value) if math.isfinite(value) and value >= 0 else None


def _extra_price(raw: str | None) -> float | None:
    price: Final = _token_price(raw)
    return price if price else None


def _flags(
    parameters: Sequence[str] | None, inputs: Sequence[str] | None, outputs: Sequence[str] | None
) -> Mapping[str, bool]:
    params: Final = frozenset(parameters or ())
    input_modalities: Final = frozenset(inputs or ())
    output_modalities: Final = frozenset(outputs or ())
    claims: Final = {
        "supports_function_calling": "tools" in params,
        "supports_tool_choice": "tool_choice" in params,
        "supports_reasoning": "reasoning" in params,
        "supports_response_schema": "structured_outputs" in params,
        "supports_vision": "image" in input_modalities,
        "supports_pdf_input": bool({"file", "pdf"} & input_modalities),
        "supports_audio_input": "audio" in input_modalities,
        "supports_video_input": "video" in input_modalities,
        "supports_audio_output": "audio" in output_modalities,
    }
    return MappingProxyType({name: True for name, claimed in claims.items() if claimed})


def _limits(max_input: int | None, max_output: int | None) -> Mapping[str, int]:
    ceiling: Final = max_output if max_output is not None else max_input
    return MappingProxyType(
        {
            **({"max_input_tokens": max_input} if max_input is not None else {}),
            **({"max_output_tokens": max_output} if max_output is not None else {}),
            **({"max_tokens": ceiling} if ceiling is not None else {}),
        }
    )


def _priced(name: str, price: float | None) -> Prices:
    return MappingProxyType({name: price} if price is not None else {})


def _output_prices(
    outputs: Sequence[str] | None, image_price: float | None, audio_price: float | None
) -> Prices | Skipped:
    modalities: Final = frozenset(outputs or ("text",))
    known: Final = {
        modality: price for modality, price in (("image", image_price), ("audio", audio_price)) if price is not None
    }
    if "text" not in modalities or not (modalities - {"text"}) <= known.keys():
        return Skipped("output priced outside the catalog")
    names: Final = {"image": "output_cost_per_image_token", "audio": "output_cost_per_audio_token"}
    return MappingProxyType({names[modality]: known[modality] for modality in modalities & known.keys()})


def _tier_threshold(boundary: int) -> int | None:
    return next((start // 1000 for start in (boundary, boundary - 1) if start > 0 and start % 1000 == 0), None)


@dataclass(frozen=True, slots=True)
class TierLadder:
    name: str
    base: float
    tiers: tuple[VercelTier, ...]
    thresholds: frozenset[int]

    def price_above(self, thousands: int) -> float | None:
        tokens: Final = thousands * 1000 + 1
        tier: Final = next(
            (tier for tier in self.tiers if (tier.min or 0) <= tokens and (tier.max is None or tokens < tier.max)), None
        )
        return _token_price(tier.cost) if tier is not None else self.base


def _ladder(name: str, base: float | None, tiers: Sequence[VercelTier] | None) -> TierLadder | Unmappable | None:
    if base is None or not tiers:
        return None
    ordered: Final = tuple(sorted(tiers, key=lambda tier: tier.min or 0))
    contiguous: Final = ordered[-1].max is None and all(
        lower.max == upper.min for lower, upper in zip(ordered, ordered[1:], strict=False)
    )
    if not contiguous:
        return Unmappable(f"{name} tiers are not contiguous")
    thresholds: Final = tuple(_tier_threshold(tier.min) for tier in ordered if tier.min)
    if None in thresholds or any(_token_price(tier.cost) is None for tier in ordered):
        return Unmappable(f"{name} tiers have a boundary that is not a whole thousand or an unusable price")
    return TierLadder(name, base, ordered, frozenset(threshold for threshold in thresholds if threshold is not None))


def _vercel_tiers(pricing: VercelPricing, cache_read: float | None, cache_write: float | None) -> Prices | Unmappable:
    parts: Final = (
        _ladder("input_cost_per_token", _token_price(pricing.input), pricing.input_tiers),
        _ladder("output_cost_per_token", _token_price(pricing.output), pricing.output_tiers),
        _ladder("cache_read_input_token_cost", cache_read, pricing.input_cache_read_tiers),
        _ladder("cache_creation_input_token_cost", cache_write, pricing.input_cache_write_tiers),
    )
    problem: Final = next((part for part in parts if isinstance(part, Unmappable)), None)
    if problem is not None:
        return problem
    ladders: Final = tuple(part for part in parts if isinstance(part, TierLadder))
    thresholds: Final = sorted(frozenset().union(*(ladder.thresholds for ladder in ladders)))
    return MappingProxyType(
        {
            f"{ladder.name}_above_{thousands}k_tokens": price
            for ladder in ladders
            for thousands in thresholds
            if (price := ladder.price_above(thousands)) is not None
        }
    )


def _openrouter_entry(model: OpenRouterModel) -> CatalogEntry | Skipped:
    pricing: Final = model.pricing
    prompt: Final = _token_price(pricing.prompt)
    completion: Final = _token_price(pricing.completion)
    if prompt is None or completion is None:
        return Skipped("unpriced or router")
    inputs: Final = model.architecture.input_modalities if model.architecture else None
    outputs: Final = model.architecture.output_modalities if model.architecture else None
    output_prices: Final = _output_prices(
        outputs, _extra_price(pricing.image_output), _extra_price(pricing.audio_output)
    )
    if isinstance(output_prices, Skipped):
        return output_prices
    fields: Final = {
        "input_cost_per_token": prompt,
        "output_cost_per_token": completion,
        **_limits(model.context_length, model.top_provider.max_completion_tokens),
        **_priced("cache_read_input_token_cost", _extra_price(pricing.input_cache_read)),
        **_priced("cache_creation_input_token_cost", _extra_price(pricing.input_cache_write)),
        **_priced("output_cost_per_reasoning_token", _extra_price(pricing.internal_reasoning)),
        **_priced("input_cost_per_audio_token", _extra_price(pricing.audio)),
        **output_prices,
        **_flags(model.supported_parameters, inputs, outputs),
    }
    return CatalogEntry(
        key=f"openrouter/{model.id}",
        provider="openrouter",
        mode="chat",
        source=f"https://openrouter.ai/{model.id}",
        fields=MappingProxyType(fields),
    )


def _vercel_entry(model: VercelModel, now_ms: int) -> CatalogEntry | Skipped:
    if model.deprecated_at is not None and model.deprecated_at <= now_ms:
        return Skipped("deprecated")
    mode: Final = VERCEL_TYPE_TO_MODE.get(model.type)
    if mode is None:
        return Skipped("not token priced")
    pricing: Final = model.pricing
    prompt: Final = _token_price(pricing.input)
    completion: Final = _token_price(pricing.output if mode != "embedding" else pricing.output or "0")
    if prompt is None or completion is None:
        return Skipped("no usable price")
    key: Final = f"vercel_ai_gateway/{model.id}"
    inputs: Final = model.modalities.input if model.modalities else None
    outputs: Final = model.modalities.output if model.modalities else None
    output_prices: Final = _output_prices(outputs, None, _extra_price(pricing.audio_output_token_cost))
    if isinstance(output_prices, Skipped):
        return output_prices
    cache_read: Final = _extra_price(pricing.input_cache_read)
    cache_write: Final = _extra_price(pricing.input_cache_write)
    tiers: Final = _vercel_tiers(pricing, cache_read, cache_write)
    if isinstance(tiers, Unmappable):
        return Skipped("tiers outside the registry's thresholds", warning=f"{key}: {tiers.problem}; row skipped")
    fields: Final = {
        "input_cost_per_token": prompt,
        "output_cost_per_token": completion,
        **_limits(model.context_window, model.max_tokens),
        **_priced("cache_read_input_token_cost", cache_read),
        **_priced("cache_creation_input_token_cost", cache_write),
        **_priced("input_cost_per_audio_token", _extra_price(pricing.audio_input_token_cost)),
        **output_prices,
        **tiers,
        **(_flags(model.supported_parameters, inputs, outputs) if mode == "chat" else {}),
    }
    return CatalogEntry(
        key=key,
        provider="vercel_ai_gateway",
        mode=mode,
        source=f"https://vercel.com/ai-gateway/models/{model.id.rsplit('/', 1)[-1]}",
        fields=MappingProxyType(fields),
        indicative_prices=pricing.varies_by_provider,
    )


def _rows(raw: bytes, url: str) -> object:
    parsed: Final = json.loads(raw)
    rows: Final = parsed.get("data") if isinstance(parsed, dict) else parsed
    if not isinstance(rows, list) or not rows:
        raise SyncError(f"GET {url} returned no model rows")
    return rows


def _catalog(provider: Provider, rows: Sequence[CatalogEntry | Skipped]) -> Catalog:
    return Catalog(
        provider=provider,
        entries=tuple(row for row in rows if isinstance(row, CatalogEntry)),
        skipped=MappingProxyType(Counter(row.reason for row in rows if isinstance(row, Skipped))),
        warnings=tuple(row.warning for row in rows if isinstance(row, Skipped) and row.warning is not None),
    )


def load_openrouter(raw: bytes) -> Catalog:
    try:
        models: Final = OPENROUTER_ADAPTER.validate_python(_rows(raw, OPENROUTER_MODELS_URL))
    except ValidationError as error:
        raise SyncError(f"the OpenRouter catalog no longer matches the expected shape: {error}") from error
    return _catalog("openrouter", tuple(map(_openrouter_entry, models)))


def load_vercel(raw: bytes, now_ms: int) -> Catalog:
    try:
        models: Final = VERCEL_ADAPTER.validate_python(_rows(raw, VERCEL_MODELS_URL))
    except ValidationError as error:
        raise SyncError(f"the Vercel AI Gateway catalog no longer matches the expected shape: {error}") from error
    return _catalog("vercel_ai_gateway", tuple(_vercel_entry(model, now_ms) for model in models))


@dataclass(frozen=True, slots=True)
class ProviderOutcome:
    provider: Provider
    added: tuple[str, ...]
    updated: tuple[str, ...]
    warnings: tuple[str, ...]
    skipped: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class SyncOutcome:
    cost_map: CostMap
    providers: tuple[ProviderOutcome, ...]

    @property
    def has_changes(self) -> bool:
        return any(outcome.added or outcome.updated for outcome in self.providers)


def _root_candidates(bare: str) -> tuple[str, ...]:
    segments: Final = bare.split("-")
    stems: Final = tuple(
        "-".join(segments[:count]) for count in range(len(segments), 0, -1) if count >= 2 or count == len(segments)
    )
    return tuple(dict.fromkeys(name for stem in stems for name in (stem, stem.replace(".", "-"))))


def _inherited(cost_map: CostMap, entry: CatalogEntry) -> Mapping[str, object]:
    bare: Final = entry.key.rsplit("/", 1)[-1].split(":", 1)[0]
    same_name: Final = frozenset((bare, bare.replace(".", "-")))
    root: Final = next(
        (
            candidate
            for name, candidate in ((name, cost_map.get(name)) for name in _root_candidates(bare))
            if isinstance(candidate, dict) and (name in same_name or candidate.get("mode") == entry.mode)
        ),
        None,
    )
    if root is None:
        return NO_TRAITS
    return MappingProxyType({name: value for name, value in root.items() if name in INHERITED_TRAITS})


def _new_entry(entry: CatalogEntry, inherited: Mapping[str, object]) -> RegistryEntry:
    return dict(
        sorted(
            {
                **inherited,
                **entry.fields,
                "litellm_provider": entry.provider,
                "mode": entry.mode,
                "source": entry.source,
                **({"price_varies_by_provider": True} if entry.indicative_prices else {}),
            }.items()
        )
    )


@dataclass(frozen=True, slots=True)
class FieldChange:
    name: str
    old: object
    new: object
    hold: str | None

    @property
    def line(self) -> str:
        held: Final = f" held back: {self.hold}" if self.hold else ""
        return f"{self.name}: {self.old!r} -> {self.new!r}{held}"


def _swing(old: float, new: float) -> str | None:
    if (old == 0) != (new == 0):
        return "a price crossing zero"
    if old and new and max(new / old, old / new) > SWING_LIMIT:
        return f"a price moving more than {SWING_LIMIT}x"
    return None


def _copied_cap_hold(current: object, ceiling: object) -> str | None:
    if isinstance(current, int) and isinstance(ceiling, int) and ceiling * SWING_LIMIT < current:
        return f"a copied output cap shrinking more than {SWING_LIMIT}x"
    return None


def _hold(name: str, old: object, new: object, curated_prices_win: bool) -> str | None:
    if "cost" in name and curated_prices_win:
        return "the catalog price varies by provider"
    if old is None:
        return None
    if name.startswith("supports_") and old is False:
        return "a capability flag curated as false"
    if name.startswith("max_") and isinstance(old, int) and isinstance(new, int) and new < old:
        return "a shrinking limit"
    if "cost" in name and isinstance(old, int | float) and isinstance(new, int | float):
        return _swing(old, new)
    return None


def _changes(existing: RegistryEntry, entry: CatalogEntry) -> tuple[FieldChange, ...]:
    curated_prices_win: Final = (
        entry.indicative_prices and "input_cost_per_token" in existing and not existing.get("price_varies_by_provider")
    )
    scalars: Final = tuple(
        FieldChange(name, existing.get(name), value, _hold(name, existing.get(name), value, curated_prices_win))
        for name, value in entry.fields.items()
        if name not in LIMIT_PAIR and existing.get(name) != value
    )
    ceiling: Final = entry.fields.get("max_output_tokens")
    if ceiling is None:
        return scalars
    current: Final = existing.get("max_output_tokens", existing.get("max_tokens"))
    hold: Final = (
        _copied_cap_hold(current, ceiling)
        if current == existing.get("max_input_tokens")
        else _hold("max_output_tokens", current, ceiling, curated_prices_win)
    )
    return (
        *scalars,
        *(FieldChange(name, existing.get(name), ceiling, hold) for name in LIMIT_PAIR if existing.get(name) != ceiling),
    )


def _updated_entry(
    existing: RegistryEntry, entry: CatalogEntry
) -> tuple[RegistryEntry, tuple[str, ...], tuple[str, ...]]:
    changes: Final = _changes(existing, entry)
    applied: Final = {change.name: change.new for change in changes if change.hold is None}
    return (
        {**existing, **dict(sorted(applied.items()))},
        tuple(change.line for change in changes if change.hold is None),
        tuple(change.line for change in changes if change.hold is not None),
    )


def _with_new_keys_in_block(ordered: CostMap, result: CostMap, new_keys: Sequence[str], prefix: str) -> CostMap:
    provider_keys: Final = tuple(key for key in ordered if key.startswith(prefix))
    if not new_keys:
        return {key: result[key] for key in ordered}
    if not provider_keys:
        return {**{key: result[key] for key in ordered}, **{key: result[key] for key in sorted(new_keys)}}
    block_end: Final = provider_keys[-1]
    return {
        key: value
        for existing in ordered
        for key, value in (
            (existing, result[existing]),
            *((new, result[new]) for new in sorted(new_keys) if existing == block_end),
        )
    }


@dataclass(frozen=True, slots=True)
class Added:
    key: str
    entry: RegistryEntry


@dataclass(frozen=True, slots=True)
class Updated:
    key: str
    entry: RegistryEntry
    line: str


@dataclass(frozen=True, slots=True)
class Warned:
    line: str


EntrySync = Added | Updated | Warned


def _sync_entry(cost_map: CostMap, entry: CatalogEntry) -> tuple[EntrySync, ...]:
    existing: Final = cost_map.get(entry.key)
    if not isinstance(existing, dict):
        return (Added(key=entry.key, entry=_new_entry(entry, _inherited(cost_map, entry))),)
    if existing.get("mode") != entry.mode:
        return (
            Warned(
                line=f"`{entry.key}` has curated mode {existing.get('mode')!r} but the catalog maps to "
                f"{entry.mode!r}; left unchanged"
            ),
        )
    new_entry, applied, held = _updated_entry(existing, entry)
    return (
        *((Updated(key=entry.key, entry=new_entry, line=f"{entry.key}: " + "; ".join(applied)),) if applied else ()),
        *((Warned(line=f"{entry.key}: " + "; ".join(held)),) if held else ()),
    )


SyncState = tuple[CostMap, tuple[ProviderOutcome, ...]]


def _sync_provider(state: SyncState, catalog: Catalog) -> SyncState:
    cost_map, outcomes = state
    syncs: Final = tuple(
        sync for entry in sorted(catalog.entries, key=lambda item: item.key) for sync in _sync_entry(cost_map, entry)
    )
    outcome: Final = ProviderOutcome(
        provider=catalog.provider,
        added=tuple(sync.key for sync in syncs if isinstance(sync, Added)),
        updated=tuple(sync.line for sync in syncs if isinstance(sync, Updated)),
        warnings=(*catalog.warnings, *(sync.line for sync in syncs if isinstance(sync, Warned))),
        skipped=catalog.skipped,
    )
    merged: Final = {**cost_map, **{sync.key: sync.entry for sync in syncs if isinstance(sync, Added | Updated)}}
    return merged, (*outcomes, outcome)


def compute_sync(cost_map: CostMap, catalogs: Sequence[Catalog]) -> SyncOutcome:
    synced, outcomes = reduce(_sync_provider, catalogs, (dict(cost_map), ()))
    return SyncOutcome(cost_map=_ordered_result(cost_map, synced, outcomes), providers=outcomes)


def _ordered_result(cost_map: CostMap, result: CostMap, outcomes: Sequence[ProviderOutcome]) -> CostMap:
    return reduce(
        lambda ordered, outcome: _with_new_keys_in_block(ordered, result, outcome.added, f"{outcome.provider}/"),
        outcomes,
        {key: result[key] for key in cost_map},
    )


def _section_block(title: str, lines: Sequence[str], backtick: bool, limit: int | None, rest: str) -> str:
    shown: Final = lines[:limit]
    bullets: Final = "\n".join(f"- `{line}`" if backtick else f"- {line}" for line in shown) or "- none"
    overflow: Final = len(lines) - len(shown)
    trailer: Final = f"\n- and {overflow} more, see the {rest}" if overflow else ""
    return f"### {title} ({len(lines)})\n{bullets}{trailer}\n"


def _provider_body(outcome: ProviderOutcome, limit: int | None, warnings_limit: int | None) -> str:
    skipped: Final = ", ".join(f"{reason} ({count})" for reason, count in sorted(outcome.skipped.items())) or "none"
    return (
        f"## {outcome.provider}\n"
        "\n"
        f"{_section_block('Added', outcome.added, True, limit, 'diff')}"
        "\n"
        f"{_section_block('Updated', outcome.updated, True, limit, 'diff')}"
        "\n"
        f"{_section_block('Warnings needing a human call', outcome.warnings, False, warnings_limit, 'workflow log')}"
        "\n"
        f"Catalog rows skipped: {skipped}\n"
    )


def _pr_body(outcome: SyncOutcome, section_limit: int | None, warnings_limit: int | None) -> str:
    return (
        "Automated sync of the openrouter and vercel_ai_gateway entries in model_prices_and_context_window.json "
        f"against `GET {OPENROUTER_MODELS_URL}` and `GET {VERCEL_MODELS_URL}` by scripts/sync_cost_map.py. "
        "The cost-map-guard check enforces that this PR only adds or reprices models. Changes the script held "
        "back (shrinking limits, prices crossing zero or moving more than 10x, per-provider prices on curated "
        "rows, capability flags curated as false) are listed under the warnings and need a human commit.\n"
        "\n" + "\n".join(_provider_body(provider, section_limit, warnings_limit) for provider in outcome.providers)
    )


def render_pr_body(outcome: SyncOutcome, section_limit: int | None = PR_BODY_SECTION_LIMIT) -> str:
    every_warning: Final = _pr_body(outcome, section_limit, None)
    if section_limit is None or len(every_warning) <= GITHUB_BODY_LIMIT:
        return every_warning
    return _pr_body(outcome, section_limit, section_limit)


def render_summary(outcome: SyncOutcome) -> str:
    return " ".join(
        f"{provider.provider}: added={len(provider.added)} updated={len(provider.updated)} "
        f"warnings={len(provider.warnings)}"
        for provider in outcome.providers
    )


def _fetch(url: str) -> bytes:
    response: Final = httpx.get(url, timeout=30, follow_redirects=True)
    if response.status_code != 200:
        raise SyncError(f"GET {url} returned {response.status_code}")
    return response.content


def _serialize(cost_map: CostMap) -> str:
    return json.dumps(cost_map, indent=4, ensure_ascii=False) + "\n"


def main(argv: Sequence[str]) -> int:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="apply the sync to the cost map files (default: dry run)")
    parser.add_argument("--openrouter-json", type=Path, help="recorded OpenRouter catalog instead of the live API")
    parser.add_argument("--vercel-json", type=Path, help="recorded Vercel AI Gateway catalog instead of the live API")
    parser.add_argument("--pr-body-file", type=Path, help="write the generated PR body to this path")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args: Final = parser.parse_args(argv)

    openrouter_raw: Final = (
        args.openrouter_json.read_bytes() if args.openrouter_json is not None else _fetch(OPENROUTER_MODELS_URL)
    )
    vercel_raw: Final = args.vercel_json.read_bytes() if args.vercel_json is not None else _fetch(VERCEL_MODELS_URL)
    catalogs: Final = (load_openrouter(openrouter_raw), load_vercel(vercel_raw, now_ms=int(time.time() * 1000)))

    cost_map_path: Final = args.repo_root / COST_MAP_RELPATHS[0]
    cost_map: Final = json.loads(cost_map_path.read_text())
    outcome: Final = compute_sync(cost_map, catalogs)

    if args.pr_body_file is not None:
        args.pr_body_file.write_text(render_pr_body(outcome))
    if args.write and outcome.has_changes:
        for relpath in COST_MAP_RELPATHS:
            (args.repo_root / relpath).write_text(_serialize(outcome.cost_map))
    print(render_summary(outcome))
    print()
    print(render_pr_body(outcome, section_limit=None))
    if not args.write:
        print("dry run: no files were touched")
    elif not outcome.has_changes:
        print("registry already in sync: no files were touched")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SyncError as error:
        print(f"SYNC FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
