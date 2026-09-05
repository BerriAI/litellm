"""Sync the openrouter and vercel_ai_gateway entries of model_prices_and_context_window.json with the live catalogs.

Pulls ``GET https://openrouter.ai/api/v1/models`` and ``GET https://ai-gateway.vercel.sh/v1/models``, maps the
catalog fields onto registry fields, and diffs the result against the registry. Dry run (the default) prints the
diff summary and the generated PR body; ``--write`` applies the changes to the root cost map and its ``litellm/``
backup copy.

Policy:
- Both catalogs price per token as decimal strings; values are normalized to six significant digits.
- An existing entry only gains or changes the fields the catalog expresses. Nothing is ever removed, a
  capability flag the catalog does not claim stays as curated, and a curated output ceiling is kept.
- Router models and rows without a usable prompt and completion price are skipped.
- A registry entry absent from its catalog is left untouched; retiring a model stays a human call.
"""

import argparse
import json
import math
import sys
import time
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
ADD_ONLY_FIELDS: Final = frozenset({"max_output_tokens", "max_tokens"})

Provider = Literal["openrouter", "vercel_ai_gateway"]
RegistryEntry = dict[str, object]
CostMap = dict[str, object]


class SyncError(RuntimeError):
    pass


class OpenRouterPricing(BaseModel):
    prompt: str
    completion: str
    input_cache_read: str | None = None
    input_cache_write: str | None = None
    internal_reasoning: str | None = None


class OpenRouterArchitecture(BaseModel):
    input_modalities: tuple[str, ...] | None = None


class OpenRouterTopProvider(BaseModel):
    max_completion_tokens: int | None = None


class OpenRouterModel(BaseModel):
    id: str
    context_length: int | None = None
    architecture: OpenRouterArchitecture | None = None
    top_provider: OpenRouterTopProvider = OpenRouterTopProvider()
    pricing: OpenRouterPricing
    supported_parameters: tuple[str, ...] | None = None


class VercelPricing(BaseModel):
    input: str | None = None
    output: str | None = None
    input_cache_read: str | None = None
    input_cache_write: str | None = None


class VercelModalities(BaseModel):
    input: tuple[str, ...] | None = None


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


@dataclass(frozen=True, slots=True)
class Catalog:
    provider: Provider
    entries: tuple[CatalogEntry, ...]
    skipped: Mapping[str, int]


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


def _flags(parameters: Sequence[str] | None, modalities: Sequence[str] | None) -> Mapping[str, bool]:
    params: Final = frozenset(parameters or ())
    mods: Final = frozenset(modalities or ())
    claims: Final = {
        "supports_function_calling": "tools" in params,
        "supports_tool_choice": "tool_choice" in params,
        "supports_reasoning": "reasoning" in params,
        "supports_response_schema": "structured_outputs" in params,
        "supports_vision": "image" in mods,
        "supports_pdf_input": bool({"file", "pdf"} & mods),
        "supports_audio_input": "audio" in mods,
        "supports_video_input": "video" in mods,
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


def _priced(name: str, price: float | None) -> Mapping[str, float]:
    return MappingProxyType({name: price} if price is not None else {})


def _openrouter_entry(model: OpenRouterModel) -> CatalogEntry | None:
    prompt: Final = _token_price(model.pricing.prompt)
    completion: Final = _token_price(model.pricing.completion)
    if prompt is None or completion is None:
        return None
    fields: Final = {
        "input_cost_per_token": prompt,
        "output_cost_per_token": completion,
        **_limits(model.context_length, model.top_provider.max_completion_tokens),
        **_priced("cache_read_input_token_cost", _extra_price(model.pricing.input_cache_read)),
        **_priced("cache_creation_input_token_cost", _extra_price(model.pricing.input_cache_write)),
        **_priced("output_cost_per_reasoning_token", _extra_price(model.pricing.internal_reasoning)),
        **_flags(model.supported_parameters, model.architecture.input_modalities if model.architecture else None),
    }
    return CatalogEntry(
        key=f"openrouter/{model.id}",
        provider="openrouter",
        mode="chat",
        source=f"https://openrouter.ai/{model.id}",
        fields=MappingProxyType(fields),
    )


def _vercel_entry(model: VercelModel) -> CatalogEntry | None:
    mode: Final = VERCEL_TYPE_TO_MODE.get(model.type)
    prompt: Final = _token_price(model.pricing.input)
    completion: Final = _token_price(model.pricing.output if mode != "embedding" else model.pricing.output or "0")
    if mode is None or prompt is None or completion is None:
        return None
    fields: Final = {
        "input_cost_per_token": prompt,
        "output_cost_per_token": completion,
        **_limits(model.context_window, model.max_tokens),
        **_priced("cache_read_input_token_cost", _extra_price(model.pricing.input_cache_read)),
        **_priced("cache_creation_input_token_cost", _extra_price(model.pricing.input_cache_write)),
        **(
            _flags(model.supported_parameters, model.modalities.input if model.modalities else None)
            if mode == "chat"
            else {}
        ),
    }
    return CatalogEntry(
        key=f"vercel_ai_gateway/{model.id}",
        provider="vercel_ai_gateway",
        mode=mode,
        source=f"https://vercel.com/ai-gateway/models/{model.id.rsplit('/', 1)[-1]}",
        fields=MappingProxyType(fields),
    )


def _rows(raw: bytes, url: str) -> object:
    parsed: Final = json.loads(raw)
    rows: Final = parsed.get("data") if isinstance(parsed, dict) else parsed
    if not isinstance(rows, list) or not rows:
        raise SyncError(f"GET {url} returned no model rows")
    return rows


def load_openrouter(raw: bytes) -> Catalog:
    try:
        models: Final = OPENROUTER_ADAPTER.validate_python(_rows(raw, OPENROUTER_MODELS_URL))
    except ValidationError as error:
        raise SyncError(f"the OpenRouter catalog no longer matches the expected shape: {error}") from error
    entries: Final = tuple(entry for entry in map(_openrouter_entry, models) if entry is not None)
    return Catalog(
        provider="openrouter",
        entries=entries,
        skipped=MappingProxyType({"unpriced or router": len(models) - len(entries)}),
    )


def load_vercel(raw: bytes, now_ms: int) -> Catalog:
    try:
        models: Final = VERCEL_ADAPTER.validate_python(_rows(raw, VERCEL_MODELS_URL))
    except ValidationError as error:
        raise SyncError(f"the Vercel AI Gateway catalog no longer matches the expected shape: {error}") from error
    live: Final = tuple(model for model in models if model.deprecated_at is None or model.deprecated_at > now_ms)
    token_priced: Final = tuple(model for model in live if model.type in VERCEL_TYPE_TO_MODE)
    entries: Final = tuple(entry for entry in map(_vercel_entry, token_priced) if entry is not None)
    return Catalog(
        provider="vercel_ai_gateway",
        entries=entries,
        skipped=MappingProxyType(
            {
                "deprecated": len(models) - len(live),
                "not token priced": len(live) - len(token_priced),
                "no usable price": len(token_priced) - len(entries),
            }
        ),
    )


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


def _new_entry(entry: CatalogEntry) -> RegistryEntry:
    return dict(
        sorted(
            {
                **entry.fields,
                "litellm_provider": entry.provider,
                "mode": entry.mode,
                "source": entry.source,
            }.items()
        )
    )


def _updated_entry(existing: RegistryEntry, entry: CatalogEntry) -> tuple[RegistryEntry, tuple[str, ...]]:
    keep_limits: Final = not ADD_ONLY_FIELDS.isdisjoint(existing)
    desired: Final = {
        name: value for name, value in entry.fields.items() if not (keep_limits and name in ADD_ONLY_FIELDS)
    }
    changes: Final = tuple(
        f"{name}: {existing.get(name)!r} -> {value!r}" for name, value in desired.items() if existing.get(name) != value
    )
    return dict(sorted({**existing, **desired}.items())), changes


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


@dataclass(frozen=True, slots=True)
class Unchanged:
    pass


EntrySync = Added | Updated | Warned | Unchanged


def _sync_entry(existing: object, entry: CatalogEntry) -> EntrySync:
    if not isinstance(existing, dict):
        return Added(key=entry.key, entry=_new_entry(entry))
    if existing.get("mode") != entry.mode:
        return Warned(
            line=f"`{entry.key}` has curated mode {existing.get('mode')!r} but the catalog maps to "
            f"{entry.mode!r}; left unchanged"
        )
    new_entry, changes = _updated_entry(existing, entry)
    if not changes:
        return Unchanged()
    return Updated(key=entry.key, entry=new_entry, line=f"{entry.key}: " + "; ".join(changes))


SyncState = tuple[CostMap, tuple[ProviderOutcome, ...]]


def _sync_provider(state: SyncState, catalog: Catalog) -> SyncState:
    cost_map, outcomes = state
    syncs: Final = tuple(
        _sync_entry(cost_map.get(entry.key), entry) for entry in sorted(catalog.entries, key=lambda item: item.key)
    )
    outcome: Final = ProviderOutcome(
        provider=catalog.provider,
        added=tuple(sync.key for sync in syncs if isinstance(sync, Added)),
        updated=tuple(sync.line for sync in syncs if isinstance(sync, Updated)),
        warnings=tuple(sync.line for sync in syncs if isinstance(sync, Warned)),
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


def _section_block(title: str, lines: Sequence[str], backtick: bool) -> str:
    bullets: Final = "\n".join(f"- `{line}`" if backtick else f"- {line}" for line in lines) or "- none"
    return f"### {title} ({len(lines)})\n{bullets}\n"


def _provider_body(outcome: ProviderOutcome) -> str:
    skipped: Final = ", ".join(f"{reason} ({count})" for reason, count in sorted(outcome.skipped.items())) or "none"
    return (
        f"## {outcome.provider}\n"
        "\n"
        f"{_section_block('Added', outcome.added, backtick=True)}"
        "\n"
        f"{_section_block('Updated', outcome.updated, backtick=True)}"
        "\n"
        f"{_section_block('Warnings needing a human call', outcome.warnings, backtick=False)}"
        "\n"
        f"Catalog rows skipped: {skipped}\n"
    )


def render_pr_body(outcome: SyncOutcome) -> str:
    return (
        "Automated sync of the openrouter and vercel_ai_gateway entries in model_prices_and_context_window.json "
        f"against `GET {OPENROUTER_MODELS_URL}` and `GET {VERCEL_MODELS_URL}` by scripts/sync_cost_map.py. "
        "The cost-map-guard check enforces that this PR only adds or reprices models.\n"
        "\n" + "\n".join(_provider_body(provider) for provider in outcome.providers)
    )


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
    body: Final = render_pr_body(outcome)

    if args.pr_body_file is not None:
        args.pr_body_file.write_text(body)
    if args.write and outcome.has_changes:
        for relpath in COST_MAP_RELPATHS:
            (args.repo_root / relpath).write_text(_serialize(outcome.cost_map))
    print(render_summary(outcome))
    print()
    print(body)
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
