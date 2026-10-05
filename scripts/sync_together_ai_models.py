"""Sync the together_ai entries of model_prices_and_context_window.json with Together's live serverless catalog.

Pulls ``GET https://api.together.ai/v1/models?serverless`` plus the deprecations doc, maps API fields onto
registry fields, merges the reviewed capability rules below for everything the API cannot express, and diffs
the result against the registry. Dry run (the default) prints the diff summary and the generated PR body;
``--write`` applies the changes to the root cost map and its ``litellm/`` backup copy.

Policy highlights:
- Prices arrive per 1M tokens with float artifacts and are normalized to clean per-token values.
- A registry entry absent from the serverless catalog is marked with ``deprecation_date`` from the docs
  deprecation table, never deleted; absences with no docs date are surfaced for a human call.
- Availability comes from the API: a model the docs list as removed but the API still serves stays live,
  with the conflict surfaced as a warning.
- Manually curated values the API cannot express (``metadata.successor``, ``max_output_tokens`` on existing
  entries, capability flags no rule covers) are never overwritten; conflicts are surfaced instead.
"""

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

MODELS_URL: Final = "https://api.together.ai/v1/models?serverless"
DEPRECATIONS_URL: Final = "https://docs.together.ai/docs/deprecations.md"
PROVIDER: Final = "together_ai"
PREFIX: Final = "together_ai/"
SOURCE_URL: Final = "https://docs.together.ai/docs/serverless-models"
COST_MAP_RELPATHS: Final = (
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
)

TYPE_TO_MODE: Final = MappingProxyType({"chat": "chat", "embedding": "embedding", "moderation": "chat"})


class SyncError(RuntimeError):
    pass


class CatalogPricing(BaseModel):
    input: float
    output: float
    cached_input: float | None = None


class CatalogModel(BaseModel):
    id: str
    type: str
    context_length: int | None = None
    pricing: CatalogPricing


CATALOG_ADAPTER: Final = TypeAdapter(list[CatalogModel])

RegistryEntry = dict[str, object]
CostMap = dict[str, object]


@dataclass(frozen=True, slots=True)
class CapabilityRule:
    model_id: str
    fields: Mapping[str, bool | int]
    provenance: str


def _rule(model_id: str, provenance: str, **fields: bool | int) -> CapabilityRule:
    return CapabilityRule(model_id=model_id, fields=MappingProxyType(dict(fields)), provenance=provenance)


_TOOLS: Final = MappingProxyType(
    {
        "supports_function_calling": True,
        "supports_parallel_function_calling": True,
        "supports_response_schema": True,
        "supports_tool_choice": True,
    }
)

CAPABILITY_RULES: Final = (
    _rule(
        "MiniMaxAI/MiniMax-M3",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/minimax-m3",
        **_TOOLS,
        supports_reasoning=True,
        supports_vision=True,
    ),
    _rule("Prism-ML/Ternary-Bonsai-27B", "reviewed for the LIT-5968 backfill; no tool or vision support documented"),
    _rule(
        "Qwen/Qwen3.5-9B",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/qwen3-5-9b",
        **_TOOLS,
        supports_reasoning=True,
        supports_vision=True,
    ),
    _rule(
        "Qwen/Qwen3.6-Plus",
        "reviewed for the LIT-5968 backfill; hybrid reasoning model without a documented tools contract",
        supports_reasoning=True,
    ),
    _rule("Qwen/Qwen3.7-Max", "reviewed for the LIT-5968 backfill; no tool or vision support documented"),
    _rule("Qwen/Qwen3.7-Plus", "reviewed for the LIT-5968 backfill; no tool or vision support documented"),
    _rule("Qwen/Qwen3.8-2.4T-A95B", "reviewed for the LIT-5968 backfill; no tool or vision support documented"),
    _rule("arize-ai/qwen-2-1.5b-instruct", "reviewed for the LIT-5968 backfill; no tool or vision support documented"),
    _rule(
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/deepseek-v4-flash",
        **_TOOLS,
    ),
    _rule(
        "deepseek-ai/DeepSeek-V4-Pro",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/deepseek-v4-pro",
        **_TOOLS,
        supports_reasoning=True,
    ),
    _rule(
        "deepseek-ai/DeepSeek-V4-Pro-0813",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/deepseek-v4-pro",
        **_TOOLS,
    ),
    _rule("google/gemma-3n-E4B-it", "reviewed for the LIT-5968 backfill; no tool or vision support documented"),
    _rule(
        "google/gemma-4-31B-it",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/gemma-4-31b-it",
        **_TOOLS,
        supports_vision=True,
    ),
    _rule(
        "intfloat/multilingual-e5-large-instruct",
        "embedding dims per https://huggingface.co/intfloat/multilingual-e5-large-instruct",
        output_vector_size=1024,
    ),
    _rule(
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "reviewed for the LIT-5968 backfill against https://docs.together.ai/docs/function-calling",
        **_TOOLS,
    ),
    _rule(
        "meta-llama/Llama-Guard-4-12B",
        "moderation classifier with a chat-shaped API; no tools per the LIT-5968 backfill review",
    ),
    _rule("meta-models/Muse-Glimmer-30B", "reviewed for the LIT-5968 backfill; no tool or vision support documented"),
    _rule(
        "moonshotai/Kimi-K2.7-Code",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/kimi-k2-7-code",
        **_TOOLS,
        supports_vision=True,
    ),
    _rule(
        "moonshotai/Kimi-K3",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/kimi-k3",
        **_TOOLS,
        supports_reasoning=True,
        supports_vision=True,
    ),
    _rule(
        "nvidia/nemotron-3-ultra-550b-a55b",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/nemotron-3-ultra",
        **_TOOLS,
        supports_reasoning=True,
    ),
    _rule(
        "openai/gpt-oss-120b",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/gpt-oss-120b",
        **_TOOLS,
        supports_reasoning=True,
    ),
    _rule(
        "openai/gpt-oss-20b",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/gpt-oss-20b",
        **_TOOLS,
    ),
    _rule("pearl-ai/gemma-4-31b-it", "reviewed for the LIT-5968 backfill; no tool or vision support documented"),
    _rule(
        "thinkingmachines/Inkling",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/inkling",
        **_TOOLS,
    ),
    _rule("thinkingmachines/Inkling-Small", "reviewed for the LIT-5968 backfill; no tool or vision support documented"),
    _rule(
        "zai-org/GLM-5.2",
        "reviewed for the LIT-5968 backfill against https://www.together.ai/models/glm-5-2;"
        " 128K output ceiling per https://docs.z.ai/guides/llm/glm-5.2",
        **_TOOLS,
        supports_reasoning=True,
        max_output_tokens=128000,
        max_tokens=128000,
    ),
    _rule(
        "zai-org/GLM-5.3-Flash",
        "reviewed for LIT-6489 against https://www.together.ai/models/glm-5-3-flash;"
        " 128K output ceiling per https://docs.z.ai/guides/llm/glm-5.3",
        **_TOOLS,
        supports_reasoning=True,
        supports_vision=True,
        max_output_tokens=128000,
        max_tokens=128000,
    ),
)

RULES_BY_ID: Final = MappingProxyType({rule.model_id: rule for rule in CAPABILITY_RULES})


@dataclass(frozen=True, slots=True)
class DeprecationDoc:
    removal_dates: Mapping[str, str]
    redirects: Mapping[str, str]


_REDIRECT_ROW: Final = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|")
_REMOVAL_ROW: Final = re.compile(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*`([^`]+)`\s*\|")


def _section(markdown: str, heading: str) -> str:
    level: Final = heading.split(" ", 1)[0]
    start: Final = markdown.find(f"\n{heading}\n")
    if start < 0:
        return ""
    body: Final = markdown[start + 1 + len(heading) :]
    next_heading: Final = re.search(rf"^{re.escape(level)} ", body, flags=re.MULTILINE)
    return body[: next_heading.start()] if next_heading else body


def parse_deprecations(markdown: str) -> DeprecationDoc:
    redirect_rows: Final = tuple(
        m.groups()
        for m in (_REDIRECT_ROW.match(line) for line in _section(markdown, "## Active model redirects").splitlines())
        if m
    )
    inference: Final = _section(_section(markdown, "## Deprecation history"), "### Inference")
    removal_rows: Final = tuple(m.groups() for m in (_REMOVAL_ROW.match(line) for line in inference.splitlines()) if m)
    if not redirect_rows or not removal_rows:
        raise SyncError(
            "deprecations doc parsed to zero redirect or removal rows; the table format at "
            f"{DEPRECATIONS_URL} changed and the parser needs updating"
        )
    removal_dates: Final = {model: date for date, model in reversed(removal_rows)}
    return DeprecationDoc(
        removal_dates=MappingProxyType(dict(reversed(removal_dates.items()))),
        redirects=MappingProxyType({original: target for original, target in redirect_rows}),
    )


def per_token(price_per_million: float) -> float:
    return float(f"{price_per_million / 1e6:.6g}")


def _resolve_name(name: str, universe: frozenset[str]) -> str | None:
    if name in universe:
        return name
    suffix_matches: Final = tuple(candidate for candidate in universe if candidate.endswith(f"/{name}"))
    return suffix_matches[0] if len(suffix_matches) == 1 else None


def resolve_successor(model_id: str, doc: DeprecationDoc, live_ids: frozenset[str]) -> str | None:
    canonical: Final = live_ids | frozenset(doc.removal_dates)
    redirects: Final = {
        (_resolve_name(raw_source, canonical) or raw_source): (_resolve_name(raw_target, canonical) or raw_target)
        for raw_source, raw_target in doc.redirects.items()
    }
    seen: Final = set()
    current = model_id  # rebind-ok: walks the redirect chain
    while current in redirects and current not in seen:
        seen.add(current)
        current = redirects[current]  # rebind-ok: walks the redirect chain
    return current if current != model_id and current in live_ids else None


@dataclass(frozen=True, slots=True)
class SyncOutcome:
    cost_map: CostMap
    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    deprecated: tuple[str, ...] = ()
    reappeared: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    skipped_types: Mapping[str, int] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.updated or self.deprecated or self.reappeared)


def _api_fields(model: CatalogModel) -> RegistryEntry:
    cached: Final = model.pricing.cached_input
    return {
        "input_cost_per_token": per_token(model.pricing.input),
        "output_cost_per_token": per_token(model.pricing.output),
        **({"cache_read_input_token_cost": per_token(cached), "supports_prompt_caching": True} if cached else {}),
        **({"max_input_tokens": model.context_length} if model.context_length is not None else {}),
    }


def _new_entry(model: CatalogModel, mode: str) -> RegistryEntry:
    rule: Final = RULES_BY_ID.get(model.id)
    legacy_ceiling: Final = {} if model.context_length is None else {"max_tokens": model.context_length}
    merged: Final = {
        **_api_fields(model),
        **legacy_ceiling,
        "litellm_provider": PROVIDER,
        "mode": mode,
        "source": SOURCE_URL,
        **(dict(rule.fields) if rule else {}),
    }
    return dict(sorted(merged.items()))


def _updated_entry(entry: RegistryEntry, model: CatalogModel) -> tuple[RegistryEntry, tuple[str, ...]]:
    rule: Final = RULES_BY_ID.get(model.id)
    desired: Final = {**_api_fields(model), **(dict(rule.fields) if rule else {})}
    dropped: Final = () if model.pricing.cached_input else ("cache_read_input_token_cost", "supports_prompt_caching")
    changes: Final = tuple(
        f"{name}: {entry.get(name)!r} -> {value!r}" for name, value in desired.items() if entry.get(name) != value
    ) + tuple(
        f"{name}: {entry[name]!r} removed (no longer in the catalog pricing)" for name in dropped if name in entry
    )
    merged: Final = {name: value for name, value in {**entry, **desired}.items() if name not in dropped}
    return dict(sorted(merged.items())), changes


def _with_new_keys_in_block(original: CostMap, result: CostMap, new_keys: Sequence[str]) -> CostMap:
    provider_keys: Final = tuple(key for key in original if key.startswith(PREFIX))
    if not new_keys or not provider_keys:
        return result
    block_end: Final = provider_keys[-1]
    return {
        key: value
        for existing in original
        for key, value in (
            (existing, result[existing]),
            *((new, result[new]) for new in sorted(new_keys) if existing == block_end),
        )
    }


def compute_sync(cost_map: CostMap, catalog: Sequence[CatalogModel], doc: DeprecationDoc) -> SyncOutcome:
    live_ids: Final = frozenset(model.id for model in catalog)
    token_models: Final = {model.id: model for model in catalog if model.type in TYPE_TO_MODE}
    skipped: Final = {
        model.type: sum(1 for m in catalog if m.type == model.type)
        for model in catalog
        if model.type not in TYPE_TO_MODE
    }
    registry_ids: Final = {key.removeprefix(PREFIX): key for key in cost_map if key.startswith(PREFIX)}

    added: Final[list[str]] = []
    updated: Final[list[str]] = []
    deprecated: Final[list[str]] = []
    reappeared: Final[list[str]] = []
    warnings: Final[list[str]] = []
    result: Final[CostMap] = dict(cost_map)

    for model_id, model in sorted(token_models.items()):
        mode: Final = TYPE_TO_MODE[model.type]
        key: Final = f"{PREFIX}{model_id}"
        if model_id in doc.removal_dates:
            warnings.append(
                f"`{key}` is listed as removed on {doc.removal_dates[model_id]} in the docs but the serverless "
                "catalog still serves it; availability kept from the API"
            )
        entry = result.get(key)
        if not isinstance(entry, dict):
            result[key] = _new_entry(model, mode)
            added.append(key)
            if model.type == "chat" and model_id not in RULES_BY_ID:
                warnings.append(
                    f"`{key}` added without a capability rule; review its tools/vision/reasoning support and add one"
                )
            continue
        if entry.get("mode") != mode:
            warnings.append(
                f"`{key}` has curated mode {entry.get('mode')!r} but the catalog maps to {mode!r}; left unchanged"
            )
        new_entry, changes = _updated_entry(entry, model)
        if "deprecation_date" in new_entry:
            new_entry.pop("deprecation_date")
            reappeared.append(key)
        if changes:
            updated.append(f"{key}: " + "; ".join(changes))
        if changes or key in reappeared:
            result[key] = new_entry

    for model_id, key in sorted(registry_ids.items()):
        if model_id in token_models:
            continue
        entry = result.get(key)
        if not isinstance(entry, dict):
            continue
        removal_date: Final = doc.removal_dates.get(model_id)
        successor: Final = resolve_successor(model_id, doc, live_ids)
        metadata = entry.get("metadata")
        curated_successor: Final = metadata.get("successor") if isinstance(metadata, dict) else None
        new_entry = dict(entry)
        if removal_date is not None and entry.get("deprecation_date") != removal_date:
            if "deprecation_date" in entry:
                warnings.append(
                    f"`{key}` has curated deprecation_date {entry.get('deprecation_date')!r} but the docs list "
                    f"{removal_date!r}; left unchanged"
                )
            else:
                new_entry["deprecation_date"] = removal_date
        if removal_date is None and "deprecation_date" not in entry:
            warnings.append(
                f"`{key}` is absent from the serverless catalog with no removal date in the docs; "
                "needs a human deprecation call"
            )
        if successor is not None:
            desired_successor: Final = f"{PREFIX}{successor}"
            if curated_successor is None:
                new_entry["metadata"] = dict(
                    sorted({**(metadata if isinstance(metadata, dict) else {}), "successor": desired_successor}.items())
                )
            elif curated_successor != desired_successor:
                warnings.append(
                    f"`{key}` has curated successor {curated_successor!r} but the docs redirects resolve to "
                    f"{desired_successor!r}; left unchanged"
                )
        if new_entry != entry:
            result[key] = dict(sorted(new_entry.items()))
            deprecated.append(f"{key}: " + ", ".join(sorted(set(new_entry) - set(entry)) or ["updated"]))

    return SyncOutcome(
        cost_map=_with_new_keys_in_block(cost_map, result, tuple(added)),
        added=tuple(added),
        updated=tuple(updated),
        deprecated=tuple(deprecated),
        reappeared=tuple(reappeared),
        warnings=tuple(warnings),
        skipped_types=MappingProxyType(skipped),
    )


def _section_block(title: str, lines: Sequence[str], backtick: bool) -> str:
    bullets: Final = "\n".join(f"- `{line}`" if backtick else f"- {line}" for line in lines) or "- none"
    return f"### {title} ({len(lines)})\n{bullets}\n"


def render_pr_body(outcome: SyncOutcome) -> str:
    skipped: Final = ", ".join(f"{kind} ({count})" for kind, count in sorted(outcome.skipped_types.items())) or "none"
    return (
        "Automated daily sync of the together_ai entries in model_prices_and_context_window.json against "
        f"`GET {MODELS_URL}` and {DEPRECATIONS_URL} by scripts/sync_together_ai_models.py.\n"
        "\n"
        f"{_section_block('Added', outcome.added, backtick=True)}"
        "\n"
        f"{_section_block('Updated', outcome.updated, backtick=True)}"
        "\n"
        f"{_section_block('Marked deprecated', outcome.deprecated, backtick=True)}"
        "\n"
        f"{_section_block('Returned to the catalog', outcome.reappeared, backtick=True)}"
        "\n"
        f"{_section_block('Warnings needing a human call', outcome.warnings, backtick=False)}"
        "\n"
        f"Catalog model types outside the sync's token-pricing scope, skipped: {skipped}\n"
    )


def render_summary(outcome: SyncOutcome) -> str:
    return (
        f"added={len(outcome.added)} updated={len(outcome.updated)} deprecated={len(outcome.deprecated)} "
        f"reappeared={len(outcome.reappeared)} warnings={len(outcome.warnings)}"
    )


def load_catalog(raw: bytes) -> list[CatalogModel]:
    parsed: Final = json.loads(raw)
    entries: Final = parsed.get("data") if isinstance(parsed, dict) else parsed
    try:
        catalog: Final = CATALOG_ADAPTER.validate_python(entries)
    except ValidationError as error:
        raise SyncError(f"the catalog response no longer matches the expected shape: {error}") from error
    if not any(model.type in TYPE_TO_MODE for model in catalog):
        raise SyncError(
            "the catalog response contains no token-priced models; refusing to mark the whole registry deprecated"
        )
    return catalog


def _fetch(url: str, headers: Mapping[str, str]) -> bytes:
    response: Final = httpx.get(url, headers=dict(headers), timeout=30, follow_redirects=True)
    if response.status_code != 200:
        raise SyncError(f"GET {url} returned {response.status_code}")
    return response.content


def _serialize(cost_map: CostMap) -> str:
    return json.dumps(cost_map, indent=4, ensure_ascii=False) + "\n"


def main(argv: Sequence[str]) -> int:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="apply the sync to the cost map files (default: dry run)")
    parser.add_argument("--models-json", type=Path, help="recorded catalog response to use instead of the live API")
    parser.add_argument(
        "--deprecations-md", type=Path, help="recorded deprecations doc to use instead of the live docs"
    )
    parser.add_argument("--pr-body-file", type=Path, help="write the generated PR body to this path")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args: Final = parser.parse_args(argv)

    if args.models_json is not None:
        catalog_raw: Final = args.models_json.read_bytes()
    else:
        api_key: Final = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise SyncError("TOGETHER_API_KEY is not set and --models-json was not given")
        catalog_raw = _fetch(MODELS_URL, {"Authorization": f"Bearer {api_key}"})  # rebind-ok: branch-dependent source
    catalog: Final = load_catalog(catalog_raw)
    markdown: Final = (
        args.deprecations_md.read_text() if args.deprecations_md is not None else _fetch(DEPRECATIONS_URL, {}).decode()
    )
    doc: Final = parse_deprecations(markdown)

    cost_map_path: Final = args.repo_root / COST_MAP_RELPATHS[0]
    cost_map: Final = json.loads(cost_map_path.read_text())
    outcome: Final = compute_sync(cost_map, catalog, doc)
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
