"""Sync the cheaperinference entries of model_prices_and_context_window.json with the gateway's live catalog.

Pulls ``GET https://api.cheaperinference.com/v1/models`` and updates the price and limit fields of the
already-registered cheaperinference/<id> entries. Dry run (the default) prints the diff summary and the
generated PR body; ``--write`` applies the changes to the root cost map and its ``litellm/`` backup copy.

Policy highlights:
- Only price fields (input/output/cache read/cache write, and their above-272k-token variants) and the two
  limit fields (max_input_tokens, max_output_tokens/max_tokens) are synced from the catalog.
- Capability flags (supports_vision, supports_reasoning, supports_tool_choice, and so on), litellm_provider,
  mode, source, and supported_endpoints are curated by hand and are never touched by this script.
- A catalog model with no matching registry entry is surfaced as a warning for a human to add, never
  auto-added: the capability flags for a new model cannot be derived from this endpoint.
- A registry entry whose catalog model disappeared is left untouched and surfaced as a warning: nothing is
  deleted.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

MODELS_URL: Final = "https://api.cheaperinference.com/v1/models"
PROVIDER: Final = "cheaperinference"
PREFIX: Final = "cheaperinference/"
SOURCE_URL: Final = "https://cheaperinference.com/docs"
COST_MAP_RELPATHS: Final = (
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
)


class SyncError(RuntimeError):
    pass


class AboveThreshold(BaseModel):
    input_per_million: str
    output_per_million: str
    cache_read_input_per_million: str
    cache_write_input_per_million: str


class CatalogPricing(BaseModel):
    input_per_million: str
    output_per_million: str
    cache_read_input_per_million: str
    cache_write_input_per_million: str
    above_threshold: AboveThreshold | None = None


class CatalogModel(BaseModel):
    id: str
    type: str
    context_length: int | None = None
    max_output_tokens: int | None = None
    pricing: CatalogPricing


CATALOG_ADAPTER: Final = TypeAdapter(list[CatalogModel])

RegistryEntry = dict[str, object]
CostMap = dict[str, object]

SYNCED_LIMIT_FIELDS: Final = ("max_input_tokens", "max_output_tokens", "max_tokens")
SYNCED_PRICE_FIELDS: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "input_cost_per_token_above_272k_tokens",
    "output_cost_per_token_above_272k_tokens",
    "cache_read_input_token_cost_above_272k_tokens",
    "cache_creation_input_token_cost_above_272k_tokens",
)


def per_token(price_per_million: str) -> float:
    return float(f"{float(price_per_million) / 1e6:.6g}")


def _price_fields(pricing: CatalogPricing) -> RegistryEntry:
    base: Final[RegistryEntry] = {
        "input_cost_per_token": per_token(pricing.input_per_million),
        "output_cost_per_token": per_token(pricing.output_per_million),
        "cache_read_input_token_cost": per_token(pricing.cache_read_input_per_million),
        "cache_creation_input_token_cost": per_token(pricing.cache_write_input_per_million),
    }
    if pricing.above_threshold is None:
        return base
    above: Final = pricing.above_threshold
    return {
        **base,
        "input_cost_per_token_above_272k_tokens": per_token(above.input_per_million),
        "output_cost_per_token_above_272k_tokens": per_token(above.output_per_million),
        "cache_read_input_token_cost_above_272k_tokens": per_token(above.cache_read_input_per_million),
        "cache_creation_input_token_cost_above_272k_tokens": per_token(above.cache_write_input_per_million),
    }


def _limit_fields(model: CatalogModel) -> RegistryEntry:
    fields: Final[RegistryEntry] = {}
    if model.context_length is not None:
        fields["max_input_tokens"] = model.context_length
    if model.max_output_tokens is not None:
        fields["max_output_tokens"] = model.max_output_tokens
        fields["max_tokens"] = model.max_output_tokens
    return fields


def _synced_fields(model: CatalogModel) -> RegistryEntry:
    return {**_price_fields(model.pricing), **_limit_fields(model)}


@dataclass(frozen=True, slots=True)
class SyncOutcome:
    cost_map: CostMap
    updated: tuple[str, ...] = ()
    unpriced_new_models: tuple[str, ...] = ()
    missing_from_catalog: tuple[str, ...] = ()

    @property
    def has_changes(self) -> bool:
        return bool(self.updated)


def compute_sync(cost_map: CostMap, catalog: Sequence[CatalogModel]) -> SyncOutcome:
    by_id: Final = {model.id: model for model in catalog}
    registry_ids: Final = {key.removeprefix(PREFIX): key for key in cost_map if key.startswith(PREFIX)}

    updated: Final[list[str]] = []
    missing: Final[list[str]] = []
    result: Final[CostMap] = dict(cost_map)

    for model_id, key in sorted(registry_ids.items()):
        model = by_id.get(model_id)
        if model is None:
            missing.append(key)
            continue
        entry = result.get(key)
        if not isinstance(entry, dict):
            continue
        desired: Final = _synced_fields(model)
        stale_tiered: Final = () if model.pricing.above_threshold else tuple(
            f for f in SYNCED_PRICE_FIELDS if f.endswith("_above_272k_tokens")
        )
        changes: Final = tuple(
            f"{name}: {entry.get(name)!r} -> {value!r}" for name, value in desired.items() if entry.get(name) != value
        ) + tuple(f"{name}: {entry[name]!r} removed (no longer above a threshold)" for name in stale_tiered if name in entry)
        if not changes:
            continue
        merged: Final = {name: value for name, value in {**entry, **desired}.items() if name not in stale_tiered}
        result[key] = dict(sorted(merged.items()))
        updated.append(f"{key}: " + "; ".join(changes))

    unpriced_new: Final = tuple(
        sorted(f"{PREFIX}{model.id}" for model in catalog if model.id not in registry_ids)
    )

    return SyncOutcome(
        cost_map=result,
        updated=tuple(updated),
        unpriced_new_models=unpriced_new,
        missing_from_catalog=tuple(sorted(missing)),
    )


def _section_block(title: str, lines: Sequence[str]) -> str:
    bullets: Final = "\n".join(f"- `{line}`" for line in lines) or "- none"
    return f"### {title} ({len(lines)})\n{bullets}\n"


def render_pr_body(outcome: SyncOutcome) -> str:
    return (
        f"Automated sync of the cheaperinference entries in model_prices_and_context_window.json against "
        f"`GET {MODELS_URL}` by scripts/sync_cheaperinference_models.py\n"
        "\n"
        f"{_section_block('Updated', outcome.updated)}"
        "\n"
        f"{_section_block('New in the catalog, not yet registered (needs a human review for capabilities)', outcome.unpriced_new_models)}"
        "\n"
        f"{_section_block('Registered but missing from the catalog', outcome.missing_from_catalog)}"
    )


def render_summary(outcome: SyncOutcome) -> str:
    return (
        f"updated={len(outcome.updated)} new_unregistered={len(outcome.unpriced_new_models)} "
        f"missing={len(outcome.missing_from_catalog)}"
    )


def load_catalog(raw: bytes) -> list[CatalogModel]:
    parsed: Final = json.loads(raw)
    entries: Final = parsed.get("data") if isinstance(parsed, dict) else parsed
    try:
        return CATALOG_ADAPTER.validate_python(entries)
    except ValidationError as error:
        raise SyncError(f"the catalog response no longer matches the expected shape: {error}") from error


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
    parser.add_argument("--pr-body-file", type=Path, help="write the generated PR body to this path")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args: Final = parser.parse_args(argv)

    if args.models_json is not None:
        catalog_raw: Final = args.models_json.read_bytes()
    else:
        api_key: Final = os.environ.get("CHEAPERINFERENCE_API_KEY")
        if not api_key:
            raise SyncError("CHEAPERINFERENCE_API_KEY is not set and --models-json was not given")
        catalog_raw = _fetch(MODELS_URL, {"Authorization": f"Bearer {api_key}"})  # rebind-ok: branch-dependent source

    catalog: Final = load_catalog(catalog_raw)
    cost_map_path: Final = args.repo_root / COST_MAP_RELPATHS[0]
    cost_map: Final = json.loads(cost_map_path.read_text())
    outcome: Final = compute_sync(cost_map, catalog)
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
