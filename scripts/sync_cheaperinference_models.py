"""Sync the cheaperinference entries of model_prices_and_context_window.json with the gateway's live catalog.

Pulls ``GET https://api.cheaperinference.com/v1/models`` and updates the price and limit fields of the
already-registered cheaperinference/<id> entries. Dry run (the default) prints the diff summary and the
generated PR body; ``--write`` applies the changes to the root cost map and its ``litellm/`` backup copy.

Policy highlights:
- Only price fields (input/output/cache read/cache write, and their long-context variants) and the two
  limit fields (max_input_tokens, max_output_tokens/max_tokens) are synced from the catalog.
- The catalog's own long-context threshold picks the cost-map band. A threshold that matches no band
  litellm understands leaves that entry untouched and is surfaced as a warning, so a moved threshold can
  never be written out under the wrong band.
- Capability flags (supports_vision, supports_reasoning, supports_tool_choice, and so on), litellm_provider,
  mode, source, and supported_endpoints are curated by hand and are never touched by this script.
- A catalog model with no matching registry entry is surfaced as a warning for a human to add, never
  auto-added: the capability flags for a new model cannot be derived from this endpoint.
- A registry entry whose catalog model disappeared is left untouched and surfaced as a warning: nothing is
  deleted.
- Warnings are appended to $GITHUB_STEP_SUMMARY and echoed as workflow annotations when running in Actions,
  so a run that changes no prices still reports catalog drift somewhere durable.
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
    input_token_price_threshold: int
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
BASE_PRICE_FIELDS: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
)
TIERED_BANDS: Final = {
    128_000: "above_128k_tokens",
    200_000: "above_200k_tokens",
    256_000: "above_256k_tokens",
    272_000: "above_272k_tokens",
    512_000: "above_512k_tokens",
}
BAND_TOLERANCE_TOKENS: Final = 1
TIERED_PRICE_FIELDS: Final = tuple(
    f"{field}_{band}" for band in TIERED_BANDS.values() for field in BASE_PRICE_FIELDS
)
SYNCED_PRICE_FIELDS: Final = BASE_PRICE_FIELDS + TIERED_PRICE_FIELDS


def per_token(price_per_million: str) -> float:
    return float(f"{float(price_per_million) / 1e6:.6g}")


def band_for_threshold(threshold: int) -> str | None:
    for supported, band in TIERED_BANDS.items():
        if abs(threshold - supported) <= BAND_TOLERANCE_TOKENS:
            return band
    return None


def _price_fields(pricing: CatalogPricing) -> tuple[RegistryEntry, int | None]:
    base: Final[RegistryEntry] = {
        "input_cost_per_token": per_token(pricing.input_per_million),
        "output_cost_per_token": per_token(pricing.output_per_million),
        "cache_read_input_token_cost": per_token(pricing.cache_read_input_per_million),
        "cache_creation_input_token_cost": per_token(pricing.cache_write_input_per_million),
    }
    if pricing.above_threshold is None:
        return base, None
    above: Final = pricing.above_threshold
    band: Final = band_for_threshold(above.input_token_price_threshold)
    if band is None:
        return base, above.input_token_price_threshold
    return {
        **base,
        f"input_cost_per_token_{band}": per_token(above.input_per_million),
        f"output_cost_per_token_{band}": per_token(above.output_per_million),
        f"cache_read_input_token_cost_{band}": per_token(above.cache_read_input_per_million),
        f"cache_creation_input_token_cost_{band}": per_token(above.cache_write_input_per_million),
    }, None


def _limit_fields(model: CatalogModel) -> RegistryEntry:
    fields: Final[RegistryEntry] = {}
    if model.context_length is not None:
        fields["max_input_tokens"] = model.context_length
    if model.max_output_tokens is not None:
        fields["max_output_tokens"] = model.max_output_tokens
        fields["max_tokens"] = model.max_output_tokens
    return fields


def _synced_fields(model: CatalogModel) -> tuple[RegistryEntry, int | None]:
    prices, unmappable_threshold = _price_fields(model.pricing)
    return {**prices, **_limit_fields(model)}, unmappable_threshold


@dataclass(frozen=True, slots=True)
class SyncOutcome:
    cost_map: CostMap
    updated: tuple[str, ...] = ()
    unpriced_new_models: tuple[str, ...] = ()
    missing_from_catalog: tuple[str, ...] = ()
    unmappable_thresholds: tuple[str, ...] = ()

    @property
    def has_changes(self) -> bool:
        return bool(self.updated)

    @property
    def warnings(self) -> tuple[str, ...]:
        return (
            tuple(f"new in the catalog, not registered: {key}" for key in self.unpriced_new_models)
            + tuple(f"registered but gone from the catalog: {key}" for key in self.missing_from_catalog)
            + tuple(f"long-context threshold matches no cost-map band, entry left untouched: {key}" for key in self.unmappable_thresholds)
        )


def compute_sync(cost_map: CostMap, catalog: Sequence[CatalogModel]) -> SyncOutcome:
    by_id: Final = {model.id: model for model in catalog}
    registry_ids: Final = {key.removeprefix(PREFIX): key for key in cost_map if key.startswith(PREFIX)}

    updated: Final[list[str]] = []
    missing: Final[list[str]] = []
    unmappable: Final[list[str]] = []
    result: Final[CostMap] = dict(cost_map)

    for model_id, key in sorted(registry_ids.items()):
        model = by_id.get(model_id)
        if model is None:
            missing.append(key)
            continue
        entry = result.get(key)
        if not isinstance(entry, dict):
            continue
        desired, unmappable_threshold = _synced_fields(model)
        if unmappable_threshold is not None:
            unmappable.append(f"{key} (threshold {unmappable_threshold})")
            continue
        stale_tiered: Final = tuple(f for f in TIERED_PRICE_FIELDS if f not in desired)
        changes: Final = tuple(
            f"{name}: {entry.get(name)!r} -> {value!r}" for name, value in desired.items() if entry.get(name) != value
        ) + tuple(f"{name}: {entry[name]!r} removed (not priced in that band any more)" for name in stale_tiered if name in entry)
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
        unmappable_thresholds=tuple(sorted(unmappable)),
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
        "\n"
        f"{_section_block('Long-context threshold with no matching cost-map band (left untouched)', outcome.unmappable_thresholds)}"
    )


def render_summary(outcome: SyncOutcome) -> str:
    return (
        f"updated={len(outcome.updated)} new_unregistered={len(outcome.unpriced_new_models)} "
        f"missing={len(outcome.missing_from_catalog)} unmappable_thresholds={len(outcome.unmappable_thresholds)}"
    )


def render_warnings_block(outcome: SyncOutcome) -> str:
    if not outcome.warnings:
        return "## cheaperinference registry sync\n\nNo catalog drift to report.\n"
    bullets: Final = "\n".join(f"- {warning}" for warning in outcome.warnings)
    return f"## cheaperinference registry sync\n\n{bullets}\n"


def load_catalog(raw: bytes) -> list[CatalogModel]:
    parsed: Final = json.loads(raw)
    entries: Final = parsed.get("data") if isinstance(parsed, dict) else parsed
    try:
        return CATALOG_ADAPTER.validate_python(entries)
    except ValidationError as error:
        raise SyncError(f"the catalog response no longer matches the expected shape: {error}") from error


def _github_step_summary() -> Path | None:
    path: Final = os.environ.get("GITHUB_STEP_SUMMARY")
    return Path(path) if path else None


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
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="append a markdown block of catalog warnings here (default: $GITHUB_STEP_SUMMARY when set)",
    )
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
    summary_path: Final = args.summary_file or _github_step_summary()
    if summary_path is not None:
        with summary_path.open("a", encoding="utf-8") as summary:
            summary.write(render_warnings_block(outcome))
    if os.environ.get("GITHUB_ACTIONS"):
        for warning in outcome.warnings:
            print(f"::warning::{warning}")
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
