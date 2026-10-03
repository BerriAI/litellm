"""Sync the aiand entries of model_prices_and_context_window.json with aiand's live model spec.

Pulls ``GET https://api.aiand.com/v1/api.json`` (public, no auth), maps spec fields onto
registry fields, and diffs the result against the registry. Dry run (the default) prints the
diff summary and the generated PR body; ``--write`` applies the changes to the root cost map
and its ``litellm/`` backup copy.

Policy highlights:
- Prices arrive per 1M tokens with float artifacts and are normalized to clean per-token values.
- Registry entries are never deleted; a model absent from the live spec is stamped with
  ``metadata.absent_from_spec_since`` (a real, PR-worthy file change) and surfaced as a
  warning for a human deprecation call; the stamp is cleared when the model reappears in
  the spec.
- The spec cannot express endpoint support or caching behavior, so ``supported_endpoints`` and
  ``supports_prompt_caching`` stay fixed for the whole provider.
- Reasoning effort levels map from the spec's ``effort`` reasoning option, when one is declared.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

SPEC_URL: Final = "https://api.aiand.com/v1/api.json"
PROVIDER: Final = "aiand"
PREFIX: Final = "aiand/"
SOURCE_URL: Final = "https://api.aiand.com/v1/api.json"
SUPPORTED_ENDPOINTS: Final = ("/v1/chat/completions", "/v1/responses", "/v1/messages")
COST_MAP_RELPATHS: Final = (
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
)


class SyncError(RuntimeError):
    pass


class SpecCost(BaseModel):
    input: float
    output: float
    cache_read: float


class SpecLimit(BaseModel):
    context: int
    output: int


class SpecModalities(BaseModel):
    input: list[str]


class SpecReasoningOption(BaseModel):
    type: str
    values: list[str]


class SpecModel(BaseModel):
    id: str
    name: str
    family: str
    reasoning: bool
    reasoning_options: list[SpecReasoningOption] = []
    tool_call: bool
    structured_output: bool
    temperature: bool
    attachment: bool
    open_weights: bool
    cost: SpecCost
    limit: SpecLimit
    modalities: SpecModalities


SPEC_ADAPTER: Final = TypeAdapter(dict[str, SpecModel])

RegistryEntry = dict[str, object]
CostMap = dict[str, object]


@dataclass(frozen=True, slots=True)
class SyncOutcome:
    cost_map: CostMap
    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.updated or self.removed or self.warnings)


def per_token(price_per_million: float) -> float:
    return float(f"{price_per_million / 1e6:.6g}")


def _today() -> str:
    return datetime.now(tz=timezone.utc).date().isoformat()


def _effort_levels(model: SpecModel) -> tuple[str, ...]:
    for option in model.reasoning_options:
        if option.type == "effort":
            return tuple(option.values)
    return ()


def _spec_fields(model: SpecModel) -> RegistryEntry:
    effort_levels: Final = _effort_levels(model)
    fields: RegistryEntry = {
        "litellm_provider": PROVIDER,
        "mode": "chat",
        "input_cost_per_token": per_token(model.cost.input),
        "output_cost_per_token": per_token(model.cost.output),
        "cache_read_input_token_cost": per_token(model.cost.cache_read),
        "max_input_tokens": model.limit.context,
        "max_output_tokens": model.limit.output,
        "max_tokens": model.limit.output,
        "supports_function_calling": model.tool_call,
        "supports_native_streaming": True,
        "supports_parallel_function_calling": model.tool_call,
        "supports_tool_choice": model.tool_call,
        "supports_response_schema": model.structured_output,
        "supports_prompt_caching": True,
        "supports_system_messages": True,
        "supports_reasoning": model.reasoning,
        "supports_vision": "image" in model.modalities.input,
        "source": SOURCE_URL,
        "supported_endpoints": list(SUPPORTED_ENDPOINTS),
    }
    if effort_levels:
        fields["reasoning_effort_levels"] = list(effort_levels)
    return fields


def _new_entry(model: SpecModel) -> RegistryEntry:
    return _spec_fields(model)


def _updated_entry(entry: RegistryEntry, model: SpecModel) -> tuple[RegistryEntry, tuple[str, ...]]:
    desired: Final = _spec_fields(model)
    changes: Final = tuple(
        f"{name}: {entry.get(name)!r} -> {value!r}" for name, value in desired.items() if entry.get(name) != value
    )
    extras: Final = dict(sorted((name, value) for name, value in entry.items() if name not in desired))
    return {**desired, **extras}, changes


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


def compute_sync(cost_map: CostMap, spec: Mapping[str, SpecModel]) -> SyncOutcome:
    spec_ids: Final = frozenset(spec)
    registry_ids: Final = {key.removeprefix(PREFIX): key for key in cost_map if key.startswith(PREFIX)}

    added: Final[list[str]] = []
    updated: Final[list[str]] = []
    removed: Final[list[str]] = []
    warnings: Final[list[str]] = []
    result: Final[CostMap] = dict(cost_map)

    for model_id, model in sorted(spec.items()):
        key: Final = f"{PREFIX}{model_id}"
        entry = result.get(key)
        if not isinstance(entry, dict):
            result[key] = _new_entry(model)
            added.append(key)
            continue
        new_entry, changes = _updated_entry(entry, model)
        metadata = new_entry.get("metadata")
        stamped = metadata.get("absent_from_spec_since") if isinstance(metadata, dict) else None
        if stamped is not None:
            remaining_metadata: Final = {
                name: value for name, value in metadata.items() if name != "absent_from_spec_since"
            }
            if remaining_metadata:
                new_entry["metadata"] = dict(sorted(remaining_metadata.items()))
            else:
                new_entry.pop("metadata")
            changes = (
                *changes,
                f"metadata.absent_from_spec_since: {stamped!r} -> None (model reappeared in the spec)",
            )
        if changes:
            updated.append(f"{key}: " + "; ".join(changes))
            result[key] = new_entry

    for model_id, key in sorted(registry_ids.items()):
        if model_id in spec_ids:
            continue
        removed.append(key)
        warnings.append(
            f"`{key}` is absent from the live spec; the registry entry is kept (never deleted) "
            "and needs a human deprecation call"
        )
        entry = result.get(key)
        if not isinstance(entry, dict):
            continue
        metadata = entry.get("metadata")
        curated: Final = metadata.get("absent_from_spec_since") if isinstance(metadata, dict) else None
        if curated is not None:
            continue
        extras: Final = dict(metadata) if isinstance(metadata, dict) else {}
        stamped_entry: Final = dict(entry)
        stamped_entry["metadata"] = dict(sorted({**extras, "absent_from_spec_since": _today()}.items()))
        result[key] = dict(sorted(stamped_entry.items()))
        updated.append(f"{key}: metadata.absent_from_spec_since: None -> {_today()!r}")
    return SyncOutcome(
        cost_map=_with_new_keys_in_block(cost_map, result, tuple(added)),
        added=tuple(added),
        updated=tuple(updated),
        removed=tuple(removed),
        warnings=tuple(warnings),
    )


def _section_block(title: str, lines: Sequence[str], backtick: bool) -> str:
    bullets: Final = "\n".join(f"- `{line}`" if backtick else f"- {line}" for line in lines) or "- none"
    return f"### {title} ({len(lines)})\n{bullets}\n"


def render_pr_body(outcome: SyncOutcome) -> str:
    return (
        "Automated daily sync of the aiand entries in model_prices_and_context_window.json against "
        f"`GET {SPEC_URL}` by scripts/sync_aiand_models.py.\n"
        "\n"
        f"{_section_block('Added', outcome.added, backtick=True)}"
        "\n"
        f"{_section_block('Updated', outcome.updated, backtick=True)}"
        "\n"
        f"{_section_block('Removed from the spec', outcome.warnings, backtick=False)}"
    )


def render_summary(outcome: SyncOutcome) -> str:
    return (
        f"added={len(outcome.added)} updated={len(outcome.updated)} "
        f"removed={len(outcome.removed)} warnings={len(outcome.warnings)}"
    )


def load_spec(raw: bytes) -> dict[str, SpecModel]:
    parsed: Final = json.loads(raw)
    provider: Final = parsed.get("aiand") if isinstance(parsed, dict) else None
    models: Final = provider.get("models") if isinstance(provider, dict) else None
    try:
        spec: Final = SPEC_ADAPTER.validate_python(models)
    except ValidationError as error:
        raise SyncError(f"the spec response no longer matches the expected shape: {error}") from error
    if not spec:
        raise SyncError("the spec response contains no aiand models; refusing to rewrite the registry")
    for model_id, model in spec.items():
        if model.id != model_id:
            raise SyncError(f"spec model id {model.id!r} does not match its key {model_id!r}")
    return spec


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
    parser.add_argument("--spec-json", type=Path, help="recorded spec response to use instead of the live API")
    parser.add_argument("--pr-body-file", type=Path, help="write the generated PR body to this path")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args: Final = parser.parse_args(argv)

    if args.spec_json is not None:
        spec_raw: Final = args.spec_json.read_bytes()
    else:
        spec_raw = _fetch(SPEC_URL)  # rebind-ok: branch-dependent source
    spec: Final = load_spec(spec_raw)

    cost_map_path: Final = args.repo_root / COST_MAP_RELPATHS[0]
    cost_map: Final = json.loads(cost_map_path.read_text())
    outcome: Final = compute_sync(cost_map, spec)
    body: Final = render_pr_body(outcome)

    if args.pr_body_file is not None and outcome.has_changes:
        args.pr_body_file.write_text(body)
    if args.write and (outcome.added or outcome.updated):
        for relpath in COST_MAP_RELPATHS:
            (args.repo_root / relpath).write_text(_serialize(outcome.cost_map))
    print(render_summary(outcome))
    print()
    print(body)
    if not args.write:
        print("dry run: no files were touched")
    elif not (outcome.added or outcome.updated):
        print("registry already in sync: no files were touched")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SyncError as error:
        print(f"SYNC FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
