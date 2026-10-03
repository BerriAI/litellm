"""Guard the cost map on pull requests.

Every pull request whose diff against its merge base touches one of the three cost map files or the pins file gets
the file checks: the files parse, the backup copy matches the root file, the JSON schema is in sync and validates the
map, and every capability pinned in ci_cd/cost_map_pins.json (a value verified against a live provider call where the
provider's model listing says otherwise) still holds in the map, so a pull request that deletes the pins file fails
too. A pull request that leaves all of them untouched
skips the checks, since merging it keeps the base branch's copies and its head tree only carries whatever state the
branch was cut from. Pull requests from the cost map sync bot (branches named litellm_cost_map_sync_*) always get the
file checks and additionally may only touch the three cost map files and may only add or update models, so a sync
can never lower a pinned capability or edit the pins.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from generate_model_prices_schema import SPECIAL_ROOT_KEYS, build_schema, render, validation_errors

COST_MAP_PATH: Final = "model_prices_and_context_window.json"
BACKUP_PATH: Final = "litellm/model_prices_and_context_window_backup.json"
SCHEMA_PATH: Final = "model_prices_and_context_window.schema.json"
PINS_PATH: Final = "ci_cd/cost_map_pins.json"
GUARDED_PATHS: Final = (COST_MAP_PATH, BACKUP_PATH, SCHEMA_PATH)
CHECKED_PATHS: Final = (*GUARDED_PATHS, PINS_PATH)
BOT_BRANCH_PREFIX: Final = "litellm_cost_map_sync_"

CostMap = dict[str, object]
_MISSING: Final = object()


@dataclass(frozen=True, slots=True)
class Snapshot:
    cost_map: str
    backup: str
    schema: str
    pins: str = ""


@dataclass(frozen=True, slots=True)
class Pin:
    key: str
    field: str
    value: object
    checked: str
    source: str


def _parse_object(text: str, path: str) -> CostMap | str:
    try:
        parsed: Final = json.loads(text)
    except json.JSONDecodeError as error:
        return f"{path} is not valid JSON: {error}"
    return parsed if isinstance(parsed, dict) else f"{path} must be a JSON object at the root"


def _rendered_schema(cost_map: CostMap) -> str:
    try:
        return render(build_schema(cost_map))
    except SystemExit as error:
        return str(error)


def _file_failures(head: Snapshot, head_map: CostMap) -> tuple[str, ...]:
    schema_text: Final = _rendered_schema(head_map)
    if not schema_text.startswith("{"):
        return (schema_text,)
    backup_failure: Final = (
        ()
        if head.backup == head.cost_map
        else (f"{BACKUP_PATH} differs from {COST_MAP_PATH}; copy the root file over it",)
    )
    schema_failure: Final = (
        ()
        if head.schema == schema_text
        else (
            f"{SCHEMA_PATH} is out of sync with {COST_MAP_PATH}; "
            "run `python ci_cd/generate_model_prices_schema.py` and commit the result",
        )
    )
    return (
        *backup_failure,
        *schema_failure,
        *(
            f"{COST_MAP_PATH} does not validate against its schema: {error}"
            for error in validation_errors(head_map, json.loads(schema_text))[:20]
        ),
    )


def _entries(cost_map: CostMap) -> dict[str, dict[str, object]]:
    return {key: entry for key, entry in cost_map.items() if isinstance(entry, dict)}


def _pin(key: str, field: str, spec: object) -> Pin | str:
    malformed: Final = f"{PINS_PATH}: {key}.{field} needs value, checked, and source"
    if not isinstance(spec, dict) or "value" not in spec:
        return malformed
    checked: Final = spec.get("checked")
    source: Final = spec.get("source")
    if not isinstance(checked, str) or not isinstance(source, str) or not checked or not source:
        return malformed
    return Pin(key, field, spec["value"], checked, source)


def _pins(parsed: CostMap) -> Iterator[Pin | str]:
    for key, fields in parsed.items():
        if not isinstance(fields, dict):
            yield f"{PINS_PATH}: {key} must map field names to pins"
            continue
        for field, spec in fields.items():
            yield _pin(key, field, spec)


def _pin_violations(parsed: CostMap, entries: dict[str, dict[str, object]]) -> Iterator[str]:
    for pin in _pins(parsed):
        if isinstance(pin, str):
            yield pin
        elif pin.key not in entries:
            yield f"{PINS_PATH} pins {pin.key}.{pin.field} but {COST_MAP_PATH} has no {pin.key} entry"
        elif entries[pin.key].get(pin.field, _MISSING) != pin.value:
            yield (
                f"{COST_MAP_PATH}: {pin.key}.{pin.field} must stay {json.dumps(pin.value)} "
                f"(verified {pin.checked}: {pin.source}); change {PINS_PATH} with new evidence first"
            )


def pin_failures(pins_text: str, head_map: CostMap) -> tuple[str, ...]:
    if not pins_text:
        return (f"{PINS_PATH} is missing; restore it, its pins were verified against live provider calls",)
    parsed: Final = _parse_object(pins_text, PINS_PATH)
    if isinstance(parsed, str):
        return (parsed,)
    return tuple(_pin_violations(parsed, _entries(head_map)))


def _bot_failures(base: Snapshot, head_map: CostMap, changed_files: Sequence[str]) -> tuple[str, ...]:
    base_map: Final = _parse_object(base.cost_map, COST_MAP_PATH)
    if isinstance(base_map, str):
        return (f"merge base: {base_map}",)
    base_entries: Final = _entries(base_map)
    head_entries: Final = _entries(head_map)
    removed_fields: Final = tuple(
        f"{key}.{field}"
        for key, entry in base_entries.items()
        if key in head_entries
        for field in entry
        if field not in head_entries[key]
    )
    return (
        *(
            f"bot PRs may only change the cost map files, not {path}"
            for path in changed_files
            if path not in GUARDED_PATHS
        ),
        *(f"bot PRs may not remove models: {key}" for key in base_map if key not in head_map),
        *(f"bot PRs may not remove fields: {ref}" for ref in removed_fields),
        *(
            f"bot PRs may not change {key}"
            for key in sorted(SPECIAL_ROOT_KEYS)
            if base_map.get(key) != head_map.get(key)
        ),
    )


def touches_cost_map(changed_files: Sequence[str]) -> bool:
    return any(path in CHECKED_PATHS for path in changed_files)


def contract_for(bot: bool, changed_files: Sequence[str]) -> str:
    if bot:
        return "bot contract enforced"
    return "human PR, file checks only" if touches_cost_map(changed_files) else "human PR, cost map untouched"


def guard_failures(base: Snapshot, head: Snapshot, changed_files: Sequence[str], bot: bool) -> tuple[str, ...]:
    if not bot and not touches_cost_map(changed_files):
        return ()
    head_map: Final = _parse_object(head.cost_map, COST_MAP_PATH)
    if isinstance(head_map, str):
        return (head_map,)
    return (
        *_file_failures(head, head_map),
        *pin_failures(head.pins, head_map),
        *(_bot_failures(base, head_map, changed_files) if bot else ()),
    )


def _git(*args: str) -> str | None:
    result: Final = subprocess.run(("git", *args), check=False, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def snapshot(revision: str) -> Snapshot:
    return Snapshot(*(_git("show", f"{revision}:{path}") or "" for path in CHECKED_PATHS))


def changed_files(base: str, head: str) -> tuple[str, ...] | None:
    diff: Final = _git("diff", "--name-only", "--no-renames", base, head)
    return None if diff is None else tuple(diff.splitlines())


def main(argv: Sequence[str]) -> int:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="merge base of the pull request")
    parser.add_argument("--head", required=True, help="head commit of the pull request")
    parser.add_argument("--head-ref", required=True, help="head branch name of the pull request")
    args: Final = parser.parse_args(argv)
    bot: Final = args.head_ref.startswith(BOT_BRANCH_PREFIX)
    changed: Final = changed_files(args.base, args.head)
    if changed is None:
        print(f"cost map guard failed: git diff {args.base} {args.head} failed, so the changed files are unknown")
        return 1
    failures: Final = guard_failures(snapshot(args.base), snapshot(args.head), changed, bot)
    contract: Final = contract_for(bot, changed)
    if failures:
        print(f"cost map guard failed ({contract}):")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print(f"cost map guard passed ({contract})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
