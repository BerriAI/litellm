"""Guard the cost map on pull requests.

Every pull request whose diff against its merge base touches one of the two cost map files or the pins file gets the
file checks: the files parse, the backup copy matches the root file, and every capability pinned in
ci_cd/cost_map_pins.json (a value verified against a live provider call where the provider's model listing says
otherwise) still holds in the tree that merging the pull request into its base branch produces, so a branch cut before
a pin was added cannot lower it, and a pull request that deletes the pins file fails too. A pull request that leaves
all three untouched skips them, since merging it keeps the base branch's copies and its head tree only carries
whatever state the branch was cut from. Pull requests from the cost map sync bot (branches named
litellm_cost_map_sync_*) always get the file checks and additionally may only touch the two cost map files and may
only add or update models, so a sync can never lower a pinned capability or edit the pins. Price sync pull requests
(branches named litellm-providers/price-sync*) and pull requests a bot account opened (COST_MAP_GUARD_AUTHOR_TYPE is
Bot) get the human checks and may not touch the pins file either.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

COST_MAP_PATH: Final = "model_prices_and_context_window.json"
BACKUP_PATH: Final = "litellm/model_prices_and_context_window_backup.json"
PINS_PATH: Final = "ci_cd/cost_map_pins.json"
GUARDED_PATHS: Final = (COST_MAP_PATH, BACKUP_PATH)
CHECKED_PATHS: Final = (*GUARDED_PATHS, PINS_PATH)
BOT_BRANCH_PREFIX: Final = "litellm_cost_map_sync_"
PRICE_SYNC_BRANCH_PREFIX: Final = "litellm-providers/price-sync"
TARGET_ENV: Final = "COST_MAP_GUARD_TARGET"
AUTHOR_TYPE_ENV: Final = "COST_MAP_GUARD_AUTHOR_TYPE"
BOT_AUTHOR_TYPE: Final = "Bot"
MERGE_CONFLICT_EXIT_CODE: Final = 1
SPECIAL_ROOT_KEYS: Final = frozenset({"sample_spec", "fallback_generalizations"})
EVALUATION_OUTPUT_PRICE_SOURCES: Final[Mapping[str, str]] = MappingProxyType({})

CostMap = dict[str, object]
_MISSING: Final = object()


@dataclass(frozen=True, slots=True)
class Snapshot:
    cost_map: str
    backup: str
    pins: str


@dataclass(frozen=True, slots=True)
class MergeFailed:
    stderr: str


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


def _evaluation_output_price_failure(
    key: str, entry: object, evaluation_output_price_sources: Mapping[str, str]
) -> str | None:
    if not isinstance(entry, dict) or entry.get("mode") not in ("evaluation", "decisions"):
        return None
    output_cost: Final = entry.get("output_cost_per_token")
    if (
        not isinstance(output_cost, (int, float))
        or isinstance(output_cost, bool)
        or output_cost <= 0
        or key in evaluation_output_price_sources
    ):
        return None
    return (
        f"{key}: mode evaluation and decisions rows bill input only; output_cost_per_token must be 0 unless the vendor "
        "prices output tokens for this model (add the source to EVALUATION_OUTPUT_PRICE_SOURCES)"
    )


def _file_failures(
    head: Snapshot,
    head_map: CostMap,
    *,
    evaluation_output_price_sources: Mapping[str, str] = EVALUATION_OUTPUT_PRICE_SOURCES,
) -> tuple[str, ...]:
    backup_failure: Final = (
        ()
        if head.backup == head.cost_map
        else (f"{BACKUP_PATH} differs from {COST_MAP_PATH}; copy the root file over it",)
    )
    return (
        *backup_failure,
        *(
            failure
            for key, entry in head_map.items()
            if (
                failure := _evaluation_output_price_failure(
                    key, entry, evaluation_output_price_sources
                )
            )
            is not None
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


def _pin_violations(parsed: CostMap, entries: Mapping[str, Mapping[str, object]]) -> Iterator[str]:
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


def pin_author_failures(head_ref: str, author_type: str, changed_files: Sequence[str]) -> tuple[str, ...]:
    if PINS_PATH not in changed_files:
        return ()
    if head_ref.startswith(PRICE_SYNC_BRANCH_PREFIX):
        return (f"price sync PRs may not change {PINS_PATH}; a pin moves only in a human PR with new evidence",)
    if author_type == BOT_AUTHOR_TYPE:
        return (f"PRs opened by a bot may not change {PINS_PATH}; a pin moves only in a human PR with new evidence",)
    return ()


def touches_cost_map(changed_files: Sequence[str]) -> bool:
    return any(path in CHECKED_PATHS for path in changed_files)


def contract_for(bot: bool, changed_files: Sequence[str]) -> str:
    if bot:
        return "bot contract enforced"
    return "human PR, file checks only" if touches_cost_map(changed_files) else "human PR, cost map untouched"


def _merged_pin_failures(base: Snapshot, merged: Snapshot | None) -> tuple[str, ...]:
    if merged is None:
        return ("the pull request does not merge cleanly into its base branch; merge the base branch in first",)
    if not merged.pins and not base.pins:
        return ()
    merged_map: Final = _parse_object(merged.cost_map, COST_MAP_PATH)
    if isinstance(merged_map, str):
        return (f"after merging into the base branch, {merged_map}",)
    return pin_failures(merged.pins, merged_map)


def guard_failures(
    base: Snapshot,
    head: Snapshot,
    merged: Snapshot | None,
    changed_files: Sequence[str],
    bot: bool,
    *,
    evaluation_output_price_sources: Mapping[str, str] = EVALUATION_OUTPUT_PRICE_SOURCES,
) -> tuple[str, ...]:
    if not bot and not touches_cost_map(changed_files):
        return ()
    head_map: Final = _parse_object(head.cost_map, COST_MAP_PATH)
    if isinstance(head_map, str):
        return (head_map,)
    return (
        *_file_failures(
            head, head_map, evaluation_output_price_sources=evaluation_output_price_sources
        ),
        *_merged_pin_failures(base, merged),
        *(_bot_failures(base, head_map, changed_files) if bot else ()),
    )


def _git(*args: str) -> str | None:
    result: Final = subprocess.run(("git", *args), check=False, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def snapshot(revision: str) -> Snapshot:
    return Snapshot(*(_git("show", f"{revision}:{path}") or "" for path in CHECKED_PATHS))


def merged_tree(merge_base: str, target: str, head: str) -> str | MergeFailed | None:
    result: Final = subprocess.run(
        ("git", "merge-tree", "--write-tree", f"--merge-base={merge_base}", target, head),
        check=False,
        capture_output=True,
        text=True,
    )
    tree: Final = result.stdout.split("\n", 1)[0]
    if result.returncode == 0:
        return tree
    if result.returncode == MERGE_CONFLICT_EXIT_CODE and tree:
        return None
    return MergeFailed(result.stderr.strip())


def changed_files(base: str, head: str) -> tuple[str, ...] | None:
    diff: Final = _git("diff", "--name-only", "--no-renames", base, head)
    return None if diff is None else tuple(diff.splitlines())


def main(argv: Sequence[str]) -> int:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="merge base of the pull request")
    parser.add_argument("--head", required=True, help="head commit of the pull request")
    env_target: Final = os.environ.get(TARGET_ENV)
    parser.add_argument(
        "--target",
        default=env_target,
        required=env_target is None,
        help=f"tip of the base branch the pull request merges into, defaulting to ${TARGET_ENV}",
    )
    parser.add_argument("--head-ref", required=True, help="head branch name of the pull request")
    args: Final = parser.parse_args(argv)
    bot: Final = args.head_ref.startswith(BOT_BRANCH_PREFIX)
    changed: Final = changed_files(args.base, args.head)
    if changed is None:
        print(f"cost map guard failed: git diff {args.base} {args.head} failed, so the changed files are unknown")
        return 1
    if not bot and not touches_cost_map(changed):
        print(f"cost map guard passed ({contract_for(bot, changed)})")
        return 0
    merged: Final = merged_tree(args.base, args.target, args.head)
    if isinstance(merged, MergeFailed):
        print(f"cost map guard failed: git merge-tree into {args.target} failed: {merged.stderr}")
        return 1
    failures: Final = (
        *guard_failures(
            snapshot(args.base), snapshot(args.head), None if merged is None else snapshot(merged), changed, bot
        ),
        *pin_author_failures(args.head_ref, os.environ.get(AUTHOR_TYPE_ENV, ""), changed),
    )
    contract: Final = contract_for(bot, changed)
    if failures:
        print(f"cost map guard failed ({contract}):")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print(f"cost map guard passed ({contract})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
