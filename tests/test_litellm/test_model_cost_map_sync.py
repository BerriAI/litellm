"""The two shipped model cost maps must stay in lockstep.

``model_prices_and_context_window.json`` (repo root) is the canonical map that gets
published and fetched at runtime. ``litellm/model_prices_and_context_window_backup.json``
is bundled into the package and is what LiteLLM falls back to when the remote fetch
fails or when ``LITELLM_LOCAL_MODEL_COST_MAP=True``. If they drift, the same model
silently resolves to different metadata depending on whether the remote fetch
succeeded -- wrong context windows, missing models, missing endpoints.

``ci_cd/check_files_match.py`` was written to enforce exactly this, but it is wired
into nothing (it used to be a pre-commit hook, and the pre-commit config was removed),
so drift accumulated unnoticed. Existing per-model checks such as
``test_gpt_5_6_backup_matches_main`` only cover the handful of models each was written
for, so they cannot catch drift in any other entry.

These tests assert the invariant globally.
"""

import filecmp
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

_RESYNC_HINT = "Run `python ci_cd/check_files_match.py` after updating the canonical map."


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _describe_drift(main_map: dict, backup_map: dict, limit: int = 20) -> str:
    only_main = sorted(set(main_map) - set(backup_map))
    only_backup = sorted(set(backup_map) - set(main_map))

    field_diffs = []
    for key in sorted(set(main_map) & set(backup_map)):
        main_entry, backup_entry = main_map[key], backup_map[key]
        if not isinstance(main_entry, dict) or not isinstance(backup_entry, dict):
            if main_entry != backup_entry:
                field_diffs.append(f"{key}: {main_entry!r} != {backup_entry!r}")
            continue
        for field in sorted(set(main_entry) | set(backup_entry)):
            main_value = main_entry.get(field, "<absent>")
            backup_value = backup_entry.get(field, "<absent>")
            if main_value != backup_value:
                field_diffs.append(f"{key}.{field}: main={main_value!r} backup={backup_value!r}")

    lines = []
    if only_main:
        lines.append(f"{len(only_main)} entries missing from backup: {only_main[:limit]}")
    if only_backup:
        lines.append(f"{len(only_backup)} entries missing from main: {only_backup[:limit]}")
    if field_diffs:
        lines.append(f"{len(field_diffs)} field differences:")
        lines.extend(f"  {diff}" for diff in field_diffs[:limit])
        if len(field_diffs) > limit:
            lines.append(f"  ... and {len(field_diffs) - limit} more")
    return "\n".join(lines)


def test_backup_cost_map_has_same_contents_as_canonical():
    """Every entry and field must resolve identically from either map."""
    main_map = _load(MAIN_PATH)
    backup_map = _load(BACKUP_PATH)

    if main_map != backup_map:
        raise AssertionError(
            "model cost maps have drifted.\n" + _describe_drift(main_map, backup_map) + "\n" + _RESYNC_HINT
        )


def test_backup_cost_map_is_byte_identical_to_canonical():
    """The two files are kept byte-identical, which is the contract ci_cd/check_files_match.py enforces.

    Separate from the parsed-content check so a pure formatting divergence reports as
    a formatting problem rather than as missing or wrong model metadata.
    """
    assert filecmp.cmp(MAIN_PATH, BACKUP_PATH, shallow=False), (
        f"{MAIN_PATH.name} and {BACKUP_PATH.name} differ byte-for-byte. " + _RESYNC_HINT
    )
