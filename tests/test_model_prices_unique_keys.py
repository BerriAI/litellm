"""The root model cost map must stay a unique-key mirror of the backup, with no negative token costs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COST_MAP = ROOT / "model_prices_and_context_window.json"
BACKUP = ROOT / "litellm" / "model_prices_and_context_window_backup.json"


def _load_with_duplicate_hook(path: Path) -> tuple[dict, list[str]]:
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, object]]) -> dict:
        seen: dict[str, object] = {}
        for key, value in pairs:
            if key in seen:
                duplicates.append(key)
            seen[key] = value
        return seen

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=hook)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must be a JSON object")
    return data, duplicates


def test_model_prices_unique_keys_and_backup_sync() -> None:
    root, root_duplicates = _load_with_duplicate_hook(COST_MAP)
    backup, backup_duplicates = _load_with_duplicate_hook(BACKUP)

    duplicate_key_count = len(root_duplicates) + len(backup_duplicates)
    assert duplicate_key_count == 0
    assert root_duplicates == []
    assert backup_duplicates == []

    only_root = set(root) - set(backup)
    only_backup = set(backup) - set(root)
    value_diffs = [key for key in set(root) & set(backup) if root[key] != backup[key]]
    diff_count = len(only_root) + len(only_backup) + len(value_diffs)
    assert diff_count == 0

    negative: list[tuple[str, str, int | float]] = []
    for key, entry in root.items():
        if not isinstance(entry, dict):
            continue
        for field in ("input_cost_per_token", "output_cost_per_token"):
            cost = entry.get(field)
            if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                continue
            if cost < 0:
                negative.append((key, field, cost))
    assert len(negative) == 0
