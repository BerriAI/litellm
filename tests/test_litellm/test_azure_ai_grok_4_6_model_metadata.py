from pathlib import Path
from typing import Final

from pydantic import TypeAdapter


REPO_ROOT: Final = Path(__file__).parents[2]
MODEL: Final = "azure_ai/grok-4.6"
COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])


def _cost_map_entry(path: Path) -> dict[str, object]:
    return COST_MAP_ADAPTER.validate_json(path.read_bytes())[MODEL]


def test_azure_ai_grok_4_6_entry_source_and_backup_match() -> None:
    main_entry = _cost_map_entry(REPO_ROOT / "model_prices_and_context_window.json")
    backup_entry = _cost_map_entry(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json")

    assert backup_entry == main_entry
