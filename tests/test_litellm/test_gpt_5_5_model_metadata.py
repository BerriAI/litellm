import json
from pathlib import Path






def test_azure_ai_gpt_5_5_backup_matches_main():
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    for model in ("azure_ai/gpt-5.5", "azure_ai/gpt-5.5-2026-04-23"):
        assert backup_cost.get(model) == main_cost.get(
            model
        ), f"{model} differs between main and backup model cost maps"
