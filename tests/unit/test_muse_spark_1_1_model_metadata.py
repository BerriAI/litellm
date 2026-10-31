import json
from pathlib import Path

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

MUSE_SPARK_MODEL = "meta/muse-spark-1.1"


def test_muse_spark_1_1_model_info():
    routed_model, provider, _, api_base = get_llm_provider(model=MUSE_SPARK_MODEL, api_key="sk-test")
    assert routed_model == "muse-spark-1.1"
    assert provider == "meta"
    assert api_base == "https://api.meta.ai/v1"


def test_muse_spark_1_1_backup_matches_main():
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    assert backup_cost.get(MUSE_SPARK_MODEL) == main_cost.get(MUSE_SPARK_MODEL), (
        f"{MUSE_SPARK_MODEL} differs between main and backup model cost maps"
    )
