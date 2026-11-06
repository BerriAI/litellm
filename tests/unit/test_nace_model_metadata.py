import json
from pathlib import Path
from typing import Final

import pytest

import litellm

pytestmark = pytest.mark.usefixtures("local_model_cost_map")

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
PRICES_PATH: Final = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PRICES_PATH: Final = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"


def _nace_rows(path: Path) -> dict[str, object]:
    prices: Final = json.loads(path.read_text())
    return {key: row for key, row in prices.items() if key.startswith("nace/")}


def test_drex_max_versions_are_priced_at_three_cents_per_million_input_tokens() -> None:
    """Both Drex Max versions cost $0.03 per 1M input tokens, half of drex-v2.0's $0.06; output is not billed
    (console.nace.ai/docs/drex/reference/models, read 2026-10-30)."""
    for model in ("nace/drex-max", "nace/drex-max-v1.1"):
        row = litellm.model_cost[model]
        assert row["input_cost_per_token"] == 3e-08
        assert row["output_cost_per_token"] == 0.0
        assert (row["litellm_provider"], row["mode"], row["max_input_tokens"]) == ("nace", "evaluation", 262144)
    assert litellm.model_cost["nace/drex-v2.0"]["input_cost_per_token"] == 6e-08


def test_drex_v1_0_is_priced_as_the_drex_max_that_now_serves_it() -> None:
    """drex-v1.0 is retired and answered by drex-max at drex-max's price and limits
    (console.nace.ai/docs/drex/reference/models, read 2026-10-30)."""
    drex_v1: Final = litellm.model_cost["nace/drex-v1.0"]
    drex_max: Final = litellm.model_cost["nace/drex-max"]

    assert drex_v1["input_cost_per_token"] == drex_max["input_cost_per_token"]
    assert drex_v1["max_input_tokens"] == drex_max["max_input_tokens"]


def test_drex_max_versions_are_the_nace_models_that_read_images() -> None:
    """Only drex-max, drex-max-v1.1 and the retired drex-v1.0 (now served by drex-max) read images; the other Drex
    models answer 422 to a non-empty images array (console.nace.ai/docs/drex/guides/drex-max, read 2026-10-31)."""
    vision_rows: Final = [key for key, row in _nace_rows(PRICES_PATH).items() if row.get("supports_vision")]

    assert litellm.supports_vision("nace/drex-max") is True
    assert litellm.supports_vision("nace/drex-max-v1.1") is True
    assert litellm.supports_vision("nace/drex-v1.0") is True
    assert sorted(vision_rows) == ["nace/drex-max", "nace/drex-max-v1.1", "nace/drex-v1.0"]


def test_nace_backup_rows_match_the_main_catalog() -> None:
    assert {"nace/drex-max", "nace/drex-max-v1.1"} <= _nace_rows(PRICES_PATH).keys()
    assert _nace_rows(BACKUP_PRICES_PATH) == _nace_rows(PRICES_PATH)
