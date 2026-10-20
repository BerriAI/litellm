import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def model_cost() -> dict[str, Any]:
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def test_replicate_models_have_valid_key_prefix(model_cost: dict[str, Any]) -> None:
    replicate_models = {k for k, v in model_cost.items() if v.get("litellm_provider") == "replicate"}
    malformed = [k for k in replicate_models if not k.startswith("replicate/")]
    assert not malformed, (
        f"Replicate models must use 'replicate/owner/model' key format, found malformed keys: {malformed}"
    )
