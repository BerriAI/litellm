from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

import litellm

REPO_ROOT: Final = Path(__file__).parents[2]
MAIN_PATH: Final = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH: Final = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

MANTLE_GPT_5_6_CHAT_MODELS: Final = (
    "bedrock_mantle/openai.gpt-5.6-sol",
    "bedrock_mantle/openai.gpt-5.6-terra",
    "bedrock_mantle/openai.gpt-5.6-luna",
    "bedrock_mantle/us-gov-west-1/openai.gpt-5.6-terra",
    "bedrock_mantle/us-gov-west-1/openai.gpt-5.6-luna",
)


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("model", MANTLE_GPT_5_6_CHAT_MODELS)
def test_bedrock_mantle_gpt_5_6_mode_is_chat(model: str) -> None:
    main_prices = _load(MAIN_PATH)
    entry = main_prices[model]
    assert entry["mode"] == "chat"
    assert entry["use_openai_responses_path"] is True
    assert entry["supported_endpoints"] == [
        "/v1/chat/completions",
        "/v1/batch",
        "/v1/responses",
    ]
    assert entry["max_input_tokens"] == 1050000


@pytest.mark.parametrize("model", MANTLE_GPT_5_6_CHAT_MODELS)
def test_bedrock_mantle_gpt_5_6_backup_matches_main(model: str) -> None:
    main_prices = _load(MAIN_PATH)
    backup_prices = _load(BACKUP_PATH)
    assert main_prices[model] == backup_prices[model]


@pytest.mark.parametrize("model", MANTLE_GPT_5_6_CHAT_MODELS)
def test_bedrock_mantle_gpt_5_6_resolves_through_get_model_info(model: str) -> None:
    info = litellm.get_model_info(model)
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 1050000
    assert litellm.model_cost[model].get("use_openai_responses_path") is True
    assert "/v1/chat/completions" in info["supported_endpoints"]
    assert "/v1/batch" in info["supported_endpoints"]
    assert "/v1/responses" in info["supported_endpoints"]
