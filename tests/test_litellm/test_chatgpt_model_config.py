import json
import os
from typing import TypeAlias, cast

import pytest

from litellm.main import responses_api_bridge_check

JsonObject: TypeAlias = dict[str, object]

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")
MODEL_NAME = "chatgpt/gpt-5.4-mini"


def _load_cost_map(path: str) -> dict[str, JsonObject]:
    with open(path) as f:
        return cast(dict[str, JsonObject], json.load(f))


def _load_root_cost_map() -> dict[str, JsonObject]:
    return _load_cost_map(os.path.join(REPO_ROOT, "model_prices_and_context_window.json"))


def _load_backup_cost_map() -> dict[str, JsonObject]:
    return _load_cost_map(os.path.join(REPO_ROOT, "litellm/model_prices_and_context_window_backup.json"))


@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), _load_backup_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_chatgpt_gpt_5_4_mini_is_responses_model(cost_map: dict[str, JsonObject]) -> None:
    info = cost_map[MODEL_NAME]

    assert info["litellm_provider"] == "chatgpt"
    assert info["mode"] == "responses"
    assert info["max_input_tokens"] == 272000
    assert info["max_output_tokens"] == 128000
    assert info["max_tokens"] == 128000
    assert info["supported_endpoints"] == ["/v1/chat/completions", "/v1/responses"]
    assert info["supports_function_calling"] is True
    assert info["supports_parallel_function_calling"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_vision"] is True


def test_chatgpt_gpt_5_4_mini_bridges_chat_completions_to_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_model_info_helper(model: str, custom_llm_provider: str) -> JsonObject:
        assert model == MODEL_NAME
        assert custom_llm_provider == "chatgpt"
        return _load_backup_cost_map()[MODEL_NAME]

    monkeypatch.setattr("litellm.main._get_model_info_helper", fake_get_model_info_helper)

    model_info, model = cast(
        tuple[JsonObject, str],
        responses_api_bridge_check(model=MODEL_NAME, custom_llm_provider="chatgpt"),
    )

    assert model == MODEL_NAME
    assert model_info["mode"] == "responses"
