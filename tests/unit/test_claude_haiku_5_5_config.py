"""
Validate Claude Haiku 5.5 model configuration entries.

Haiku 5.5 ships with adaptive thinking on by default, but unlike Sonnet 5.5 /
Opus 5.5 thinking can still be turned off (``thinking: disabled`` at high
effort or below) and it accepts a forced ``tool_choice`` (``any`` or a named
tool). Its cost-map rows therefore carry ``thinking_always_on: false`` and
``supports_forced_tool_use: true``.
"""

import json
import os
from collections.abc import Iterator
from typing import Final, cast

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.llms.anthropic.common_utils import AnthropicModelInfo

REPO_ROOT: Final = os.path.join(os.path.dirname(__file__), "../..")

GET_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    },
}


@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def _load_root_cost_map() -> dict[str, dict[str, object]]:
    json_path: Final = os.path.join(REPO_ROOT, "model_prices_and_context_window.json")
    with open(json_path) as f:
        return cast(dict[str, dict[str, object]], json.load(f))


HAIKU_5_5_VARIANTS: Final = (
    "claude-haiku-5-5",
    "anthropic.claude-haiku-5-5",
    "apac.anthropic.claude-haiku-5-5",
    "au.anthropic.claude-haiku-5-5",
    "eu.anthropic.claude-haiku-5-5",
    "global.anthropic.claude-haiku-5-5",
    "jp.anthropic.claude-haiku-5-5",
    "us.anthropic.claude-haiku-5-5",
    "us-gov.anthropic.claude-haiku-5-5",
    "bedrock/us-gov-east-1/anthropic.claude-haiku-5-5",
    "bedrock/us-gov-west-1/anthropic.claude-haiku-5-5",
    "bedrock_mantle/anthropic.claude-haiku-5-5",
    "bedrock_mantle/us-gov-west-1/anthropic.claude-haiku-5-5",
    "vertex_ai/claude-haiku-5-5",
    "vertex_ai/claude-haiku-5-5@default",
)


@pytest.mark.parametrize("model_name", HAIKU_5_5_VARIANTS)
def test_haiku_5_5_rows_allow_disabling_thinking_and_forced_tools(
    model_name: str,
) -> None:
    root: Final = _load_root_cost_map()
    backup: Final = GetModelCostMap.load_local_model_cost_map()
    assert model_name in root
    row: Final = root[model_name]
    # https://platform.claude.com/docs/en/models/haiku-5-5/whats-new-haiku-5-5 (2026-10-07):
    # thinking can be disabled, forced tool_choice accepted
    assert row["thinking_always_on"] is False
    assert row["supports_forced_tool_use"] is True
    assert backup[model_name] == row


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("claude-haiku-5-5", "anthropic"),
        ("anthropic/claude-haiku-5-5", "anthropic"),
        ("vertex_ai/claude-haiku-5-5", "vertex_ai"),
    ],
)
def test_haiku_5_5_runtime_profile(local_model_cost_map: None, model: str, provider: str) -> None:
    assert AnthropicModelInfo.is_adaptive_thinking_model(model, provider) is True
    assert AnthropicModelInfo._is_always_on_thinking_model(model, provider) is False
    assert AnthropicModelInfo.forced_tool_use_unsupported(model.removeprefix("anthropic/")) is False


def test_haiku_5_5_anthropic_tool_choice_required_maps_to_any(
    local_model_cost_map: None,
) -> None:
    optional_params: Final = litellm.AnthropicConfig().map_openai_params(
        non_default_params={
            "tools": [dict(GET_WEATHER_TOOL)],
            "tool_choice": "required",
        },
        optional_params={},
        model="claude-haiku-5-5",
        drop_params=False,
    )
    assert optional_params["tool_choice"] == {"type": "any"}


def test_haiku_5_5_bedrock_tool_choice_required_maps_to_any(
    local_model_cost_map: None,
) -> None:
    optional_params: Final = litellm.AmazonConverseConfig().map_openai_params(
        non_default_params={
            "tools": [dict(GET_WEATHER_TOOL)],
            "tool_choice": "required",
        },
        optional_params={},
        model="us.anthropic.claude-haiku-5-5",
        drop_params=False,
    )
    assert optional_params["tool_choice"] == {"any": {}}
