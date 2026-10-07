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

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.llms.anthropic.common_utils import AnthropicModelInfo

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")


@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch):
    """Never depend on the published cost map having this new model yet."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def _load_root_cost_map() -> dict:
    json_path = os.path.join(REPO_ROOT, "model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


HAIKU_5_5_VARIANTS = (
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
def test_haiku_5_5_rows_allow_disabling_thinking_and_forced_tools(model_name):
    root = _load_root_cost_map()
    backup = GetModelCostMap.load_local_model_cost_map()
    assert model_name in root
    row = root[model_name]
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
def test_haiku_5_5_runtime_profile(local_model_cost_map, model, provider):
    assert AnthropicModelInfo.is_adaptive_thinking_model(model, provider) is True
    assert AnthropicModelInfo._is_always_on_thinking_model(model, provider) is False
    assert AnthropicModelInfo.forced_tool_use_unsupported(model.removeprefix("anthropic/")) is False


def test_haiku_5_5_anthropic_tool_choice_required_maps_to_any(local_model_cost_map):
    """``tool_choice="required"`` on ``claude-haiku-5-5`` must not raise the
    forced-tool-use UnsupportedParamsError and maps to Anthropic ``any``."""
    optional_params = litellm.AnthropicConfig().map_openai_params(
        non_default_params={
            "tools": [
                {
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
            ],
            "tool_choice": "required",
        },
        optional_params={},
        model="claude-haiku-5-5",
        drop_params=False,
    )
    assert optional_params["tool_choice"] == {"type": "any"}


def test_haiku_5_5_bedrock_tool_choice_required_maps_to_any(local_model_cost_map):
    """``tool_choice="required"`` on the Bedrock Converse row maps to the
    Converse ``any`` block."""
    optional_params = litellm.AmazonConverseConfig().map_openai_params(
        non_default_params={
            "tools": [
                {
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
            ],
            "tool_choice": "required",
        },
        optional_params={},
        model="us.anthropic.claude-haiku-5-5",
        drop_params=False,
    )
    assert optional_params["tool_choice"] == {"any": {}}
