"""
Validate Claude Sonnet 5 model configuration entries.

Sonnet 5 ships with the gen-5 adaptive-thinking profile (adaptive thinking
always on, no extended thinking, ``effort`` defaults to ``high``), so it must
mirror the sampling-param and prefill restrictions that Fable 5 / Opus 4.8 carry
rather than the older Sonnet 4.6 behavior. The cost-map entries are also what
populate ``litellm.anthropic_models`` at import, which is what lets a bare
``claude-sonnet-5`` name resolve to the ``anthropic`` provider (and match an
``anthropic/*`` wildcard deployment).

Sonnet 5.5 (``claude-sonnet-5-5``) is covered here too. It carries the same
gen-5 profile, and forced tool use is not supported, same as Opus 5.5. Thinking
cannot be turned off via ``thinking.type=disabled``, the Sonnet 5 shape; the
off-equivalent is ``thinking.type=between_tools``, which ``maybe_drop_disabled_thinking``
remaps a caller's ``disabled`` request to.
"""

import json
import os

import pytest

from litellm.constants import BEDROCK_CONVERSE_MODELS
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../..")


def _load_root_cost_map() -> dict:
    json_path = os.path.join(REPO_ROOT, "model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


ALL_SONNET_5_VARIANTS = (
    "claude-sonnet-5",
    "anthropic.claude-sonnet-5",
    "global.anthropic.claude-sonnet-5",
    "us.anthropic.claude-sonnet-5",
    "eu.anthropic.claude-sonnet-5",
    "au.anthropic.claude-sonnet-5",
    "jp.anthropic.claude-sonnet-5",
    "vertex_ai/claude-sonnet-5",
    "vertex_ai/claude-sonnet-5@default",
    "azure_ai/claude-sonnet-5",
)


def test_sonnet_5_registered_for_bedrock_converse():
    assert "anthropic.claude-sonnet-5" in BEDROCK_CONVERSE_MODELS


SONNET_5_5_VARIANTS = (
    "claude-sonnet-5-5",
    "us.anthropic.claude-sonnet-5-5",
    "vertex_ai/claude-sonnet-5-5",
    "vertex_ai/claude-sonnet-5-5@default",
    "azure_ai/claude-sonnet-5-5",
    "openrouter/anthropic/claude-sonnet-5.5",
)


@pytest.mark.parametrize("model_name", SONNET_5_5_VARIANTS)
def test_sonnet_5_5_present_in_bundled_backup(model_name):
    backup = GetModelCostMap.load_local_model_cost_map()
    root = _load_root_cost_map()
    assert model_name in backup
    assert model_name in root
    assert backup[model_name] == root[model_name]


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("claude-sonnet-5-5", "anthropic"),
        ("anthropic/claude-sonnet-5-5", "anthropic"),
        ("vertex_ai/claude-sonnet-5-5", "vertex_ai"),
        ("azure_ai/claude-sonnet-5-5", "azure_ai"),
    ],
)
def test_sonnet_5_5_thinking_profile(local_model_cost_map, model, provider):
    """Sonnet 5.5 has thinking always on with the adaptive thinking surface, and
    no forced tool use, same as Opus 5.5. Unlike the rest of that family, it also
    has a real off-equivalent for thinking: between_tools."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    assert AnthropicModelInfo._is_adaptive_thinking_model(model, provider) is True
    assert AnthropicModelInfo._is_always_on_thinking_model(model, provider) is True
    assert AnthropicModelInfo._supports_between_tools_thinking(model, provider) is True
    assert AnthropicModelInfo.forced_tool_use_unsupported(model.removeprefix("anthropic/")) is True


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("claude-opus-5-5", "anthropic"),
        ("claude-fable-5", "anthropic"),
        ("claude-mythos-5", "anthropic"),
    ],
)
def test_other_always_on_thinking_models_lack_between_tools(local_model_cost_map, model, provider):
    """Sonnet 5.5 is the only always-on-thinking family with a confirmed
    between_tools off-equivalent today; the rest must keep falling back to the
    drop-and-warn path in maybe_drop_disabled_thinking."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    assert AnthropicModelInfo._is_always_on_thinking_model(model, provider) is True
    assert AnthropicModelInfo._supports_between_tools_thinking(model, provider) is False
