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
gen-5 profile but thinking cannot be turned off and forced tool use is not
supported, same as Opus 5.5
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


def test_sonnet_5_5_present_in_bundled_backup():
    backup = GetModelCostMap.load_local_model_cost_map()
    root = _load_root_cost_map()
    assert "claude-sonnet-5-5" in backup
    assert "claude-sonnet-5-5" in root
    assert backup["claude-sonnet-5-5"] == root["claude-sonnet-5-5"]


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"])
def test_sonnet_5_5_thinking_profile(local_model_cost_map, model):
    """Sonnet 5.5 has thinking always on with the adaptive thinking surface, and
    no forced tool use, same as Opus 5.5."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is True
    assert AnthropicModelInfo._is_always_on_thinking_model(model, "anthropic") is True
    assert AnthropicModelInfo.forced_tool_use_unsupported(model.removeprefix("anthropic/")) is True


