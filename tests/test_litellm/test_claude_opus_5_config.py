"""
Validate Claude Opus 5 model configuration entries.

Opus 5.5 (``claude-opus-5-5``) is covered here too.

Opus 5 carries Opus 4.8's pricing ($5 / $25 per MTok) and the gen-5 adaptive
thinking profile, but differs from 4.8 in two ways that are behavior-bearing in
LiteLLM: the cacheable-prefix minimum drops to 512 tokens, and Bedrock's Opus 5
validator accepts the full effort ladder, so the entries must not carry the
``bedrock_output_config_effort_ceiling`` that silently clamps ``max`` to
``xhigh`` on 4.8. The cost-map entries are also what populate
``litellm.anthropic_models`` at import, which is what lets a bare
``claude-opus-5`` name resolve to the ``anthropic`` provider (and match an
``anthropic/*`` wildcard deployment).
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


ALL_OPUS_5_VARIANTS = (
    "claude-opus-5",
    "anthropic.claude-opus-5",
    "global.anthropic.claude-opus-5",
    "us.anthropic.claude-opus-5",
    "eu.anthropic.claude-opus-5",
    "au.anthropic.claude-opus-5",
    "jp.anthropic.claude-opus-5",
    "vertex_ai/claude-opus-5",
    "vertex_ai/claude-opus-5@default",
    "azure_ai/claude-opus-5",
)

BEDROCK_OPUS_5_VARIANTS = (
    "anthropic.claude-opus-5",
    "global.anthropic.claude-opus-5",
    "us.anthropic.claude-opus-5",
    "eu.anthropic.claude-opus-5",
    "au.anthropic.claude-opus-5",
    "jp.anthropic.claude-opus-5",
)


@pytest.mark.parametrize("model_name", BEDROCK_OPUS_5_VARIANTS)
def test_opus_5_bedrock_rejects_strict_tools(model_name, local_model_cost_map):
    """Bedrock Converse routes Opus through a validator that rejects
    ``toolSpec.strict`` (``tools.0.custom.strict: Extra inputs are not
    permitted``), same as Opus 4.7/4.8; verified against Bedrock on 2026-07-24.
    Without the flag LiteLLM forwards ``strict`` and every tool call 400s."""
    from litellm.llms.bedrock.common_utils import bedrock_converse_supports_strict_tools

    assert bedrock_converse_supports_strict_tools(model_name) is False


def test_opus_5_registered_for_bedrock_converse():
    assert "anthropic.claude-opus-5" in BEDROCK_CONVERSE_MODELS


def test_opus_5_5_present_in_bundled_backup():
    backup = GetModelCostMap.load_local_model_cost_map()
    root = _load_root_cost_map()
    assert "claude-opus-5-5" in backup
    assert "claude-opus-5-5" in root
    assert backup["claude-opus-5-5"] == root["claude-opus-5-5"]


@pytest.mark.parametrize("model", ["claude-opus-5-5", "anthropic/claude-opus-5-5"])
def test_opus_5_5_thinking_profile(local_model_cost_map, model):
    """Opus 5.5 has thinking always on with the adaptive thinking surface, and
    no forced tool use, same as Fable 5.1."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    assert AnthropicModelInfo._is_adaptive_thinking_model(model, "anthropic") is True
    assert AnthropicModelInfo._is_always_on_thinking_model(model, "anthropic") is True
    assert AnthropicModelInfo.forced_tool_use_unsupported(model.removeprefix("anthropic/")) is True


BEDROCK_OPUS_5_5_VARIANTS = tuple(m.replace("claude-opus-5", "claude-opus-5-5") for m in BEDROCK_OPUS_5_VARIANTS)


def test_opus_5_5_bedrock_present_in_bundled_backup():
    backup = GetModelCostMap.load_local_model_cost_map()
    root = _load_root_cost_map()
    for model in BEDROCK_OPUS_5_5_VARIANTS:
        assert backup[model] == root[model]


def test_opus_5_5_registered_for_bedrock_converse():
    assert "anthropic.claude-opus-5-5" in BEDROCK_CONVERSE_MODELS


@pytest.mark.parametrize("model_name", BEDROCK_OPUS_5_5_VARIANTS)
def test_opus_5_5_bedrock_pricing(model_name, local_model_cost_map):
    import litellm

    info = litellm.get_model_info(model_name, custom_llm_provider="bedrock")
    multiplier = 1.0 if model_name.split(".")[0] in ("anthropic", "global") else 1.1
    assert info["input_cost_per_token"] == pytest.approx(4e-06 * multiplier)
    assert info["output_cost_per_token"] == pytest.approx(2e-05 * multiplier)
    assert info["cache_creation_input_token_cost"] == pytest.approx(5e-06 * multiplier)
    assert info["cache_read_input_token_cost"] == pytest.approx(2e-07 * multiplier)
    assert info["max_input_tokens"] == 1_000_000
    assert info["max_output_tokens"] == 128_000


@pytest.mark.parametrize("model_name", BEDROCK_OPUS_5_5_VARIANTS)
def test_opus_5_5_bedrock_thinking_profile(model_name, local_model_cost_map):
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    from litellm.llms.bedrock.common_utils import bedrock_converse_supports_strict_tools

    assert AnthropicModelInfo._is_adaptive_thinking_model(model_name, "bedrock") is True
    assert AnthropicModelInfo._is_always_on_thinking_model(model_name, "bedrock") is True
    assert AnthropicModelInfo.forced_tool_use_unsupported(model_name) is True
    assert bedrock_converse_supports_strict_tools(model_name) is False
