"""Regression tests for Bedrock Converse ``toolSpec.strict`` forwarding.

Bedrock Converse routes Claude Opus 4.7/4.8 through an Anthropic-compatible
validator that rejects ``toolSpec.strict`` even though Anthropic's native API
accepts ``strict`` as a top-level tool field for the same models. See
BerriAI/litellm#31582.
"""

import pytest

from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt
from litellm.llms.bedrock.common_utils import bedrock_converse_supports_strict_tools

_STRICT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "strict": True,
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius"]},
                },
                "required": ["city", "unit"],
                "additionalProperties": False,
            },
        },
    }
]


@pytest.mark.parametrize(
    "model_id",
    [
        "bedrock/us.anthropic.claude-opus-4-7",
        "bedrock/us.anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-7",
        "anthropic.claude-opus-4-8",
        "anthropic.claude-opus-4-7-v1:0",
        "bedrock/eu.anthropic.claude-opus-4-8-v1:0",
        "bedrock/global.anthropic.claude-opus-4-7",
    ],
)
def test_bedrock_tools_pt_strict_dropped_for_opus_47_48(model_id: str) -> None:
    """Opus 4.7/4.8 on Bedrock Converse reject toolSpec.strict — must be dropped."""
    result = _bedrock_tools_pt(_STRICT_TOOL, model=model_id)
    assert (
        "strict" not in result[0]["toolSpec"]
    ), f"strict leaked into toolSpec for {model_id}: {result[0]['toolSpec']}"


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "bedrock/us.anthropic.claude-sonnet-4-6",
        "bedrock/us.anthropic.claude-opus-4-6",
        "bedrock/us.anthropic.claude-opus-4-5",
    ],
)
def test_bedrock_tools_pt_strict_kept_for_other_anthropic(model_id: str) -> None:
    """Sonnet 4.5/4.6 and Opus <=4.6 accept toolSpec.strict — keep forwarding it."""
    result = _bedrock_tools_pt(_STRICT_TOOL, model=model_id)
    assert (
        result[0]["toolSpec"]["strict"] is True
    ), f"strict missing for {model_id}: {result[0]['toolSpec']}"


@pytest.mark.parametrize(
    "model_id",
    [
        "us.amazon.nova-micro-v1:0",
        "meta.llama3-2-11b-instruct-v1:0",
    ],
)
def test_bedrock_tools_pt_strict_dropped_for_non_anthropic(model_id: str) -> None:
    """Non-Anthropic Bedrock families reject toolSpec.strict — must be dropped."""
    result = _bedrock_tools_pt(_STRICT_TOOL, model=model_id)
    assert "strict" not in result[0]["toolSpec"]


def test_bedrock_converse_supports_strict_tools_helper() -> None:
    """Direct check for the gate helper used by factory.py."""
    assert (
        bedrock_converse_supports_strict_tools("bedrock/us.anthropic.claude-opus-4-7")
        is False
    )
    assert (
        bedrock_converse_supports_strict_tools("bedrock/us.anthropic.claude-opus-4-8")
        is False
    )
    assert (
        bedrock_converse_supports_strict_tools(
            "anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        is True
    )
    assert (
        bedrock_converse_supports_strict_tools("bedrock/us.anthropic.claude-opus-4-6")
        is True
    )
    assert bedrock_converse_supports_strict_tools("us.amazon.nova-micro-v1:0") is False
    assert bedrock_converse_supports_strict_tools("") is False


@pytest.mark.parametrize(
    "cost_map_key",
    [
        "anthropic.claude-opus-4-7",
        "us.anthropic.claude-opus-4-7",
        "anthropic.claude-opus-4-8",
        "us.anthropic.claude-opus-4-8",
    ],
)
def test_strict_tools_flag_set_in_model_cost_map(cost_map_key: str) -> None:
    """The gate is driven by ``bedrock_converse_supports_strict_tools: false`` in
    ``model_prices_and_context_window.json``, not hardcoded model patterns."""
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    cost_map = GetModelCostMap.load_local_model_cost_map()
    assert cost_map[cost_map_key]["bedrock_converse_supports_strict_tools"] is False
