"""Regression tests for Bedrock Converse ``toolSpec.strict`` forwarding.

Bedrock Converse routes several Claude model families through a validator that
rejects ``toolSpec.strict`` (and can hang on large strict MCP tool sets via
compiled-grammar limits) even though Anthropic's native API accepts ``strict``.
See BerriAI/litellm#31582 (Opus), #31943 (Sonnet 4), #34388 (Haiku 4.5 / Sonnet 4.6).
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
        # Sonnet 4 also rejects toolSpec.strict on Bedrock Converse
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/apac.anthropic.claude-sonnet-4-20250514-v1:0",
        # Haiku 4.5 / Sonnet 4.6: grammar-size hang / strict rejection (#34388)
        "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock/us.anthropic.claude-sonnet-4-6",
        "anthropic.claude-sonnet-4-6",
        "bedrock/global.anthropic.claude-sonnet-4-6",
        "bedrock/eu.anthropic.claude-sonnet-4-6",
    ],
)
def test_bedrock_tools_pt_strict_dropped_for_strict_unsupported_models(
    model_id: str,
) -> None:
    """Models with bedrock_converse_supports_strict_tools:false drop strict fields."""
    result = _bedrock_tools_pt(_STRICT_TOOL, model=model_id)
    tool_spec = result[0]["toolSpec"]
    assert (
        "strict" not in tool_spec
    ), f"strict leaked into toolSpec for {model_id}: {tool_spec}"
    assert (
        "additionalProperties" not in tool_spec["inputSchema"]["json"]
    ), f"additionalProperties leaked into toolSpec for {model_id}: {tool_spec}"


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "bedrock/us.anthropic.claude-opus-4-6",
        "bedrock/us.anthropic.claude-opus-4-5",
    ],
)
def test_bedrock_tools_pt_strict_kept_for_other_anthropic(model_id: str) -> None:
    """Sonnet 4.5 and Opus <=4.6 still accept toolSpec.strict — keep forwarding it."""
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
    # Sonnet 4 also rejects strict on Bedrock Converse
    assert (
        bedrock_converse_supports_strict_tools(
            "anthropic.claude-sonnet-4-20250514-v1:0"
        )
        is False
    )
    assert (
        bedrock_converse_supports_strict_tools(
            "bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0"
        )
        is False
    )
    # #34388 Haiku 4.5 / Sonnet 4.6
    assert (
        bedrock_converse_supports_strict_tools(
            "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        is False
    )
    assert (
        bedrock_converse_supports_strict_tools("bedrock/us.anthropic.claude-sonnet-4-6")
        is False
    )


@pytest.mark.parametrize(
    "cost_map_key",
    [
        "anthropic.claude-opus-4-7",
        "us.anthropic.claude-opus-4-7",
        "anthropic.claude-opus-4-8",
        "us.anthropic.claude-opus-4-8",
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "global.anthropic.claude-sonnet-4-20250514-v1:0",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "eu.anthropic.claude-sonnet-4-20250514-v1:0",
        "apac.anthropic.claude-sonnet-4-20250514-v1:0",
        # #34388
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-4-6",
        "global.anthropic.claude-sonnet-4-6",
    ],
)
def test_strict_tools_flag_set_in_model_cost_map(cost_map_key: str) -> None:
    """The gate is driven by ``bedrock_converse_supports_strict_tools: false`` in
    ``model_prices_and_context_window.json``, not hardcoded model patterns."""
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    cost_map = GetModelCostMap.load_local_model_cost_map()
    assert cost_map[cost_map_key]["bedrock_converse_supports_strict_tools"] is False
