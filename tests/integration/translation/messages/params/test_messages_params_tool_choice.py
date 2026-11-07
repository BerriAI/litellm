from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.azure import GPT_6_LUNA_TEST_CASE as AZURE_GPT_6_LUNA_TEST_CASE
from integration.translation.messages.bases.bedrock_converse import CLAUDE_SONNET_4_6_TEST_CASE
from integration.translation.messages.bases.openai import GPT_5_4_TEST_CASE as OPENAI_GPT_5_4_TEST_CASE
from integration.translation.runner import assert_translation
from pydantic import JsonValue

LOOKUP_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
ANTHROPIC_TOOLS: Final[list[JsonValue]] = [
    {"name": "lookup_weather", "description": "Look up the weather for a city.", "input_schema": LOOKUP_SCHEMA}
]
RESPONSES_TOOLS: Final[list[JsonValue]] = [
    {
        "type": "function",
        "name": "lookup_weather",
        "strict": False,
        "description": "Look up the weather for a city.",
        "parameters": LOOKUP_SCHEMA,
    }
]
BEDROCK_TOOL_CONFIG: Final[dict[str, JsonValue]] = {
    "tools": [
        {
            "toolSpec": {
                "name": "lookup_weather",
                "description": "Look up the weather for a city.",
                "inputSchema": {"json": LOOKUP_SCHEMA},
            }
        }
    ],
    "toolChoice": {"auto": {}},
}
DISABLE_PARALLEL_TOOL_USE: Final[dict[str, JsonValue]] = {"type": "auto", "disable_parallel_tool_use": True}
SCENARIO: Final = "tool_choice_disable_parallel_tool_use"

OPENAI_GPT_5_4_DISABLE_PARALLEL_TEST_CASE: Final = replace(
    OPENAI_GPT_5_4_TEST_CASE,
    scenario=SCENARIO,
    litellm_request={
        **OPENAI_GPT_5_4_TEST_CASE.litellm_request,
        "tools": ANTHROPIC_TOOLS,
        "tool_choice": DISABLE_PARALLEL_TOOL_USE,
    },
    expected_provider_request={
        **OPENAI_GPT_5_4_TEST_CASE.expected_provider_request,
        "tools": RESPONSES_TOOLS,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    },
)

AZURE_GPT_6_LUNA_DISABLE_PARALLEL_TEST_CASE: Final = replace(
    AZURE_GPT_6_LUNA_TEST_CASE,
    scenario=SCENARIO,
    litellm_request={**AZURE_GPT_6_LUNA_TEST_CASE.litellm_request, "tool_choice": DISABLE_PARALLEL_TOOL_USE},
    expected_provider_request={
        **AZURE_GPT_6_LUNA_TEST_CASE.expected_provider_request,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    },
)

CLAUDE_SONNET_4_6_DISABLE_PARALLEL_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario=SCENARIO,
    litellm_request={
        **CLAUDE_SONNET_4_6_TEST_CASE.litellm_request,
        "tools": ANTHROPIC_TOOLS,
        "tool_choice": DISABLE_PARALLEL_TOOL_USE,
    },
    expected_provider_request={
        **CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request,
        "toolConfig": BEDROCK_TOOL_CONFIG,
        "additionalModelRequestFields": {"tool_choice": {"disable_parallel_tool_use": True}},
    },
)

CASES: Final = (
    OPENAI_GPT_5_4_DISABLE_PARALLEL_TEST_CASE,
    AZURE_GPT_6_LUNA_DISABLE_PARALLEL_TEST_CASE,
    CLAUDE_SONNET_4_6_DISABLE_PARALLEL_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_messages_params_tool_choice(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
