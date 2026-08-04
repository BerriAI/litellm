from typing import cast

import pytest
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams


def _function_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


def test_drop_unsupported_tools_keeps_function_tools_only():
    optional_params = {
        "tools": [
            _function_tool("shell"),
            {"type": "namespace", "name": "container.exec"},
            _function_tool("apply_patch"),
        ],
        "tool_choice": "auto",
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert [tool["function"]["name"] for tool in result["tools"]] == [
        "shell",
        "apply_patch",
    ]
    assert all(tool["type"] == "function" for tool in result["tools"])
    assert result["tool_choice"] == "auto"


def test_drop_unsupported_tools_drops_dangling_tool_choice_when_none_survive():
    optional_params = {
        "tools": [{"type": "namespace", "name": "container.exec"}],
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "temperature": 0.2,
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert "tools" not in result
    assert "tool_choice" not in result
    assert "parallel_tool_calls" not in result
    assert result["temperature"] == 0.2


def test_drop_unsupported_tools_is_noop_for_function_only():
    optional_params = {
        "tools": [_function_tool("shell")],
        "tool_choice": "auto",
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert result is optional_params


def test_drop_unsupported_tools_is_noop_without_tools():
    optional_params = {"temperature": 0.7}

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert result is optional_params


def test_transform_request_strips_unsupported_tools_from_body():
    config = DeepSeekChatConfig()
    body = config.transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                _function_tool("shell"),
                {"type": "namespace", "name": "container.exec"},
            ],
            "tool_choice": "auto",
        },
        litellm_params={},
        headers={},
    )

    assert [tool["type"] for tool in body["tools"]] == ["function"]
    assert body["tools"][0]["function"]["name"] == "shell"


async def test_async_transform_request_strips_unsupported_tools_from_body():
    config = DeepSeekChatConfig()
    body = await config.async_transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                _function_tool("shell"),
                {"type": "namespace", "name": "container.exec"},
            ],
            "tool_choice": "auto",
        },
        litellm_params={},
        headers={},
    )

    assert [tool["type"] for tool in body["tools"]] == ["function"]
    assert body["tools"][0]["function"]["name"] == "shell"


@pytest.mark.parametrize("reasoning_effort", ["low", "max", "future-tier"])
@pytest.mark.parametrize("drop_params", [False, True])
def test_reasoning_effort_passes_through_unchanged(reasoning_effort, drop_params):
    result = DeepSeekChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": reasoning_effort},
        optional_params={},
        model="deepseek-v4-flash",
        drop_params=drop_params,
    )

    assert result["reasoning_effort"] == reasoning_effort
    assert "thinking" not in result


def test_reasoning_effort_none_disables_thinking():
    result = DeepSeekChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": "none"},
        optional_params={},
        model="deepseek-v4-flash",
        drop_params=False,
    )

    assert result["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in result


def test_explicit_thinking_and_non_none_effort_are_preserved():
    result = DeepSeekChatConfig().map_openai_params(
        non_default_params={
            "thinking": {"type": "disabled"},
            "reasoning_effort": "future-tier",
        },
        optional_params={},
        model="deepseek-v4-flash",
        drop_params=False,
    )

    assert result["thinking"] == {"type": "disabled"}
    assert result["reasoning_effort"] == "future-tier"


def test_reasoning_effort_none_overrides_enabled_thinking():
    result = DeepSeekChatConfig().map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled"},
            "reasoning_effort": "none",
        },
        optional_params={},
        model="deepseek-v4-flash",
        drop_params=False,
    )

    assert result["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in result


@pytest.mark.parametrize(
    ("reasoning_effort", "expected"),
    [
        ("none", {"thinking": {"type": "disabled"}}),
        ("low", {"reasoning_effort": "low"}),
        ("max", {"reasoning_effort": "max"}),
        ("future-tier", {"reasoning_effort": "future-tier"}),
    ],
)
def test_responses_bridge_inherits_deepseek_effort_mapping(reasoning_effort, expected):
    responses_request = cast(
        ResponsesAPIOptionalRequestParams,
        {"reasoning": {"effort": reasoning_effort}},
    )
    chat_request = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="deepseek-v4-flash",
        input="Solve 1 + 1.",
        responses_api_request=responses_request,
        custom_llm_provider="deepseek",
    )

    result = DeepSeekChatConfig().map_openai_params(
        non_default_params=chat_request,
        optional_params={},
        model="deepseek-v4-flash",
        drop_params=False,
    )

    assert {key: result[key] for key in expected} == expected
    if reasoning_effort == "none":
        assert "reasoning_effort" not in result
    else:
        assert "thinking" not in result


@pytest.mark.parametrize(
    ("reasoning_effort", "expected"),
    [
        ("none", {"thinking": {"type": "disabled"}}),
        ("max", {"reasoning_effort": "max"}),
    ],
)
def test_responses_bridge_extracts_effort_from_reasoning_with_summary(reasoning_effort, expected):
    reasoning = {"effort": reasoning_effort, "summary": "detailed"}
    responses_request = cast(
        ResponsesAPIOptionalRequestParams,
        {"reasoning": reasoning},
    )
    chat_request = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="deepseek-v4-flash",
        input="Solve 1 + 1.",
        responses_api_request=responses_request,
        custom_llm_provider="deepseek",
    )

    assert chat_request["reasoning_effort"] == reasoning

    result = DeepSeekChatConfig().map_openai_params(
        non_default_params=chat_request,
        optional_params={},
        model="deepseek-v4-flash",
        drop_params=False,
    )

    assert {key: result[key] for key in expected} == expected
    if reasoning_effort == "none":
        assert "reasoning_effort" not in result
    else:
        assert "thinking" not in result
