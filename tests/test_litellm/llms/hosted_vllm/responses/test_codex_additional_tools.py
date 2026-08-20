"""Codex sends its code-mode tools as an `additional_tools` input item with an
empty top-level `tools` array. vLLM either 400s on that item or ignores it and
runs the model with no tools, so hosted_vllm hoists it into `tools`.
"""

import pytest

from litellm.llms.hosted_vllm.responses.transformation import (
    HostedVLLMResponsesAPIConfig,
)
from litellm.responses.codex_compat import hoist_codex_additional_tools
from litellm.types.router import GenericLiteLLMParams

EXEC_TOOL = {
    "type": "custom",
    "name": "exec",
    "description": "Run JavaScript to orchestrate tool calls",
    "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.*/"},
}
WAIT_TOOL = {
    "type": "function",
    "name": "wait",
    "description": "Wait for something",
    "parameters": {"type": "object", "properties": {}},
}


def _codex_input():
    """The shape Codex actually sends: tools nested under a namespace."""
    return [
        {
            "type": "additional_tools",
            "role": "developer",
            "tools": [
                {
                    "type": "namespace",
                    "name": "functions",
                    "description": "",
                    "tools": [EXEC_TOOL, WAIT_TOOL],
                }
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "run ls"}],
        },
    ]


def test_hoists_namespaced_tools_and_drops_the_item():
    new_input, tools, changed = hoist_codex_additional_tools(_codex_input(), [])

    assert changed is True
    assert [i["type"] for i in new_input] == ["message"]
    assert [t["name"] for t in tools] == ["exec", "wait"]


def test_custom_tool_is_shimmed_to_a_single_string_arg_function():
    """Engines skip tools whose type != "function", which would drop `exec`."""
    _, tools, _ = hoist_codex_additional_tools(_codex_input(), [])
    exec_tool = next(t for t in tools if t["name"] == "exec")

    assert exec_tool["type"] == "function"
    assert exec_tool["parameters"]["required"] == ["input"]
    assert exec_tool["parameters"]["properties"]["input"]["type"] == "string"
    # the lark grammar cannot survive the shim
    assert "format" not in exec_tool


def test_existing_tools_are_preserved_and_not_duplicated():
    existing = [{"type": "function", "name": "wait", "parameters": {}}]
    _, tools, _ = hoist_codex_additional_tools(_codex_input(), existing)

    assert [t["name"] for t in tools] == ["wait", "exec"]
    assert tools[0] is existing[0], "pre-existing declaration should win"


@pytest.mark.parametrize(
    "input_value",
    ["just a string", [{"type": "message", "role": "user", "content": "hi"}]],
)
def test_noop_without_additional_tools(input_value):
    new_input, tools, changed = hoist_codex_additional_tools(input_value, None)

    assert changed is False
    assert new_input == input_value
    assert tools is None


def test_transform_moves_tools_into_the_request_body():
    config = HostedVLLMResponsesAPIConfig()

    result = config.transform_responses_api_request(
        model="deepseek-v4-flash",
        input=_codex_input(),
        response_api_optional_request_params={"tools": []},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert [t["name"] for t in result["tools"]] == ["exec", "wait"]
    assert all(t["type"] == "function" for t in result["tools"])
    assert all(i["type"] != "additional_tools" for i in result["input"])


def test_custom_tool_description_is_not_truncated():
    """Codex ships ~15KB of code-mode API surface in exec's description."""
    big = "x" * 15000
    input_items = [
        {
            "type": "additional_tools",
            "role": "developer",
            "tools": [{**EXEC_TOOL, "description": big}],
        }
    ]
    _, tools, _ = hoist_codex_additional_tools(input_items, [])

    assert tools[0]["description"] == big


def test_nameless_builtin_tools_are_forwarded():
    """web_search / mcp entries carry no name and must not be dropped."""
    input_items = [
        {
            "type": "additional_tools",
            "role": "developer",
            "tools": [{"type": "web_search"}, {"type": "mcp", "server_label": "x"}],
        }
    ]
    _, tools, _ = hoist_codex_additional_tools(input_items, [])

    assert [t["type"] for t in tools] == ["web_search", "mcp"]


def test_nested_namespaces_are_expanded_recursively():
    """A leftover inner namespace would re-introduce the unsupported type."""
    input_items = [
        {
            "type": "additional_tools",
            "role": "developer",
            "tools": [
                {
                    "type": "namespace",
                    "name": "outer",
                    "tools": [{"type": "namespace", "name": "inner", "tools": [WAIT_TOOL]}],
                }
            ],
        }
    ]
    _, tools, _ = hoist_codex_additional_tools(input_items, [])

    assert [t["name"] for t in tools] == ["wait"]
    assert all(t["type"] != "namespace" for t in tools)


def test_colliding_names_across_namespaces_keep_the_first():
    """Flattening loses namespace qualification; document the consequence."""
    read_a = {"type": "function", "name": "read", "description": "fs read"}
    read_b = {"type": "function", "name": "read", "description": "git read"}
    input_items = [
        {
            "type": "additional_tools",
            "role": "developer",
            "tools": [
                {"type": "namespace", "name": "fs", "tools": [read_a]},
                {"type": "namespace", "name": "git", "tools": [read_b]},
            ],
        }
    ]
    _, tools, _ = hoist_codex_additional_tools(input_items, [])

    assert len(tools) == 1
    assert tools[0]["description"] == "fs read"
