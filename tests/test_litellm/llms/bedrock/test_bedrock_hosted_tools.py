"""
Tests for Anthropic hosted-tool support on AWS Bedrock Converse.

Hosted tools (memory_20250818, web_search_*, web_fetch_*, code_execution_*)
must be forwarded via `additionalModelRequestFields.tools` rather than being
mapped into Converse `toolConfig.tools`, because the Converse `toolSpec`
schema does not represent them and would otherwise emit a malformed tool
named `litellm_unnamed_tool_X`.
"""

from litellm.llms.bedrock.chat.converse_transformation import (
    BEDROCK_ANTHROPIC_HOSTED_TOOL_PREFIXES,
    BEDROCK_HOSTED_TOOL_BETA_HEADERS,
    AmazonConverseConfig,
    _is_anthropic_hosted_tool,
)


CLAUDE_MODEL = "anthropic.claude-sonnet-4-6-20250929-v1:0"


class TestIsAnthropicHostedTool:
    """Unit tests for the hosted-tool predicate."""

    def test_memory_tool_type_recognised(self):
        assert _is_anthropic_hosted_tool(
            {"type": "memory_20250818", "name": "memory"}
        )

    def test_web_search_tool_type_recognised(self):
        assert _is_anthropic_hosted_tool(
            {"type": "web_search_20250305", "name": "web_search"}
        )

    def test_function_tool_not_hosted(self):
        assert not _is_anthropic_hosted_tool(
            {"type": "function", "function": {"name": "foo", "parameters": {}}}
        )

    def test_missing_type_not_hosted(self):
        assert not _is_anthropic_hosted_tool({"name": "memory"})

    def test_non_dict_not_hosted(self):
        assert not _is_anthropic_hosted_tool("memory_20250818")  # type: ignore[arg-type]


class TestMemoryToolForwardedViaAdditionalModelRequestFields:
    """The memory tool must end up in additionalModelRequestFields, not toolConfig."""

    def test_memory_tool_not_in_tool_config(self):
        config = AmazonConverseConfig()
        tools = [{"type": "memory_20250818", "name": "memory"}]

        result = config._transform_request_helper(
            model=CLAUDE_MODEL,
            system_content_blocks=[],
            optional_params={"tools": tools},
            messages=[{"role": "user", "content": "Hi"}],
            headers={},
        )

        # toolConfig should not contain a phantom litellm_unnamed_tool entry
        tool_config = result.get("toolConfig")
        if tool_config is not None:
            for tool_block in tool_config.get("tools", []):
                spec = tool_block.get("toolSpec", {})
                assert not spec.get("name", "").startswith(
                    "litellm_unnamed_tool"
                ), "memory tool should not be transformed into a generic toolSpec"

    def test_memory_tool_in_additional_model_request_fields(self):
        config = AmazonConverseConfig()
        tools = [{"type": "memory_20250818", "name": "memory"}]

        result = config._transform_request_helper(
            model=CLAUDE_MODEL,
            system_content_blocks=[],
            optional_params={"tools": tools},
            messages=[{"role": "user", "content": "Hi"}],
            headers={},
        )

        additional = result["additionalModelRequestFields"]
        assert "tools" in additional
        assert {"type": "memory_20250818", "name": "memory"} in additional["tools"]

    def test_memory_tool_beta_header_injected(self):
        config = AmazonConverseConfig()
        tools = [{"type": "memory_20250818", "name": "memory"}]

        result = config._transform_request_helper(
            model=CLAUDE_MODEL,
            system_content_blocks=[],
            optional_params={"tools": tools},
            messages=[{"role": "user", "content": "Hi"}],
            headers={},
        )

        additional = result["additionalModelRequestFields"]
        assert "anthropic_beta" in additional
        assert "context-management-2025-06-27" in additional["anthropic_beta"]


class TestFunctionToolNoRegression:
    """Plain function tools must still be transformed into toolConfig.tools."""

    def test_function_tool_emits_tool_spec(self):
        config = AmazonConverseConfig()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]

        result = config._transform_request_helper(
            model=CLAUDE_MODEL,
            system_content_blocks=[],
            optional_params={"tools": tools},
            messages=[{"role": "user", "content": "Hi"}],
            headers={},
        )

        tool_config = result.get("toolConfig")
        assert tool_config is not None
        names = [
            block["toolSpec"]["name"]
            for block in tool_config["tools"]
            if "toolSpec" in block
        ]
        assert "get_weather" in names
        # additionalModelRequestFields should not contain a tools array
        # for plain function tools
        assert "tools" not in result["additionalModelRequestFields"]


class TestMixedTools:
    """Memory + function tool together: both routed correctly, no cross-talk."""

    def test_memory_and_function_tool_split(self):
        config = AmazonConverseConfig()
        tools = [
            {"type": "memory_20250818", "name": "memory"},
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            },
        ]

        result = config._transform_request_helper(
            model=CLAUDE_MODEL,
            system_content_blocks=[],
            optional_params={"tools": tools},
            messages=[{"role": "user", "content": "Hi"}],
            headers={},
        )

        # function tool -> toolConfig
        tool_config = result["toolConfig"]
        spec_names = [
            block["toolSpec"]["name"]
            for block in tool_config["tools"]
            if "toolSpec" in block
        ]
        assert "get_weather" in spec_names
        assert not any(n.startswith("litellm_unnamed_tool") for n in spec_names)

        # memory tool -> additionalModelRequestFields
        additional = result["additionalModelRequestFields"]
        assert any(
            t.get("type") == "memory_20250818" for t in additional["tools"]
        )
        # memory beta header propagated
        assert "context-management-2025-06-27" in additional["anthropic_beta"]


class TestConstantsExposed:
    """Sanity check that the new public constants are wired up correctly."""

    def test_memory_prefix_in_hosted_tool_list(self):
        assert "memory" in BEDROCK_ANTHROPIC_HOSTED_TOOL_PREFIXES

    def test_memory_beta_header_mapping(self):
        assert (
            BEDROCK_HOSTED_TOOL_BETA_HEADERS["memory"]
            == "context-management-2025-06-27"
        )
