import pytest

from litellm.llms.litellm_proxy.skills.code_execution import (
    LITELLM_CODE_EXECUTION_TOOL,
    CodeExecutionHandler,
    LiteLLMInternalTools,
    get_litellm_code_execution_tool,
    get_litellm_code_execution_tool_anthropic,
)
from litellm.llms.litellm_proxy.skills.constants import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_SANDBOX_TIMEOUT,
)

_DESCRIPTION = (
    "Execute Python code in a sandboxed environment. Use this to run code that "
    "generates files, processes data, or performs computations. Generated files "
    "will be returned directly."
)


class TestInternalToolName:
    def test_code_execution_tool_name_is_stable(self):
        assert LiteLLMInternalTools.CODE_EXECUTION.value == "litellm_code_execution"

    def test_enum_is_str_subclass_so_it_serializes_as_the_bare_name(self):
        assert isinstance(LiteLLMInternalTools.CODE_EXECUTION, str)


class TestOpenAIToolSchema:
    def test_schema_matches_openai_function_tool_contract_exactly(self):
        assert get_litellm_code_execution_tool() == {
            "type": "function",
            "function": {
                "name": "litellm_code_execution",
                "description": _DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                    "required": ["code"],
                },
            },
        }

    def test_returns_a_fresh_dict_each_call_so_callers_cannot_mutate_the_shared_one(self):
        first = get_litellm_code_execution_tool()
        first["function"]["name"] = "clobbered"
        assert get_litellm_code_execution_tool()["function"]["name"] == "litellm_code_execution"

    def test_singleton_matches_the_factory(self):
        assert LITELLM_CODE_EXECUTION_TOOL == get_litellm_code_execution_tool()


class TestAnthropicToolSchema:
    def test_schema_matches_anthropic_messages_tool_contract_exactly(self):
        assert get_litellm_code_execution_tool_anthropic() == {
            "name": "litellm_code_execution",
            "description": _DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                "required": ["code"],
            },
        }

    def test_anthropic_shape_is_flat_and_carries_no_openai_only_keys(self):
        tool = get_litellm_code_execution_tool_anthropic()
        assert "input_schema" in tool
        assert "type" not in tool
        assert "function" not in tool
        assert "parameters" not in tool

    def test_returns_a_fresh_dict_each_call(self):
        get_litellm_code_execution_tool_anthropic()["name"] = "clobbered"
        assert get_litellm_code_execution_tool_anthropic()["name"] == "litellm_code_execution"

    def test_both_surfaces_agree_on_name_and_description(self):
        openai_tool = get_litellm_code_execution_tool()
        anthropic_tool = get_litellm_code_execution_tool_anthropic()
        assert anthropic_tool["name"] == openai_tool["function"]["name"]
        assert anthropic_tool["description"] == openai_tool["function"]["description"]
        assert anthropic_tool["input_schema"] == openai_tool["function"]["parameters"]


class TestHandlerDefaults:
    def test_defaults_come_from_constants_when_nothing_is_passed(self):
        handler = CodeExecutionHandler()
        assert handler.max_iterations == DEFAULT_MAX_ITERATIONS
        assert handler.sandbox_timeout == DEFAULT_SANDBOX_TIMEOUT

    def test_explicit_values_win_over_the_defaults(self):
        handler = CodeExecutionHandler(max_iterations=3, sandbox_timeout=7)
        assert handler.max_iterations == 3
        assert handler.sandbox_timeout == 7

    def test_each_argument_falls_back_independently(self):
        assert CodeExecutionHandler(max_iterations=3).sandbox_timeout == DEFAULT_SANDBOX_TIMEOUT
        assert CodeExecutionHandler(max_iterations=3).max_iterations == 3
        assert CodeExecutionHandler(sandbox_timeout=7).max_iterations == DEFAULT_MAX_ITERATIONS
        assert CodeExecutionHandler(sandbox_timeout=7).sandbox_timeout == 7

    @pytest.mark.parametrize("falsy", [0, None])
    def test_falsy_values_fall_back_to_the_defaults(self, falsy):
        handler = CodeExecutionHandler(max_iterations=falsy, sandbox_timeout=falsy)
        assert handler.max_iterations == DEFAULT_MAX_ITERATIONS
        assert handler.sandbox_timeout == DEFAULT_SANDBOX_TIMEOUT
