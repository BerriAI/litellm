import json

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.utils import Choices, Message, ModelResponse


def _tool_search_response() -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-tool-search",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_tool_search",
                            "type": "function",
                            "function": {
                                "name": "tool_search",
                                "arguments": json.dumps({"query": "weather"}),
                            },
                        }
                    ],
                ),
            )
        ],
    )


def test_client_tool_search_becomes_chat_function_tool() -> None:
    result, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
        tools=[{"type": "tool_search", "execution": "client"}]
    )

    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "tool_search"


def test_server_tool_search_stays_hosted_and_is_dropped() -> None:
    result, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
        tools=[{"type": "tool_search", "execution": "server"}]
    )

    assert result == []


def test_client_tool_search_preserves_custom_parameters() -> None:
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    result, _ = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
        tools=[{"type": "tool_search", "execution": "client", "parameters": parameters}]
    )

    assert result[0]["function"]["parameters"] == parameters


def test_tool_search_call_output_round_trips_to_chat_messages() -> None:
    tools = [{"type": "function", "name": "get_weather"}]
    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        [
            {
                "type": "tool_search_call",
                "id": "ts_1",
                "call_id": "call_tool_search",
                "arguments": {"query": "weather"},
            },
            {"type": "tool_search_output", "call_id": "call_tool_search", "tools": tools},
        ],
        responses_api_request={"tools": [{"type": "tool_search", "execution": "client"}]},
    )

    assert messages[0]["tool_calls"][0]["function"]["name"] == "tool_search"
    assert messages[1] == {
        "role": "tool",
        "content": json.dumps(tools),
        "tool_call_id": "call_tool_search",
    }


def test_tool_search_output_falls_back_to_output_field() -> None:
    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        [{"type": "tool_search_output", "call_id": "call_tool_search", "output": "no tools"}],
        responses_api_request={},
    )

    assert messages[0]["content"] == "no tools"


def test_tool_search_output_without_call_id_is_dropped() -> None:
    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        [{"type": "tool_search_output", "tools": []}],
        responses_api_request={},
    )

    assert messages == []


def test_tool_search_response_becomes_tool_search_call_item() -> None:
    output = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
        _tool_search_response(),
        {"tools": [{"type": "tool_search", "execution": "client"}]},
    )

    assert output == [
        {
            "type": "tool_search_call",
            "id": "call_tool_search",
            "call_id": "call_tool_search",
            "status": "completed",
            "execution": "client",
            "arguments": {"query": "weather"},
        }
    ]
