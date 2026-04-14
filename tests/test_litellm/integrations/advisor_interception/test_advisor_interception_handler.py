import pytest

from litellm.integrations.advisor_interception.handler import AdvisorInterceptionLogger
from litellm.integrations.advisor_interception.tools import (
    LITELLM_ADVISOR_TOOL_NAME,
    get_litellm_advisor_tool_openai,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
)


@pytest.mark.asyncio
async def test_pre_request_hook_non_native_converts_advisor_tool():
    logger = AdvisorInterceptionLogger(enabled_providers=["openai"])
    kwargs = {
        "tools": [
            {
                "type": "advisor_20260301",
                "name": "advisor",
                "model": "claude-opus-4-6",
                "max_uses": 2,
            }
        ],
        "stream": True,
        "litellm_call_id": "call-1",
        "litellm_params": {"custom_llm_provider": "openai"},
    }

    result = await logger.async_pre_request_hook(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Help"}],
        kwargs=kwargs,
    )
    assert result is not None
    tool = result["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == LITELLM_ADVISOR_TOOL_NAME
    assert logger._advisor_config_by_call_id["call-1"]["advisor_model"] == "claude-opus-4-6"
    assert logger._advisor_config_by_call_id["call-1"]["max_uses"] == 2
    assert result["stream"] is False
    assert result["_advisor_interception_converted_stream"] is True


@pytest.mark.asyncio
async def test_pre_request_hook_anthropic_converts_standard_tool_to_native():
    logger = AdvisorInterceptionLogger(default_advisor_model="claude-opus-4-6")
    kwargs = {
        "tools": [get_litellm_advisor_tool_openai()],
        "litellm_params": {"custom_llm_provider": "anthropic"},
    }

    result = await logger.async_pre_request_hook(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Help"}],
        kwargs=kwargs,
    )
    assert result is not None
    tool = result["tools"][0]
    assert tool["type"] == "advisor_20260301"
    assert tool["name"] == "advisor"
    assert tool["model"] == "claude-opus-4-6"


@pytest.mark.asyncio
async def test_should_run_chat_completion_agentic_loop_detects_advisor_tool_call():
    logger = AdvisorInterceptionLogger(enabled_providers=["openai"])
    mock_response = ModelResponse(
        id="test",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=Function(
                                name=LITELLM_ADVISOR_TOOL_NAME,
                                arguments='{"question":"What should I do?"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        model="gpt-4o-mini",
        object="chat.completion",
        created=123,
    )

    should_run, tools_dict = await logger.async_should_run_chat_completion_agentic_loop(
        response=mock_response,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Help"}],
        tools=[get_litellm_advisor_tool_openai()],
        stream=False,
        custom_llm_provider="openai",
        kwargs={
            "litellm_call_id": "call-2",
        },
    )
    assert should_run is True
    assert len(tools_dict["advisor_calls"]) == 1
    assert tools_dict["advisor_calls"][0]["question"] == "What should I do?"


@pytest.mark.asyncio
async def test_should_run_chat_completion_agentic_loop_detects_legacy_function_call():
    logger = AdvisorInterceptionLogger(enabled_providers=["gemini"])
    mock_response = ModelResponse(
        id="test",
        choices=[
            Choices(
                finish_reason="function_call",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    function_call={
                        "name": LITELLM_ADVISOR_TOOL_NAME,
                        "arguments": '{"question":"Can you confirm advisor path?"}',
                    },
                ),
            )
        ],
        model="gemini/gemini-2.5-flash",
        object="chat.completion",
        created=123,
    )

    should_run, tools_dict = await logger.async_should_run_chat_completion_agentic_loop(
        response=mock_response,
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": "Help"}],
        tools=[get_litellm_advisor_tool_openai()],
        stream=False,
        custom_llm_provider="gemini",
        kwargs={"litellm_call_id": "call-3"},
    )
    assert should_run is True
    assert len(tools_dict["advisor_calls"]) == 1
    assert tools_dict["advisor_calls"][0]["question"] == "Can you confirm advisor path?"
