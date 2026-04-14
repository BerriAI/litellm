import pytest

import litellm
from litellm.integrations.advisor_interception.handler import AdvisorInterceptionLogger
from litellm.integrations.advisor_interception.tools import (
    LITELLM_ADVISOR_TOOL_NAME,
    get_litellm_advisor_tool_openai,
)
from litellm.types.utils import (
    CallTypes,
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


@pytest.mark.asyncio
async def test_run_chat_completion_agentic_loop_aggregates_subcall_costs(monkeypatch):
    logger = AdvisorInterceptionLogger(enabled_providers=["openai"])
    initial_response = ModelResponse(
        id="initial",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_abc",
                            type="function",
                            function=Function(
                                name=LITELLM_ADVISOR_TOOL_NAME,
                                arguments='{"question":"Need advisor guidance"}',
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
    initial_response._hidden_params["response_cost"] = 1.0

    advisor_subcall_response = ModelResponse(
        id="advisor-subcall",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    role="assistant",
                    content="Advisor says this looks good.",
                ),
            )
        ],
        model="claude-opus-4-6",
        object="chat.completion",
        created=124,
    )
    advisor_subcall_response._hidden_params["response_cost"] = 0.3

    final_response = ModelResponse(
        id="final",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    role="assistant",
                    content="integration ok.",
                ),
            )
        ],
        model="gpt-4o-mini",
        object="chat.completion",
        created=125,
    )
    final_response._hidden_params["response_cost"] = 0.7

    calls = {"count": 0}

    async def mock_acompletion(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return advisor_subcall_response
        if calls["count"] == 2:
            return final_response
        raise AssertionError("Unexpected extra acompletion call")

    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    response = await logger.async_run_chat_completion_agentic_loop(
        tools={"advisor_config": {"advisor_model": "claude-opus-4-6", "max_uses": 3}},
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test"}],
        response=initial_response,
        optional_params={"tools": [get_litellm_advisor_tool_openai()], "max_tokens": 256},
        logging_obj=None,
        stream=False,
        kwargs={"litellm_call_id": "cost-loop-1", "custom_llm_provider": "openai"},
    )

    assert calls["count"] == 2
    assert response is final_response
    assert response._hidden_params["response_cost"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_should_run_chat_completion_agentic_loop_cleans_up_config_on_no_tool_call():
    logger = AdvisorInterceptionLogger(enabled_providers=["openai"])
    logger._advisor_config_by_call_id["cleanup-call-1"] = {
        "advisor_model": "claude-opus-4-6",
        "max_uses": 3,
    }

    mock_response = ModelResponse(
        id="test",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="No tool was called."),
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
        kwargs={"litellm_call_id": "cleanup-call-1"},
    )

    assert should_run is False
    assert tools_dict == {}
    assert "cleanup-call-1" not in logger._advisor_config_by_call_id


@pytest.mark.asyncio
async def test_post_call_hook_cleans_up_config_when_should_run_is_false():
    logger = AdvisorInterceptionLogger(enabled_providers=["openai"])
    logger._advisor_config_by_call_id["cleanup-call-2"] = {
        "advisor_model": "claude-opus-4-6",
        "max_uses": 3,
    }

    mock_response = ModelResponse(
        id="test",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="Final answer without advisor."),
            )
        ],
        model="gpt-4o-mini",
        object="chat.completion",
        created=123,
    )

    request_data = {
        "litellm_call_id": "cleanup-call-2",
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Help"}],
        "tools": [get_litellm_advisor_tool_openai()],
        "stream": False,
        "custom_llm_provider": "openai",
    }

    result = await logger.async_post_call_success_deployment_hook(
        request_data=request_data,
        response=mock_response,
        call_type=CallTypes.acompletion,
    )

    assert result is None
    assert "cleanup-call-2" not in logger._advisor_config_by_call_id
