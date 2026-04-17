from unittest.mock import AsyncMock, MagicMock, patch

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
    assert "_advisor_interception_converted_stream" not in result
    assert "call-1" in logger._converted_stream_call_ids


@pytest.mark.asyncio
async def test_pre_request_hook_anthropic_converts_standard_tool_to_native():
    logger = AdvisorInterceptionLogger(default_advisor_model="claude-opus-4-6")
    kwargs = {
        "tools": [get_litellm_advisor_tool_openai()],
        "litellm_params": {"custom_llm_provider": "anthropic"},
    }

    with patch(
        "litellm.integrations.advisor_interception.handler.supports_native_advisor_tool",
        return_value=True,
    ):
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
    initial_response.usage = {  # type: ignore[attr-defined]
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }

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
    advisor_subcall_response.usage = {  # type: ignore[attr-defined]
        "prompt_tokens": 200,
        "completion_tokens": 150,
        "total_tokens": 350,
    }

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
    final_response.usage = {  # type: ignore[attr-defined]
        "prompt_tokens": 300,
        "completion_tokens": 40,
        "total_tokens": 340,
    }

    calls = {"count": 0}

    async def mock_acompletion(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return advisor_subcall_response
        if calls["count"] == 2:
            return final_response
        raise AssertionError("Unexpected extra acompletion call")

    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    fake_logging_obj = MagicMock()
    fake_logging_obj.set_cost_breakdown = MagicMock()

    response = await logger.async_run_chat_completion_agentic_loop(
        tools={"advisor_config": {"advisor_model": "claude-opus-4-6", "max_uses": 3}},
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test"}],
        response=initial_response,
        optional_params={"tools": [get_litellm_advisor_tool_openai()], "max_tokens": 256},
        logging_obj=fake_logging_obj,
        stream=False,
        kwargs={"litellm_call_id": "cost-loop-1", "custom_llm_provider": "openai"},
    )

    assert calls["count"] == 2
    assert response is final_response
    assert response._hidden_params["response_cost"] == pytest.approx(2.0)

    fake_logging_obj.set_cost_breakdown.assert_called_once()
    breakdown_kwargs = fake_logging_obj.set_cost_breakdown.call_args.kwargs
    assert breakdown_kwargs["total_cost"] == pytest.approx(2.0)
    assert breakdown_kwargs["input_cost"] == pytest.approx(0.7)
    assert breakdown_kwargs["additional_costs"] == {
        "Main Model (initial)": pytest.approx(1.0),
        "Advisor Model": pytest.approx(0.3),
    }

    message = response.choices[0].message
    advisor_iterations = message.provider_specific_fields["advisor_iterations"]
    assert len(advisor_iterations) == 3
    assert advisor_iterations[0]["type"] == "message"
    assert advisor_iterations[0]["input_tokens"] == 100
    assert advisor_iterations[0]["output_tokens"] == 20
    assert advisor_iterations[1]["type"] == "advisor_message"
    assert advisor_iterations[1]["model"] == "claude-opus-4-6"
    assert advisor_iterations[1]["input_tokens"] == 200
    assert advisor_iterations[1]["output_tokens"] == 150
    assert advisor_iterations[2]["type"] == "message"
    assert advisor_iterations[2]["input_tokens"] == 300
    assert advisor_iterations[2]["output_tokens"] == 40


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


@pytest.mark.asyncio
async def test_post_call_hook_wraps_response_as_stream_when_converted():
    """
    When the pre-call hook converted stream=True to stream=False, the
    post-call hook must wrap the ModelResponse in a MockResponseIterator
    so the proxy can async-iterate over it.
    """
    from litellm.llms.base_llm.base_model_iterator import MockResponseIterator

    logger = AdvisorInterceptionLogger(enabled_providers=["openai"])
    logger._converted_stream_call_ids.add("stream-call-1")

    mock_response = ModelResponse(
        id="test-stream",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="No advisor called."),
            )
        ],
        model="gpt-4o-mini",
        object="chat.completion",
        created=123,
    )

    request_data = {
        "litellm_call_id": "stream-call-1",
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

    assert result is not None
    assert isinstance(result, MockResponseIterator)
    assert "stream-call-1" not in logger._converted_stream_call_ids

    chunks = []
    async for chunk in result:
        chunks.append(chunk)
    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_post_call_hook_returns_none_when_stream_not_converted():
    """
    When stream was not converted (call_id not in _converted_stream_call_ids),
    the post-call hook should return None as before.
    """
    logger = AdvisorInterceptionLogger(enabled_providers=["openai"])

    mock_response = ModelResponse(
        id="test-nostream",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="No advisor called."),
            )
        ],
        model="gpt-4o-mini",
        object="chat.completion",
        created=123,
    )

    request_data = {
        "litellm_call_id": "nostream-call-1",
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


@pytest.mark.asyncio
async def test_should_run_chat_completion_agentic_loop_skips_mixed_tool_calls():
    logger = AdvisorInterceptionLogger(enabled_providers=["openai"])
    logger._advisor_config_by_call_id["mixed-call-1"] = {
        "advisor_model": "claude-opus-4-6",
        "max_uses": 3,
    }
    mock_response = ModelResponse(
        id="test-mixed",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_advisor",
                            type="function",
                            function=Function(
                                name=LITELLM_ADVISOR_TOOL_NAME,
                                arguments='{"question":"Need advisor guidance"}',
                            ),
                        ),
                        ChatCompletionMessageToolCall(
                            id="call_weather",
                            type="function",
                            function=Function(
                                name="get_weather",
                                arguments='{"city":"San Francisco"}',
                            ),
                        ),
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
        kwargs={"litellm_call_id": "mixed-call-1"},
    )

    assert should_run is False
    assert tools_dict == {}
    assert "mixed-call-1" not in logger._advisor_config_by_call_id


def test_prepare_followup_kwargs_removes_litellm_call_id():
    kwargs = {
        "litellm_call_id": "original-call-id",
        "metadata": {"k": "v"},
        "user_defined_key": "should_remain",
    }

    filtered_kwargs = AdvisorInterceptionLogger._prepare_followup_kwargs(kwargs)

    assert "litellm_call_id" not in filtered_kwargs
    assert "metadata" not in filtered_kwargs
    assert filtered_kwargs["user_defined_key"] == "should_remain"


def test_from_config_yaml_with_all_params():
    config = {
        "default_advisor_model": "my-advisor",
        "enabled_providers": ["openai", "vertex_ai"],
    }
    logger = AdvisorInterceptionLogger.from_config_yaml(config)
    assert logger.default_advisor_model == "my-advisor"
    assert logger.enabled_providers is not None
    assert "openai" in logger.enabled_providers
    assert "vertex_ai" in logger.enabled_providers


def test_from_config_yaml_empty_config():
    logger = AdvisorInterceptionLogger.from_config_yaml({})
    assert logger.default_advisor_model is None
    assert logger.enabled_providers is None


def test_initialize_from_proxy_config_reads_litellm_settings():
    litellm_settings = {
        "advisor_interception_params": {
            "default_advisor_model": "proxy-advisor",
            "enabled_providers": ["bedrock"],
        }
    }
    logger = AdvisorInterceptionLogger.initialize_from_proxy_config(
        litellm_settings=litellm_settings,
        callback_specific_params={},
    )
    assert logger.default_advisor_model == "proxy-advisor"
    assert logger.enabled_providers is not None
    assert "bedrock" in logger.enabled_providers


def test_initialize_from_proxy_config_reads_callback_specific_params():
    callback_specific_params = {
        "advisor_interception": {
            "default_advisor_model": "callback-advisor",
        }
    }
    logger = AdvisorInterceptionLogger.initialize_from_proxy_config(
        litellm_settings={},
        callback_specific_params=callback_specific_params,
    )
    assert logger.default_advisor_model == "callback-advisor"


def test_initialize_from_proxy_config_no_params():
    logger = AdvisorInterceptionLogger.initialize_from_proxy_config(
        litellm_settings={},
        callback_specific_params={},
    )
    assert logger.default_advisor_model is None
    assert logger.enabled_providers is None


def test_default_advisor_model_is_none_by_default():
    logger = AdvisorInterceptionLogger()
    assert logger.default_advisor_model is None


def test_is_native_anthropic_advisor_model_delegates_to_model_capability(monkeypatch):
    monkeypatch.setattr(
        "litellm.integrations.advisor_interception.handler.resolve_proxy_model_alias_to_litellm_model",
        lambda model: "anthropic/new-native-advisor-model"
        if model == "proxy_alias"
        else "",
    )
    monkeypatch.setattr(
        "litellm.integrations.advisor_interception.handler.supports_native_advisor_tool",
        lambda model, custom_llm_provider=None: (
            custom_llm_provider == "anthropic"
            and model == "anthropic/new-native-advisor-model"
        ),
    )

    assert (
        AdvisorInterceptionLogger._is_native_anthropic_advisor_model("proxy_alias")
        is True
    )
    assert (
        AdvisorInterceptionLogger._is_native_anthropic_advisor_model(
            "anthropic/claude-opus-4-6"
        )
        is False
    )


def test_convert_tools_raises_when_no_advisor_model():
    logger = AdvisorInterceptionLogger()
    kwargs = {
        "tools": [get_litellm_advisor_tool_openai()],
        "litellm_call_id": "test-no-model",
    }
    with pytest.raises(ValueError, match="No advisor model configured"):
        logger._convert_tools_for_provider(kwargs=kwargs, custom_llm_provider="openai")


@pytest.mark.asyncio
async def test_run_agentic_loop_raises_when_no_advisor_model():
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
                                arguments='{"question":"test"}',
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

    with pytest.raises(ValueError, match="No advisor model configured"):
        await logger.async_run_chat_completion_agentic_loop(
            tools={"advisor_config": {}},
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Test"}],
            response=initial_response,
            optional_params={"tools": [get_litellm_advisor_tool_openai()], "max_tokens": 256},
            logging_obj=None,
            stream=False,
            kwargs={"litellm_call_id": "no-model-test", "custom_llm_provider": "openai"},
        )


@pytest.mark.asyncio
async def test_async_log_failure_event_cleans_up_call_tracking_sets():
    logger = AdvisorInterceptionLogger(enabled_providers=["openai"])
    logger._advisor_config_by_call_id["failed-call"] = {
        "advisor_model": "claude-opus-4-6",
        "max_uses": 2,
    }
    logger._converted_stream_call_ids.add("failed-call")
    logger._skip_post_hook_call_ids.add("failed-call")

    await logger.async_log_failure_event(
        kwargs={"litellm_call_id": "failed-call"},
        response_obj=Exception("boom"),
        start_time=None,
        end_time=None,
    )

    assert "failed-call" not in logger._advisor_config_by_call_id
    assert "failed-call" not in logger._converted_stream_call_ids
    assert "failed-call" not in logger._skip_post_hook_call_ids


@pytest.mark.asyncio
async def test_agentic_loop_uses_router_when_available(monkeypatch):
    logger = AdvisorInterceptionLogger(
        enabled_providers=["openai"],
        default_advisor_model="my-advisor-deployment",
    )
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
                            id="call_router",
                            type="function",
                            function=Function(
                                name=LITELLM_ADVISOR_TOOL_NAME,
                                arguments='{"question":"route me"}',
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
    initial_response._hidden_params["response_cost"] = 0.5

    advisor_resp = ModelResponse(
        id="advisor",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="Advice from router."),
            )
        ],
        model="my-advisor-deployment",
        object="chat.completion",
        created=124,
    )
    advisor_resp._hidden_params["response_cost"] = 0.2

    final_resp = ModelResponse(
        id="final",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="Done."),
            )
        ],
        model="gpt-4o-mini",
        object="chat.completion",
        created=125,
    )
    final_resp._hidden_params["response_cost"] = 0.3

    mock_router = MagicMock()
    router_calls = {"count": 0}

    async def mock_router_acompletion(*args, **kwargs):
        router_calls["count"] += 1
        if router_calls["count"] == 1:
            assert kwargs.get("model") == "my-advisor-deployment"
            return advisor_resp
        if router_calls["count"] == 2:
            return final_resp
        raise AssertionError("Unexpected extra router call")

    mock_router.acompletion = mock_router_acompletion

    monkeypatch.setattr(
        AdvisorInterceptionLogger, "_get_llm_router", staticmethod(lambda: mock_router)
    )

    response = await logger.async_run_chat_completion_agentic_loop(
        tools={
            "advisor_config": {
                "advisor_model": "my-advisor-deployment",
                "max_uses": 3,
            }
        },
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Test"}],
        response=initial_response,
        optional_params={"tools": [get_litellm_advisor_tool_openai()], "max_tokens": 256},
        logging_obj=None,
        stream=False,
        kwargs={"litellm_call_id": "router-test", "custom_llm_provider": "openai"},
    )

    assert router_calls["count"] == 2
    assert response is final_resp
    assert response._hidden_params["response_cost"] == pytest.approx(1.0)
