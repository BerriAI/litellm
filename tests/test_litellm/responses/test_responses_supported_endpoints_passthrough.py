"""
A deployment with `model_info.supported_endpoints` containing `/v1/responses` forwards
`/v1/responses` natively to `{api_base}/responses`. Without it, generic OpenAI-compatible
providers such as `custom_openai` keep bridging through `/v1/chat/completions`.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.openai_like.responses.transformation import OpenAILikeResponsesConfig
from litellm.responses.main import _resolve_responses_api_provider_config
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ModelResponse

API_BASE = "https://backend.example/v1"
RESPONSES_URL = f"{API_BASE}/responses"
CHAT_URL = f"{API_BASE}/chat/completions"
OPT_IN = {"supported_endpoints": ["/v1/chat/completions", "/v1/responses"]}

RESPONSES_BODY = {
    "id": "resp_native",
    "object": "response",
    "created_at": 1741476542,
    "status": "completed",
    "model": "my-model",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "native", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}

CHAT_BODY = {
    "id": "chatcmpl_bridged",
    "object": "chat.completion",
    "created": 1741476542,
    "model": "my-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "bridged"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

SSE_BODY = (
    "event: response.created\n"
    f"data: {json.dumps({'type': 'response.created', 'response': RESPONSES_BODY})}\n\n"
    "event: response.completed\n"
    f"data: {json.dumps({'type': 'response.completed', 'response': RESPONSES_BODY})}\n\n"
)


def _mock_backend(router: respx.MockRouter) -> tuple[respx.Route, respx.Route]:
    responses_route = router.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json=RESPONSES_BODY))
    chat_route = router.post(CHAT_URL).mock(return_value=httpx.Response(200, json=CHAT_BODY))
    return responses_route, chat_route


SWAPPED_MODEL = "deepseek/deepseek-chat"
SWAPPED_API_BASE = "https://api.deepseek.com/beta"


def _prompt_manager_swapping_to(model: str) -> MagicMock:
    """A logging object whose prompt hook rewrites the request's model, as a prompt manager does."""
    prompt_return = (model, [{"role": "user", "content": "hi"}], {})
    logging_obj = MagicMock()
    logging_obj.__class__ = LiteLLMLoggingObj
    logging_obj.should_run_prompt_management_hooks.return_value = True
    logging_obj.get_chat_completion_prompt.return_value = prompt_return
    logging_obj.async_get_chat_completion_prompt = AsyncMock(return_value=prompt_return)
    logging_obj.model_call_details = {}
    return logging_obj


def _mock_swap_targets(router: respx.MockRouter, monkeypatch) -> tuple[respx.Route, respx.Route]:
    """The swapped provider's chat endpoint, plus the `/responses` it does not serve but a stale
    opt-in would send to."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    swapped_chat_route = router.post(f"{SWAPPED_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=CHAT_BODY)
    )
    stale_responses_route = router.post(f"{SWAPPED_API_BASE}/responses").mock(
        return_value=httpx.Response(200, json=RESPONSES_BODY)
    )
    return swapped_chat_route, stale_responses_route


@pytest.fixture(autouse=True)
def _respx_interceptable_httpx_client(monkeypatch):
    monkeypatch.setattr(litellm, "num_retries", 0)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.parametrize(
    "model_info, expected_type",
    [
        (OPT_IN, OpenAILikeResponsesConfig),
        ({"supported_endpoints": ["/v1/chat/completions"]}, type(None)),
        ({}, type(None)),
        (None, type(None)),
        ("/v1/responses", type(None)),
    ],
)
def test_resolver_opt_in_gates_openai_like_config(model_info, expected_type):
    config = _resolve_responses_api_provider_config("my-model", "custom_openai", model_info)
    assert type(config) is expected_type


def test_resolver_keeps_native_provider_config():
    """`openai/` already routes /v1/responses natively; the opt-in must not swap its config."""
    config = _resolve_responses_api_provider_config("gpt-4.1", "openai", OPT_IN)
    assert type(config) is OpenAIResponsesAPIConfig


@respx.mock
async def test_opt_in_forwards_responses_natively():
    responses_route, chat_route = _mock_backend(respx.mock)

    result = await litellm.aresponses(
        model="custom_openai/my-model",
        input="hi",
        api_base=API_BASE,
        api_key="sk-backend",
        model_info=OPT_IN,
    )

    assert responses_route.call_count == 1
    assert chat_route.call_count == 0
    request = responses_route.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-backend"
    assert json.loads(request.content)["input"] == "hi"
    assert isinstance(result, ResponsesAPIResponse)
    assert result.output[0].content[0].text == "native"


@respx.mock
async def test_opt_in_forwards_streaming_responses_natively(monkeypatch):
    """The router registers each deployment in `litellm.model_cost`; an unregistered model is
    treated as non-streaming and would be faked, so mirror that registration here."""
    monkeypatch.setitem(litellm.model_cost, "custom_openai/my-model", {"litellm_provider": "custom_openai"})
    responses_route = respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(200, text=SSE_BODY, headers={"content-type": "text/event-stream"})
    )
    chat_route = respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=CHAT_BODY))

    stream = await litellm.aresponses(
        model="custom_openai/my-model",
        input="hi",
        stream=True,
        api_base=API_BASE,
        api_key="sk-backend",
        model_info=OPT_IN,
    )
    events = [event async for event in stream]

    assert responses_route.call_count == 1
    assert chat_route.call_count == 0
    assert json.loads(responses_route.calls.last.request.content)["stream"] is True
    assert [event.type for event in events] == ["response.created", "response.completed"]


@respx.mock
async def test_without_opt_in_still_bridges_through_chat_completions():
    responses_route, chat_route = _mock_backend(respx.mock)

    result = await litellm.aresponses(
        model="custom_openai/my-model",
        input="hi",
        api_base=API_BASE,
        api_key="sk-backend",
        model_info={"supported_endpoints": ["/v1/chat/completions"]},
    )

    assert chat_route.call_count == 1
    assert responses_route.call_count == 0
    assert isinstance(result, ResponsesAPIResponse)
    assert result.output[0].content[0].text == "bridged"


@respx.mock
async def test_prompt_swap_to_other_provider_drops_deployment_opt_in(monkeypatch):
    """When a prompt manager moves the request to another provider, the original deployment's
    `supported_endpoints` no longer describes the upstream, so the swapped provider bridges."""
    swapped_chat_route, stale_responses_route = _mock_swap_targets(respx.mock, monkeypatch)

    result = await litellm.aresponses(
        model="custom_openai/my-model",
        input="hi",
        prompt_id="p1",
        litellm_logging_obj=_prompt_manager_swapping_to(SWAPPED_MODEL),
        model_info=OPT_IN,
    )

    assert swapped_chat_route.call_count == 1
    assert stale_responses_route.call_count == 0
    assert isinstance(result, ResponsesAPIResponse)
    assert result.output[0].content[0].text == "bridged"


@respx.mock
def test_sync_prompt_swap_to_other_provider_drops_deployment_opt_in(monkeypatch):
    swapped_chat_route, stale_responses_route = _mock_swap_targets(respx.mock, monkeypatch)

    result = litellm.responses(
        model="custom_openai/my-model",
        input="hi",
        prompt_id="p1",
        litellm_logging_obj=_prompt_manager_swapping_to(SWAPPED_MODEL),
        model_info=OPT_IN,
    )

    assert swapped_chat_route.call_count == 1
    assert stale_responses_route.call_count == 0
    assert isinstance(result, ResponsesAPIResponse)
    assert result.output[0].content[0].text == "bridged"


@respx.mock
async def test_mode_responses_chat_completion_reaches_native_responses(monkeypatch):
    """A `mode: responses` deployment bridges chat completions into the Responses API; with
    the opt-in that inner call must reach `{api_base}/responses` instead of bouncing back
    to `/chat/completions`."""
    responses_route, chat_route = _mock_backend(respx.mock)
    monkeypatch.setitem(
        litellm.model_cost,
        "custom_openai/my-model",
        {"mode": "responses", "litellm_provider": "custom_openai"},
    )

    result = await litellm.acompletion(
        model="custom_openai/my-model",
        messages=[{"role": "user", "content": "hi"}],
        api_base=API_BASE,
        api_key="sk-backend",
        model_info={"mode": "responses", **OPT_IN},
    )

    assert responses_route.call_count == 1
    assert chat_route.call_count == 0
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "native"
