import json
from collections.abc import Mapping
from functools import partial
from itertools import chain
from typing import Final

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.anthropic.pass_through.messages.handler import anthropic_messages
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.utils import Choices, ModelResponseStream

API_BASE: Final = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
ANTHROPIC_BASE: Final = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/apps/anthropic"
MODEL: Final = "qwen3.8-max"
PREFIXED_MODEL: Final = f"alibaba_token_plan/{MODEL}"


def chat_messages() -> list[AllMessageValues]:
    return [
        {"role": "assistant", "content": "Earlier answer", "reasoning_content": "Earlier reasoning"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image", "cache_control": {"type": "ephemeral"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
            ],
        },
    ]


def chat_tools() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {"name": "describe_image", "parameters": {"type": "object"}},
            "cache_control": {"type": "ephemeral"},
        }
    ]


def chat_transport(request: httpx.Request, model: str) -> httpx.Response:
    assert str(request.url) == f"{API_BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer token-plan-test-key"
    payload: Final = json.loads(request.content)
    assert payload["model"] == model
    assert payload["messages"] == chat_messages()
    assert payload["max_completion_tokens"] == 123
    assert "max_tokens" not in payload
    assert payload["reasoning_effort"] == "low"
    assert payload["enable_thinking"] is True
    assert payload["tool_choice"] == "auto"
    assert payload["tools"] == chat_tools()
    response: Final = {
        "id": "chatcmpl-token-plan",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Inspecting image",
                    "tool_calls": [
                        {
                            "id": "call_image",
                            "type": "function",
                            "function": {"name": "describe_image", "arguments": '{"image":1}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }
    if not payload.get("stream"):
        return httpx.Response(200, json=response)
    chunks: Final = (
        {
            "id": "chatcmpl-token-plan",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "Inspecting image"}}],
        },
        {
            "id": "chatcmpl-token-plan",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_image",
                                "type": "function",
                                "function": {"name": "describe_image", "arguments": '{"image":1}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
    )


def assert_chat_response(response: litellm.ModelResponse) -> None:
    assert isinstance(response.choices[0], Choices)
    assert response.choices[0].finish_reason == "tool_calls"
    message: Final = response.choices[0].message
    assert message.reasoning_content == "Inspecting image"
    assert message.tool_calls is not None
    assert message.tool_calls[0].function.name == "describe_image"
    assert json.loads(message.tool_calls[0].function.arguments) == {"image": 1}
    assert response.usage.total_tokens == 20


def assert_chat_stream(chunks: list[ModelResponseStream]) -> None:
    assert (
        "".join(getattr(chunk.choices[0].delta, "reasoning_content", "") or "" for chunk in chunks)
        == "Inspecting image"
    )
    calls: Final = tuple(chain.from_iterable(chunk.choices[0].delta.tool_calls or [] for chunk in chunks))
    assert calls[0].id == "call_image"
    assert calls[0].function.name == "describe_image"
    assert json.loads("".join(call.function.arguments or "" for call in calls)) == {"image": 1}
    assert any(chunk.choices[0].finish_reason == "tool_calls" for chunk in chunks)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("use_http_handler", [False, True])
@pytest.mark.parametrize("model", [MODEL, "auto"])
def test_chat_wire_contract(stream: bool, use_http_handler: bool, model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "token-plan-test-key")
    monkeypatch.delenv("ALIBABA_TOKEN_PLAN_API_BASE", raising=False)
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", str(use_http_handler).lower())
    with httpx.Client(transport=httpx.MockTransport(partial(chat_transport, model=model))) as http_client:
        client: Final = (
            HTTPHandler(client=http_client)
            if use_http_handler
            else OpenAI(api_key="token-plan-test-key", base_url=API_BASE, http_client=http_client)
        )
        response: Final = litellm.completion(
            model=f"alibaba_token_plan/{model}",
            messages=chat_messages(),
            client=client,
            max_completion_tokens=123,
            reasoning_effort="low",
            extra_body={"enable_thinking": True},
            tools=chat_tools(),
            tool_choice="auto",
            stream=stream,
        )
        if stream:
            assert_chat_stream(list(response))
        else:
            assert isinstance(response, litellm.ModelResponse)
            assert_chat_response(response)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("use_http_handler", [False, True])
@pytest.mark.parametrize("model", [MODEL, "auto"])
async def test_async_chat_wire_contract(
    stream: bool, use_http_handler: bool, model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "token-plan-test-key")
    monkeypatch.delenv("ALIBABA_TOKEN_PLAN_API_BASE", raising=False)
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", str(use_http_handler).lower())
    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(partial(chat_transport, model=model)))
    async with handler.client as http_client:
        client: Final = (
            handler
            if use_http_handler
            else AsyncOpenAI(api_key="token-plan-test-key", base_url=API_BASE, http_client=http_client)
        )
        response: Final = await litellm.acompletion(
            model=f"alibaba_token_plan/{model}",
            messages=chat_messages(),
            client=client,
            max_completion_tokens=123,
            reasoning_effort="low",
            extra_body={"enable_thinking": True},
            tools=chat_tools(),
            tool_choice="auto",
            stream=stream,
        )
        if stream:
            assert_chat_stream([chunk async for chunk in response])
        else:
            assert isinstance(response, litellm.ModelResponse)
            assert_chat_response(response)


@pytest.mark.parametrize("explicit", [False, True])
def test_provider_resolution_and_overrides(explicit: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "env-key")
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_BASE", "https://gateway.example/env/v1")
    assert get_llm_provider(
        model=PREFIXED_MODEL,
        api_base="https://gateway.example/explicit/v1" if explicit else None,
        api_key="explicit-key" if explicit else None,
    ) == (
        MODEL,
        "alibaba_token_plan",
        "explicit-key" if explicit else "env-key",
        "https://gateway.example/explicit/v1" if explicit else "https://gateway.example/env/v1",
    )


@pytest.mark.parametrize("env_key,api_key", [(None, None), ("", None), ("env-key", None), (None, "explicit-key")])
@pytest.mark.parametrize("global_key", [None, "global-key"])
def test_environment_validation_requires_subscription_key(
    env_key: str | None, api_key: str | None, global_key: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "api_key", global_key)
    if env_key is None:
        monkeypatch.delenv("ALIBABA_TOKEN_PLAN_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", env_key)
    has_key: Final = bool(env_key or api_key)
    assert litellm.validate_environment(model=PREFIXED_MODEL, api_key=api_key) == {
        "keys_in_environment": has_key,
        "missing_keys": [] if has_key else ["ALIBABA_TOKEN_PLAN_API_KEY"],
    }


def test_subscription_key_discovers_provider_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "token-plan-test-key")
    catalog: Final = frozenset(
        model for model, info in litellm.model_cost.items() if info.get("litellm_provider") == "alibaba_token_plan"
    )
    assert PREFIXED_MODEL in catalog
    assert (
        frozenset(litellm.get_valid_models(custom_llm_provider="alibaba_token_plan", check_provider_endpoint=False))
        == catalog
    )
    assert catalog.issubset(litellm.get_valid_models(check_provider_endpoint=False))


def test_cost_map_reload_updates_provider_models() -> None:
    model: Final = "alibaba_token_plan/test-model-from-cost-map"
    assert model not in litellm.models_by_provider["alibaba_token_plan"]

    litellm.add_known_models(model_cost_map={model: {"litellm_provider": "alibaba_token_plan", "mode": "chat"}})
    try:
        assert model in litellm.alibaba_token_plan_models
        assert model in litellm.models_by_provider["alibaba_token_plan"]
    finally:
        litellm.alibaba_token_plan_models.discard(model)

    assert model not in litellm.models_by_provider["alibaba_token_plan"]


@pytest.mark.asyncio
@pytest.mark.parametrize("from_environment", [False, True])
@pytest.mark.parametrize(
    "api_base,expected_url",
    [
        (None, f"{ANTHROPIC_BASE}/v1/messages"),
        (API_BASE, f"{ANTHROPIC_BASE}/v1/messages"),
        (
            "https://gateway.example/token-plan/compatible-mode/v1",
            "https://gateway.example/token-plan/apps/anthropic/v1/messages",
        ),
    ],
)
async def test_native_anthropic_messages(
    api_base: str | None, expected_url: str, from_environment: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "token-plan-test-key")
    monkeypatch.delenv("ALIBABA_TOKEN_PLAN_API_BASE", raising=False)
    if from_environment and api_base is not None:
        monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_BASE", api_base)

    def transport(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == expected_url
        assert request.headers["authorization"] == "Bearer token-plan-test-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        payload: Final = json.loads(request.content)
        assert payload["model"] == MODEL
        assert payload["system"] == "Be concise"
        assert payload["max_tokens"] == 200
        assert payload["thinking"] == {"type": "enabled", "budget_tokens": 128}
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert "max_completion_tokens" not in payload
        return httpx.Response(
            200,
            json={
                "id": "msg_token_plan",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
        )

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(transport))
    async with handler.client:
        response: Final = await anthropic_messages(
            model=PREFIXED_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=200,
            system="Be concise",
            thinking={"type": "enabled", "budget_tokens": 128},
            api_base=None if from_environment else api_base,
            client=handler,
        )
    assert response["content"] == [{"type": "text", "text": "Hello"}]
    assert response["usage"] == {"input_tokens": 4, "output_tokens": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("use_http_handler", [False, True])
@pytest.mark.parametrize("from_environment", [False, True])
async def test_chat_preserves_custom_openai_compatible_base(
    use_http_handler: bool, from_environment: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_base: Final = "https://gateway.example/token-plan/compatible-mode/v1"
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "token-plan-test-key")
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", str(use_http_handler).lower())
    if from_environment:
        monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_BASE", api_base)
    else:
        monkeypatch.delenv("ALIBABA_TOKEN_PLAN_API_BASE", raising=False)

    def transport(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://gateway.example/token-plan/compatible-mode/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer token-plan-test-key"
        assert json.loads(request.content)["model"] == MODEL
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
            },
        )

    injected_transport: Final = httpx.MockTransport(transport)
    litellm.in_memory_llm_clients_cache.flush_cache()
    async_handler: Final = AsyncHTTPHandler(transport=injected_transport)
    async with async_handler.client as async_client:
        with httpx.Client(transport=injected_transport) as sync_client:
            monkeypatch.setattr(litellm, "client_session", sync_client)
            monkeypatch.setattr(litellm, "aclient_session", async_client)
            response: Final = await litellm.acompletion(
                model=PREFIXED_MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                api_base=None if from_environment else api_base,
                client=async_handler if use_http_handler else None,
            )
            assert response.choices[0].message.content == "OK"
    litellm.in_memory_llm_clients_cache.flush_cache()


def successful_reply(request: httpx.Request) -> Mapping[str, object]:
    if request.url.path.endswith("/messages"):
        return {
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["sdk_chat", "http_chat", "messages"])
@pytest.mark.parametrize("credential_source", ["explicit", "environment", "generic"])
async def test_requests_prefer_token_plan_credentials_over_openai_keys(
    api: str, credential_source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-environment-key")
    monkeypatch.setattr(litellm, "openai_key", "unrelated-openai-key")
    monkeypatch.setattr(litellm, "api_key", "generic-key")
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", str(api == "http_chat").lower())
    monkeypatch.delenv("ALIBABA_TOKEN_PLAN_API_BASE", raising=False)
    monkeypatch.delenv("ALIBABA_TOKEN_PLAN_API_KEY", raising=False)
    if credential_source in ("explicit", "environment"):
        monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "environment-key")
    explicit_key: Final = "explicit-key" if credential_source == "explicit" else None
    expected_key: Final = f"{credential_source}-key"
    assert get_llm_provider(model=PREFIXED_MODEL)[1] == "alibaba_token_plan"

    def transport(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {expected_key}"
        return httpx.Response(200, json=successful_reply(request))

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(transport))
    litellm.in_memory_llm_clients_cache.flush_cache()
    async with handler.client as client:
        monkeypatch.setattr(litellm, "aclient_session", client)
        request: Final = (
            anthropic_messages(
                model=PREFIXED_MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=20,
                api_key=explicit_key,
                client=handler,
            )
            if api == "messages"
            else litellm.acompletion(
                model=PREFIXED_MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                api_key=explicit_key,
                client=handler if api == "http_chat" else None,
            )
        )
        assert await request is not None
    litellm.in_memory_llm_clients_cache.flush_cache()
