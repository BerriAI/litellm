import asyncio
import contextlib
import gzip
import io
import json
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from functools import partial
from itertools import chain
from typing import Final, Literal

import httpx
import pytest
import respx
from openai.types.responses import Response as OpenAIResponse
from pydantic import JsonValue

import litellm
from litellm.llms.bigmodel.chat.transformation import BigModelChatConfig
from litellm.llms.bigmodel.common_utils import get_bigmodel_url
from litellm.llms.bigmodel.messages.transformation import BigModelMessagesConfig
from litellm.llms.bigmodel.responses.transformation import BigModelResponsesAPIConfig
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

MODEL: Final = "glm-5.3"
BASE: Final = "https://open.bigmodel.cn"
KEY: Final = "test-bigmodel-key"
BASE_PATHS: Final = (
    "",
    "/",
    "/api",
    "/api/paas/v4",
    "/api/paas/v4/",
    "/api/paas/v4/chat/completions",
    "/api/v1",
    "/api/v1/responses",
    "/api/anthropic",
    "/api/anthropic/v1",
    "/api/anthropic/v1/messages",
)


class _ByteChunks(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = chunks
        self.is_closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.is_closed = True


@pytest.fixture(autouse=True)
def bigmodel_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("BIGMODEL_API_KEY", KEY)
    monkeypatch.delenv("BIGMODEL_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    yield


def response_payload() -> Mapping[str, JsonValue]:
    return {
        "id": "resp_bigmodel_test",
        "created_at": 1,
        "object": "response",
        "model": MODEL,
        "status": "completed",
        "parallel_tool_calls": False,
        "tools": None,
        "tool_choice": None,
        "output": [
            {"id": "rs_test", "type": "reasoning", "content": {"type": "reasoning_text", "text": "Thinking"}},
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": {"type": "output_text", "text": "Hello", "annotations": []},
            },
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 10,
            "input_tokens_details": {"cached_tokens": 5},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }


def test_provider_resolution_and_all_native_endpoints() -> None:
    model, provider, key, base = litellm.get_llm_provider(f"bigmodel/{MODEL}")
    assert (model, provider, key, base) == (MODEL, "bigmodel", KEY, f"{BASE}/api/paas/v4")
    assert isinstance(ProviderConfigManager.get_provider_chat_config(MODEL, LlmProviders.BIGMODEL), BigModelChatConfig)
    assert isinstance(
        ProviderConfigManager.get_provider_responses_api_config("bigmodel", MODEL), BigModelResponsesAPIConfig
    )
    assert isinstance(
        ProviderConfigManager.get_provider_anthropic_messages_config(MODEL, LlmProviders.BIGMODEL),
        BigModelMessagesConfig,
    )
    assert "bigmodel" in litellm.provider_list
    assert litellm.get_llm_provider(f"zai/{MODEL}")[3] == "https://api.z.ai/api/paas/v4"


@pytest.mark.parametrize("path", BASE_PATHS)
def test_one_base_selects_each_api_without_duplicate_segments(path: str) -> None:
    assert get_bigmodel_url(f"{BASE}{path}", "chat") == f"{BASE}/api/paas/v4"
    assert get_bigmodel_url(f"{BASE}{path}", "responses") == f"{BASE}/api/v1/responses"
    assert get_bigmodel_url(f"{BASE}{path}", "messages") == f"{BASE}/api/anthropic/v1/messages"


@pytest.mark.parametrize(
    "path", ("/api/coding/paas/v4", "/api/coding/paas/v4/", "/api/coding/paas/v4/chat/completions")
)
def test_coding_plan_base_keeps_its_chat_path_and_shares_the_other_apis(path: str) -> None:
    assert get_bigmodel_url(f"{BASE}{path}", "chat") == f"{BASE}/api/coding/paas/v4"
    assert get_bigmodel_url(f"{BASE}{path}", "responses") == f"{BASE}/api/v1/responses"
    assert get_bigmodel_url(f"{BASE}{path}", "messages") == f"{BASE}/api/anthropic/v1/messages"


def test_custom_base_and_environment_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIGMODEL_API_BASE", "https://env.example/prefix/api/v1")
    assert get_bigmodel_url(None, "messages") == "https://env.example/prefix/api/anthropic/v1/messages"
    assert (
        get_bigmodel_url("https://override.example/prefix/api/anthropic/", "chat")
        == "https://override.example/prefix/api/paas/v4"
    )
    assert (
        BigModelResponsesAPIConfig().validate_environment({}, MODEL, GenericLiteLLMParams(api_key="explicit"))[
            "Authorization"
        ]
        == "Bearer explicit"
    )
    assert BigModelResponsesAPIConfig().validate_environment({}, MODEL, None)["Authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("is_async", (False, True))
@pytest.mark.asyncio
@respx.mock
async def test_chat_uses_native_endpoint_and_maps_token_limit(is_async: bool) -> None:
    route: Final = respx.post(f"{BASE}/api/paas/v4/chat/completions").respond(
        json={
            "id": "chat_test",
            "object": "chat.completion",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
    )
    params: Final = {
        "model": f"bigmodel/{MODEL}",
        "messages": [{"role": "user", "content": "Hello"}],
        "api_base": f"{BASE}/api/anthropic",
        "max_completion_tokens": 10,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "extra_body": {"clear_thinking": False},
    }
    response: Final = await litellm.acompletion(**params) if is_async else litellm.completion(**params)
    assert response.choices[0].message.content == "Hello"
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert body["model"] == MODEL
    assert body["max_tokens"] == 10
    assert "max_completion_tokens" not in body
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "low"
    assert body["clear_thinking"] is False
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert route.call_count == 1
    if is_async:
        health: Final = await litellm.ahealth_check(params, mode="chat")
        assert "error" not in health
        assert json.loads(route.calls.last.request.content)["thinking"] == {"type": "enabled"}


@respx.mock
def test_chat_does_not_send_the_openai_organization_to_bigmodel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-openai-only")
    route: Final = respx.post(f"{BASE}/api/paas/v4/chat/completions").respond(
        json={
            "id": "chat_test",
            "object": "chat.completion",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    _ = litellm.completion(model=f"bigmodel/{MODEL}", messages=[{"role": "user", "content": "Hello"}])
    assert "openai-organization" not in route.calls.last.request.headers


@pytest.mark.parametrize("is_async", (False, True))
@pytest.mark.asyncio
@respx.mock
async def test_responses_native_body_text_usage_and_continuation(is_async: bool) -> None:
    route: Final = respx.post(f"{BASE}/api/v1/responses").respond(json=response_payload())
    params: Final = {
        "model": f"bigmodel/{MODEL}",
        "input": "Hello",
        "api_base": BASE,
        "max_output_tokens": 1,
        "reasoning": {"effort": "max"},
    }
    response: Final = await litellm.aresponses(**params) if is_async else litellm.responses(**params)
    assert isinstance(response, ResponsesAPIResponse)
    assert response.output_text == "Hello"
    assert isinstance(response.usage, ResponseAPIUsage)
    assert response.usage.total_tokens == 30
    assert response.usage.input_tokens_details.cached_tokens == 5
    assert response.usage.output_tokens_details.reasoning_tokens == 3
    assert response.tools == []
    assert response.tool_choice == "auto"
    OpenAIResponse.model_validate(response.model_dump())
    first_body: Final = json.loads(route.calls.last.request.content)
    assert first_body["model"] == MODEL
    assert first_body["max_output_tokens"] == 1
    assert first_body["reasoning"] == {"effort": "max"}
    assert first_body.get("store") is not True
    assert "messages" not in first_body
    assert route.calls.last.request.headers["authorization"] == f"Bearer {KEY}"
    followup: Final = {**params, "previous_response_id": response.id, "store": True}
    if is_async:
        await litellm.aresponses(**followup)
    else:
        litellm.responses(**followup)
    assert json.loads(route.calls.last.request.content)["previous_response_id"] == "resp_bigmodel_test"
    assert json.loads(route.calls.last.request.content)["store"] is True


@pytest.mark.asyncio
@respx.mock
async def test_responses_decodes_compressed_response_once() -> None:
    respx.post(f"{BASE}/api/v1/responses").respond(
        content=gzip.compress(json.dumps(response_payload()).encode()),
        headers={"content-encoding": "gzip", "content-type": "application/json"},
    )
    response: Final = await litellm.aresponses(model=f"bigmodel/{MODEL}", input="Hello")
    assert response.output_text == "Hello"
    assert response.usage.total_tokens == 30


@pytest.mark.parametrize(
    "reasoning_params",
    (
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        {"reasoning_effort": "low"},
    ),
)
def test_messages_preserves_native_reasoning_params(reasoning_params: Mapping[str, JsonValue]) -> None:
    payload: Final = BigModelMessagesConfig().transform_anthropic_messages_request(
        model=MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 2048, **reasoning_params},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert {key: payload.get(key) for key in reasoning_params} == reasoning_params


@pytest.mark.parametrize(
    ("model", "extra", "expected"),
    (
        (MODEL, {}, {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}),
        (MODEL, {"reasoning_effort": "high"}, {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}),
        ("glm-4.7", {}, {"thinking": {"type": "disabled"}, "reasoning_effort": None}),
    ),
)
@respx.mock
def test_chat_asks_an_always_thinking_model_for_its_lowest_effort_when_thinking_is_disabled(
    model: str, extra: Mapping[str, JsonValue], expected: Mapping[str, JsonValue]
) -> None:
    route: Final = respx.post(f"{BASE}/api/paas/v4/chat/completions").respond(
        json={
            "id": "chat_test",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
    )
    litellm.completion(
        model=f"bigmodel/{model}",
        messages=[{"role": "user", "content": "Hello"}],
        thinking={"type": "disabled"},
        **extra,
    )
    body: Final = json.loads(route.calls.last.request.content)
    assert {key: body.get(key) for key in expected} == expected


@pytest.mark.parametrize(
    ("model", "extra", "expected"),
    (
        (MODEL, {}, {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}),
        (
            MODEL,
            {"output_config": {"effort": "high"}},
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        ),
        (
            MODEL,
            {"reasoning_effort": "high"},
            {"thinking": {"type": "adaptive"}, "reasoning_effort": "high", "output_config": None},
        ),
        ("glm-4.7", {}, {"thinking": {"type": "disabled"}, "output_config": None}),
    ),
)
def test_messages_asks_an_always_thinking_model_for_its_lowest_effort_when_thinking_is_disabled(
    model: str, extra: Mapping[str, JsonValue], expected: Mapping[str, JsonValue]
) -> None:
    payload: Final = BigModelMessagesConfig().transform_anthropic_messages_request(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 64, "thinking": {"type": "disabled"}, **extra},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert {key: payload.get(key) for key in expected} == expected


@pytest.mark.parametrize(
    "reasoning_params",
    ({"thinking": {"type": "enabled", "budget_tokens": 1024}}, {"output_config": {"effort": "low"}}),
)
@pytest.mark.parametrize("temperature", (0, 0.2))
def test_messages_keeps_the_callers_temperature_next_to_thinking(
    reasoning_params: Mapping[str, JsonValue], temperature: float
) -> None:
    payload: Final = BigModelMessagesConfig().transform_anthropic_messages_request(
        model=MODEL,
        messages=[{"role": "user", "content": "Hello"}],
        anthropic_messages_optional_request_params={"max_tokens": 2048, "temperature": temperature, **reasoning_params},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert payload.get("temperature") == temperature


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("web_fetch_requests", (None, 2))
async def test_messages_forward_native_tools_thinking_and_usage(web_fetch_requests: int | None) -> None:
    route: Final = respx.post(f"{BASE}/api/anthropic/v1/messages").respond(
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 20,
                "output_tokens": 10,
                "server_tool_use": {
                    "web_search_requests": 1,
                    **({"web_fetch_requests": web_fetch_requests} if web_fetch_requests is not None else {}),
                },
            },
        }
    )
    tools: Final = [{"name": "weather", "input_schema": {"type": "object", "properties": {}}}]
    params: Final = {
        "model": f"bigmodel/{MODEL}",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": tools,
        "thinking": {"type": "enabled", "budget_tokens": 50},
        "api_base": f"{BASE}/api/paas/v4",
        "provider_specific_header": {
            "custom_llm_provider": "bigmodel",
            "native_anthropic_messages_only": True,
            "extra_headers": {"x-native-message": "preserved"},
        },
    }
    response: Final = await litellm.anthropic.messages.acreate(**params)
    assert response["content"][0]["text"] == "Hello"
    assert response["usage"]["server_tool_use"] == {
        "web_search_requests": 1,
        "web_fetch_requests": web_fetch_requests or 0,
    }
    body: Final = json.loads(route.calls.last.request.content)
    assert body["model"] == MODEL
    assert body["tools"] == tools
    assert body["thinking"]["type"] == "enabled"
    assert "input" not in body
    assert route.calls.last.request.headers["x-api-key"] == KEY
    assert route.calls.last.request.headers["x-native-message"] == "preserved"


@pytest.mark.parametrize(
    "terminal_type,status",
    (("response.completed", "completed"), ("response.incomplete", "incomplete")),
)
@pytest.mark.parametrize("include_logprobs", (False, True))
@pytest.mark.parametrize("content_type", ("text/event-stream", "application/json"))
@pytest.mark.asyncio
@respx.mock
async def test_responses_sse_without_done_preserves_terminal_usage(
    terminal_type: str, status: str, include_logprobs: bool, content_type: str
) -> None:
    logprobs: Final = [{"token": "Hello", "bytes": [72, 101, 108, 108, 111], "logprob": -0.1, "top_logprobs": []}]
    events: Final = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                **{key: value for key, value in response_payload().items() if key != "parallel_tool_calls"},
                "status": "in_progress",
                "output": [],
                "usage": None,
            },
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_test",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
        },
        {
            "type": "response.content_part.done",
            "sequence_number": 2,
            "item_id": "msg_test",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "Hello"},
        },
        {
            "type": "response.content_part.done",
            "sequence_number": 3,
            "item_id": "rs_test",
            "output_index": 1,
            "content_index": 0,
            "part": {"type": "reasoning_text", "text": "Thinking"},
        },
        {
            "type": "response.output_text.done",
            "sequence_number": 4,
            "item_id": "msg_test",
            "output_index": 0,
            "content_index": 0,
            "text": "Hello",
            **({"logprobs": logprobs} if include_logprobs else {}),
        },
        {"type": terminal_type, "sequence_number": 5, "response": {**response_payload(), "status": status}},
    )
    sse: Final = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    respx.post(f"{BASE}/api/v1/responses").respond(content=sse, headers={"content-type": content_type})
    stream: Final = await litellm.aresponses(model=f"bigmodel/{MODEL}", input="Hello", stream=True)
    chunks: Final = tuple([chunk async for chunk in stream])
    assert chunks[-1].type == terminal_type
    assert chunks[-1].response.status == status
    assert chunks[-1].response.usage.total_tokens == 30
    assert chunks[-1].response.output_text == "Hello"
    OpenAIResponse.model_validate(chunks[-1].response.model_dump())
    assert chunks[0].response.parallel_tool_calls is True
    assert chunks[-2].logprobs == (logprobs if include_logprobs else [])
    assert tuple(chunk.delta for chunk in chunks if chunk.type == "response.output_text.delta") == ("Hello",)
    assert tuple(chunk.part.text for chunk in chunks if chunk.type == "response.content_part.done") == (
        "Hello",
        "Thinking",
    )


@respx.mock
def test_get_response_uses_bigmodel_url_and_normalizes_usage() -> None:
    route: Final = respx.post(f"{BASE}/api/v1/responses").respond(json=response_payload())
    response: Final = litellm.responses(model=f"bigmodel/{MODEL}", input="Hello", store=True)
    respx.get(f"{BASE}/api/v1/responses/resp_bigmodel_test").respond(json=response_payload())
    retrieved: Final = litellm.get_responses(response_id=response.id)
    assert retrieved.output_text == "Hello"
    assert retrieved.usage.total_tokens == 30
    assert route.call_count == 1


@pytest.mark.parametrize("provider_api", ("chat", "responses", "messages"))
def test_missing_bigmodel_key_does_not_use_other_provider_credentials(
    provider_api: Literal["chat", "responses", "messages"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BIGMODEL_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-anthropic-key")
    call: Final = {
        "chat": partial(litellm.completion, model=f"bigmodel/{MODEL}", messages=[{"role": "user", "content": "Hello"}]),
        "responses": partial(BigModelResponsesAPIConfig().validate_environment, {}, MODEL, None),
        "messages": partial(BigModelMessagesConfig().validate_anthropic_messages_environment, {}, MODEL, (), {}, {}),
    }[provider_api]
    with pytest.raises(litellm.AuthenticationError, match="BIGMODEL_API_KEY"):
        call()


def _audio_file() -> io.BytesIO:
    audio: Final = io.BytesIO(b"RIFF0000WAVEfmt ")
    audio.name = "audio.wav"
    return audio


@pytest.mark.parametrize(
    "call",
    (
        partial(litellm.speech, model=f"bigmodel/{MODEL}", input="Hello", voice="tongtong"),
        partial(litellm.image_generation, model=f"bigmodel/{MODEL}", prompt="Hello"),
        lambda: litellm.transcription(model=f"bigmodel/{MODEL}", file=_audio_file()),
    ),
    ids=("speech", "image_generation", "transcription"),
)
@respx.mock
def test_openai_routes_bigmodel_does_not_serve_send_no_request(
    call: Callable[[], object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    upstream: Final = respx.route().mock(return_value=httpx.Response(200, json={}))
    with contextlib.suppress(Exception):
        call()
    assert not upstream.called


@pytest.mark.parametrize(
    ("code", "error_class", "status_code"),
    (
        (401, litellm.AuthenticationError, 401),
        (1113, litellm.RateLimitError, 429),
        (1301, litellm.BadRequestError, 400),
        (1200, litellm.InternalServerError, 500),
    ),
)
@respx.mock
def test_responses_failure_envelope_raises_its_status_instead_of_a_parse_error(
    code: int, error_class: type[Exception], status_code: int
) -> None:
    respx.post(f"{BASE}/api/v1/responses").respond(json={"code": code, "msg": "Token expired", "success": False})
    with pytest.raises(error_class, match="Token expired") as raised:
        litellm.responses(model=f"bigmodel/{MODEL}", input="Hello")
    assert getattr(raised.value, "status_code", None) == status_code


@pytest.mark.parametrize(
    ("code", "error_class"),
    ((1261, litellm.ContextWindowExceededError), (1301, litellm.ContentPolicyViolationError)),
)
@pytest.mark.parametrize("route", ("chat", "responses"))
@respx.mock
def test_prompt_too_long_and_unsafe_content_raise_the_errors_fallbacks_catch(
    code: int, error_class: type[Exception], route: Literal["chat", "responses"]
) -> None:
    if route == "chat":
        respx.post(f"{BASE}/api/paas/v4/chat/completions").respond(
            status_code=400, json={"error": {"code": str(code), "message": "rejected by BigModel"}}
        )
    else:
        respx.post(f"{BASE}/api/v1/responses").respond(
            json={"code": code, "msg": "rejected by BigModel", "success": False}
        )
    call: Final = (
        (lambda: litellm.completion(model=f"bigmodel/{MODEL}", messages=[{"role": "user", "content": "Hello"}]))
        if route == "chat"
        else (lambda: litellm.responses(model=f"bigmodel/{MODEL}", input="Hello"))
    )
    with pytest.raises(error_class, match="rejected by BigModel") as raised:
        call()
    assert raised.value.status_code == 400


@pytest.mark.parametrize(
    ("code", "error_class"),
    ((1261, litellm.ContextWindowExceededError), (1301, litellm.ContentPolicyViolationError)),
)
@pytest.mark.parametrize("is_async", (False, True))
@pytest.mark.asyncio
@respx.mock
async def test_streamed_prompt_too_long_and_unsafe_content_raise_the_errors_fallbacks_catch(
    code: int, error_class: type[Exception], is_async: bool
) -> None:
    respx.post(f"{BASE}/api/v1/responses").respond(json={"code": code, "msg": "rejected by BigModel", "success": False})
    params: Final = {"model": f"bigmodel/{MODEL}", "input": "Hello", "stream": True}

    async def call() -> None:
        if is_async:
            await litellm.aresponses(**params)
            return
        litellm.responses(**params)

    with pytest.raises(error_class, match="rejected by BigModel") as raised:
        await call()
    assert raised.value.status_code == 400


@pytest.mark.parametrize("route", ("responses", "messages"))
@pytest.mark.asyncio
@respx.mock
async def test_a_non_json_answer_surfaces_its_body_text(route: Literal["responses", "messages"]) -> None:
    path: Final = "/api/v1/responses" if route == "responses" else "/api/anthropic/v1/messages"
    respx.post(f"{BASE}{path}").respond(
        content=b"<html>gateway maintenance</html>", headers={"content-type": "text/html"}
    )
    call: Final = (
        (lambda: litellm.aresponses(model=f"bigmodel/{MODEL}", input="Hello"))
        if route == "responses"
        else (
            lambda: litellm.anthropic.messages.acreate(
                model=f"bigmodel/{MODEL}", max_tokens=16, messages=[{"role": "user", "content": "Hello"}]
            )
        )
    )
    with pytest.raises(Exception, match="gateway maintenance") as raised:
        await call()
    assert "validation error" not in str(raised.value)


@pytest.mark.parametrize("stream", (False, True))
@respx.mock
def test_responses_forward_codex_tool_params_and_echo_them_when_the_answer_omits_them(stream: bool) -> None:
    tools: Final = [{"type": "function", "name": "weather", "parameters": {"type": "object", "properties": {}}}]
    completed: Final = {
        "type": "response.completed",
        "sequence_number": 0,
        "response": {**response_payload(), "parallel_tool_calls": None},
    }
    route: Final = respx.post(f"{BASE}/api/v1/responses").respond(
        **(
            {"content": f"event: response.completed\ndata: {json.dumps(completed)}\n\n"}
            if stream
            else {"json": {**response_payload(), "parallel_tool_calls": None}}
        )
    )
    answer: Final = litellm.responses(
        model=f"bigmodel/{MODEL}",
        input="Hello",
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=True,
        include=["reasoning.encrypted_content"],
        stream=stream,
    )
    response: Final = tuple(answer)[-1].response if stream else answer
    validated: Final = OpenAIResponse.model_validate(response.model_dump())
    sent: Final = json.loads(route.calls.last.request.content)
    assert sent["parallel_tool_calls"] is True
    assert sent["include"] == ["reasoning.encrypted_content"]
    assert validated.tools[0].name == "weather"
    assert validated.tool_choice == "required"
    assert validated.parallel_tool_calls is True


@pytest.mark.asyncio
@respx.mock
async def test_responses_stream_answered_with_a_failure_envelope_raises_its_status() -> None:
    respx.post(f"{BASE}/api/v1/responses").respond(json={"code": 401, "msg": "Token expired", "success": False})
    with pytest.raises(litellm.AuthenticationError, match="Token expired"):
        await litellm.aresponses(model=f"bigmodel/{MODEL}", input="Hello", stream=True)
    with pytest.raises(litellm.AuthenticationError, match="Token expired"):
        litellm.responses(model=f"bigmodel/{MODEL}", input="Hello", stream=True)


@pytest.mark.parametrize("is_async", (False, True))
@pytest.mark.asyncio
@respx.mock
async def test_responses_stream_answered_with_one_json_response_replays_it_as_events(is_async: bool) -> None:
    respx.post(f"{BASE}/api/v1/responses").respond(json=response_payload())
    request: Final = {"model": f"bigmodel/{MODEL}", "input": "Hello", "stream": True}
    chunks: Final = (
        tuple([chunk async for chunk in await litellm.aresponses(**request)])
        if is_async
        else tuple(litellm.responses(**request))
    )
    assert chunks[-1].type == "response.completed"
    assert chunks[-1].response.output_text == "Hello"
    assert chunks[-1].response.usage.total_tokens == 30


class _GatedStream(httpx.AsyncByteStream):
    def __init__(self, first: bytes, rest: bytes, gate: asyncio.Event) -> None:
        self._first = first
        self._rest = rest
        self._gate = gate

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first
        await self._gate.wait()
        yield self._rest


@pytest.mark.asyncio
@respx.mock
async def test_responses_stream_labeled_json_without_a_length_yields_events_before_it_ends() -> None:
    created: Final = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {**response_payload(), "status": "in_progress", "output": [], "usage": None},
    }
    completed: Final = {"type": "response.completed", "sequence_number": 1, "response": response_payload()}
    gate: Final = asyncio.Event()
    respx.post(f"{BASE}/api/v1/responses").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_GatedStream(
                f"event: response.created\ndata: {json.dumps(created)}\n\n".encode(),
                f"event: response.completed\ndata: {json.dumps(completed)}\n\n".encode(),
                gate,
            ),
        )
    )

    stream: Final = await asyncio.wait_for(
        litellm.aresponses(model=f"bigmodel/{MODEL}", input="Hello", stream=True), timeout=10
    )
    first: Final = await asyncio.wait_for(anext(stream), timeout=10)
    gate.set()
    rest: Final = tuple([chunk async for chunk in stream])
    assert first.type == "response.created"
    assert rest[-1].type == "response.completed"
    assert rest[-1].response.output_text == "Hello"


@pytest.mark.parametrize("event_type", ("response.failed", "error"))
@pytest.mark.asyncio
@respx.mock
async def test_responses_failure_events_raise_instead_of_completing(event_type: str) -> None:
    event: Final = (
        {"type": event_type, "sequence_number": 1, "code": "server_error", "message": "Provider failed"}
        if event_type == "error"
        else {
            "type": event_type,
            "sequence_number": 1,
            "response": {
                **response_payload(),
                "status": "failed",
                "error": {"code": "server_error", "message": "Provider failed"},
            },
        }
    )
    respx.post(f"{BASE}/api/v1/responses").respond(
        content=f"event: {event_type}\ndata: {json.dumps(event)}\n\n",
        headers={"content-type": "text/event-stream"},
    )
    stream: Final = await litellm.aresponses(model=f"bigmodel/{MODEL}", input="Hello", stream=True)
    with pytest.raises(litellm.exceptions.MidStreamFallbackError, match="Provider failed"):
        await anext(stream)


@respx.mock
def test_responses_preserve_tools_multimodal_input_and_existing_usage_total() -> None:
    payload: Final = response_payload()
    route: Final = respx.post(f"{BASE}/api/v1/responses").respond(
        json={**payload, "usage": {**payload["usage"], "total_tokens": 31}}
    )
    input_items: Final = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this"},
                {"type": "input_image", "image_url": "https://example.com/image.png"},
            ],
        }
    ]
    tools: Final = [{"type": "function", "name": "weather", "parameters": {"type": "object", "properties": {}}}]
    response: Final = litellm.responses(
        model=f"bigmodel/{MODEL}",
        input=input_items,
        tools=tools,
        tool_choice="auto",
        text={"format": {"type": "json_object"}},
    )
    assert response.usage.total_tokens == 31
    assert response.output[0].content[0].text == "Thinking"
    body: Final = json.loads(route.calls.last.request.content)
    assert body["input"] == input_items
    assert body["tools"] == tools
    assert body["text"] == {"format": {"type": "json_object"}}


@respx.mock
def test_responses_preserve_explicit_tool_configuration() -> None:
    tools: Final = [{"type": "function", "name": "weather", "parameters": {"type": "object"}, "strict": False}]
    tool_choice: Final = {"type": "function", "name": "weather"}
    reasoning: Final = {
        "id": "rs_test",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "Planning"}],
        "content": [{"type": "reasoning_text", "text": "Thinking"}],
    }
    tool_call: Final = {
        "type": "function_call",
        "id": "fc_test",
        "call_id": "call_test",
        "name": "weather",
        "arguments": "{}",
        "status": "completed",
    }
    respx.post(f"{BASE}/api/v1/responses").respond(
        json={
            **response_payload(),
            "output": [reasoning, tool_call],
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": True,
        }
    )
    response: Final = litellm.responses(model=f"bigmodel/{MODEL}", input="Hello", tools=tools, tool_choice=tool_choice)
    validated: Final = OpenAIResponse.model_validate(response.model_dump())
    assert validated.tools[0].name == "weather"
    assert validated.tool_choice.name == "weather"
    assert validated.parallel_tool_calls is True
    assert validated.output[0].summary[0].text == reasoning["summary"][0]["text"]
    assert validated.output[0].content[0].text == reasoning["content"][0]["text"]
    assert validated.output[1].model_dump(exclude_none=True) == tool_call


@pytest.mark.asyncio
@respx.mock
async def test_router_routes_one_model_configuration_to_three_native_endpoints() -> None:
    chat: Final = respx.post(f"{BASE}/api/paas/v4/chat/completions").respond(
        json={
            "id": "chat_test",
            "created": 1,
            "model": MODEL,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
    )
    responses: Final = respx.post(f"{BASE}/api/v1/responses").respond(json=response_payload())
    messages: Final = respx.post(f"{BASE}/api/anthropic/v1/messages").respond(
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
    )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "glm-native",
                "litellm_params": {"model": f"bigmodel/{MODEL}", "api_key": KEY, "api_base": BASE},
            }
        ],
        num_retries=0,
    )
    chat_result: Final = await router.acompletion(model="glm-native", messages=[{"role": "user", "content": "Hello"}])
    responses_result: Final = await router.aresponses(model="glm-native", input="Hello")
    messages_result: Final = await router.aanthropic_messages(
        model="glm-native", max_tokens=10, messages=[{"role": "user", "content": "Hello"}]
    )
    assert (
        chat_result.choices[0].message.content
        == responses_result.output_text
        == messages_result["content"][0]["text"]
        == "Hello"
    )
    assert tuple(route.call_count for route in (chat, responses, messages)) == (1, 1, 1)
    assert all(json.loads(route.calls.last.request.content)["model"] == MODEL for route in (chat, responses, messages))
    assert (
        chat.calls.last.request.headers["authorization"]
        == responses.calls.last.request.headers["authorization"]
        == f"Bearer {KEY}"
    )
    assert messages.calls.last.request.headers["x-api-key"] == KEY


@pytest.mark.asyncio
@respx.mock
async def test_chat_stream_preserves_deltas_and_final_usage() -> None:
    chunks: Final = (
        {
            "id": "chat_test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {"reasoning_content": "Thinking"}, "finish_reason": None}],
        },
        {
            "id": "chat_test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
        },
        {
            "id": "chat_test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        },
    )
    respx.post(f"{BASE}/api/paas/v4/chat/completions").respond(
        content="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )
    stream: Final = await litellm.acompletion(
        model=f"bigmodel/{MODEL}",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    output: Final = tuple([chunk async for chunk in stream])
    assert (
        "".join(choice.delta.content or "" for choice in chain.from_iterable(chunk.choices for chunk in output))
        == "Hello"
    )
    assert (
        "".join(
            getattr(choice.delta, "reasoning_content", "") or ""
            for choice in chain.from_iterable(chunk.choices for chunk in output)
        )
        == "Thinking"
    )
    assert tuple(chunk.usage.total_tokens for chunk in output if hasattr(chunk, "usage") and chunk.usage) == (30,)


@pytest.mark.asyncio
@respx.mock
async def test_messages_stream_preserves_native_events() -> None:
    events: Final = (
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 0,
                    "server_tool_use": {"web_search_requests": 0},
                },
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "先思考"}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "test-signature"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "你好 🌏"}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 10},
        },
        {"type": "message_stop"},
    )
    native_events: Final = tuple(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events
    )
    content: Final = (
        "id: bigmodel-event\nretry: 1000\n"
        + native_events[0]
        + "event: ping\n\n"
        + "".join(native_events[1:])
        + "data: [DONE]\n\n"
    ).encode()
    upstream: Final = _ByteChunks(tuple(content[index : index + 7] for index in range(0, len(content), 7)))
    respx.post(f"{BASE}/api/anthropic/v1/messages").mock(
        return_value=httpx.Response(
            200,
            stream=upstream,
            headers={"content-type": "text/event-stream"},
        )
    )
    stream: Final = await litellm.anthropic.messages.acreate(
        model=f"bigmodel/{MODEL}",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    output: Final = tuple([chunk async for chunk in stream])
    lines: Final = tuple(chain.from_iterable(chunk.decode().splitlines() for chunk in output))
    parsed: Final = tuple(json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: {"))
    assert tuple(chunk["type"] for chunk in parsed) == tuple(event["type"] for event in events)
    assert tuple(event["delta"] for event in parsed if event["type"] == "content_block_delta") == (
        {"type": "thinking_delta", "thinking": "先思考"},
        {"type": "signature_delta", "signature": "test-signature"},
        {"type": "text_delta", "text": "你好 🌏"},
    )
    assert parsed[-2]["usage"]["output_tokens"] == 10
    assert parsed[0]["message"]["usage"]["server_tool_use"] == {
        "web_search_requests": 0,
        "web_fetch_requests": 0,
    }
    assert tuple(line for line in lines if line.startswith("data: {"))[1:] == tuple(
        f"data: {json.dumps(event, ensure_ascii=False)}" for event in events[1:]
    )
    assert "event: ping" in lines
    assert "data: [DONE]" in lines
    assert b"id: bigmodel-event\n" in output[0]
    assert b"retry: 1000\n" in output[0]
    assert upstream.is_closed


@respx.mock
def test_stored_response_lifecycle_reuses_base_and_unwraps_id() -> None:
    respx.post(f"{BASE}/api/v1/responses").respond(json=response_payload())
    response: Final = litellm.responses(model=f"bigmodel/{MODEL}", input="Hello", store=True)
    listing: Final = respx.get(f"{BASE}/api/v1/responses/resp_bigmodel_test/input_items").respond(
        json={"object": "list", "data": [], "has_more": False}
    )
    deletion: Final = respx.delete(f"{BASE}/api/v1/responses/resp_bigmodel_test").respond(
        json={"id": "resp_bigmodel_test", "object": "response.deleted", "deleted": True}
    )
    assert litellm.list_input_items(response_id=response.id)["data"] == []
    assert litellm.delete_responses(response_id=response.id).deleted is True
    assert listing.call_count == deletion.call_count == 1
    assert deletion.calls.last.request.headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("operation", ("cancel", "compact"))
@respx.mock
def test_unsupported_responses_operations_fail_without_an_upstream_request(operation: str) -> None:
    call: Final = (
        partial(litellm.cancel_responses, response_id="resp_test", custom_llm_provider="bigmodel")
        if operation == "cancel"
        else partial(litellm.compact_responses, model=f"bigmodel/{MODEL}", input="Hello")
    )
    with pytest.raises(litellm.UnsupportedParamsError, match="BigModel"):
        call()
    assert respx.calls.call_count == 0


@pytest.mark.parametrize("model", ("glm-5.3", "glm-5.3-flash", "glm-5.3-flashx"))
def test_bigmodel_model_metadata_and_cached_token_costs(model: str) -> None:
    entry: Final = litellm.model_cost[f"bigmodel/{model}"]
    info: Final = litellm.get_model_info(model=model, custom_llm_provider="bigmodel")
    assert info["litellm_provider"] == "bigmodel"
    assert info["max_input_tokens"] == entry["max_input_tokens"]
    assert info["max_output_tokens"] == entry["max_output_tokens"]
    assert info["supports_reasoning"]
    assert info["supports_function_calling"]
    assert bool(info.get("supports_vision")) == bool(entry.get("supports_vision"))
    prompt_cost, output_cost = litellm.cost_per_token(
        model=model,
        custom_llm_provider="bigmodel",
        prompt_tokens=100,
        completion_tokens=20,
        cache_read_input_tokens=40,
    )
    assert prompt_cost == pytest.approx(60 * entry["input_cost_per_token"] + 40 * entry["cache_read_input_token_cost"])
    assert output_cost == pytest.approx(20 * entry["output_cost_per_token"])
    assert prompt_cost > 0
    assert output_cost > 0
