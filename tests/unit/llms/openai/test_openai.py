import asyncio
import importlib
import os
import json
from collections.abc import Mapping
from types import ModuleType
from typing import Final, Protocol
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI
from openai._types import Response as SDKResponse

import litellm
from litellm.llms.openai.openai import (
    AssistantEventHandler,
    AsyncAssistantEventHandler,
    AsyncCursorPage,
    MessageData,
    OpenAIChatCompletion,
    OpenAIMessage as Message,
    Run,
    SyncCursorPage,
    Thread,
)
from litellm.types.utils import ImageResponse, ModelResponse
from litellm import create_thread, get_thread
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.utils import _invalidate_model_cost_lowercase_map
from openai.types.beta.assistant import Assistant
from openai.types.beta.assistant_deleted import AssistantDeleted
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


@pytest.mark.parametrize(
    "api_base",
    [
        None,
        "https://api.openai.com/v1",
        "https://api.openai.com:443/v1",
        "https://southcentralus.privatelink.api.openai.com/v1",
        "https://eu.api.openai.com/v1",
        "https://us.api.openai.com/v1",
        "HTTPS://API.OPENAI.COM/v1/",
    ],
)
def test_get_stream_options_defaults_include_usage_on_every_openai_backed_host(api_base):
    """
    PrivateLink and regional hostnames reach the real OpenAI backend, so a stream with no caller
    stream_options must ask for the usage chunk exactly as the default base does. Regression guard
    for LIT-6875: spend for those deployments fell back to local token counting.
    """
    assert OpenAIChatCompletion().get_stream_options(stream_options=None, api_base=api_base) == {
        "stream_options": {"include_usage": True}
    }


@pytest.mark.parametrize(
    "api_base",
    [
        "https://my-gateway.example/v1",
        "https://api.openai.com.evil.example/v1",
        "https://notapi.openai.com/v1",
        "https://gateway.example/v1?upstream=api.openai.com",
        "https://openai.internal.example/api.openai.com/v1",
    ],
)
def test_get_stream_options_leaves_foreign_hosts_without_a_usage_default(api_base):
    """Only the host decides: an OpenAI-compatible backend elsewhere may not support stream_options at all."""
    assert OpenAIChatCompletion().get_stream_options(stream_options=None, api_base=api_base) == {}


@pytest.mark.parametrize(
    "api_base",
    ["https://southcentralus.privatelink.api.openai.com/v1", "https://my-gateway.example/v1"],
)
def test_get_stream_options_passes_caller_stream_options_through_on_any_host(api_base):
    caller_options = {"include_usage": False}
    assert OpenAIChatCompletion().get_stream_options(stream_options=caller_options, api_base=api_base) == {
        "stream_options": caller_options
    }


@pytest.fixture(params=("httpx", "sdk"))
def http_backend(request: pytest.FixtureRequest) -> ModuleType:
    return importlib.import_module("httpx" if request.param == "httpx" else SDKResponse.__module__)


@pytest.fixture(params=("openai", "perplexity", "cerebras", "nvidia_nim"))
def sdk_provider(request: pytest.FixtureRequest) -> str:
    return str(request.param)


class _Request(Protocol):
    @property
    def content(self) -> bytes: ...

    @property
    def extensions(self) -> Mapping[str, object]: ...

    @property
    def url(self) -> object: ...

    @property
    def headers(self) -> Mapping[str, str]: ...


@pytest.mark.asyncio
async def test_acompletion_returns_json_reply_over_injected_transport(
    http_backend: ModuleType, sdk_provider: str, monkeypatch: pytest.MonkeyPatch
):
    outbound: Final = asyncio.Queue()

    def respond(request: _Request) -> httpx.Response | SDKResponse:
        assert str(request.url) == f"https://{sdk_provider}.example/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {sdk_provider}-key"
        assert request.extensions["timeout"] == {"connect": 7, "read": 7, "write": 7, "pool": 7}
        outbound.put_nowait(json.loads(request.content))
        return http_backend.Response(
            200,
            json={
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "created": 0,
                "model": "sdk-compat",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "smoke-json-reply"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    async with http_backend.AsyncClient(transport=http_backend.MockTransport(respond)) as http_client:
        monkeypatch.setattr(litellm, "aclient_session", http_client)
        response: Final = await asyncio.wait_for(
            litellm.acompletion(
                model=f"{sdk_provider}/sdk-compat",
                api_key=f"{sdk_provider}-key",
                api_base=f"https://{sdk_provider}.example/v1",
                messages=[{"role": "user", "content": "smoke-json-request"}],
                timeout=7,
                num_retries=0,
                max_retries=0,
            ),
            timeout=10,
        )
        request: Final = await asyncio.wait_for(outbound.get(), timeout=10)
        assert request["model"] == "sdk-compat"
        assert request["messages"] == [{"role": "user", "content": "smoke-json-request"}]
        assert not request.get("stream")
        assert outbound.empty()
        assert response.choices[0].message.content == "smoke-json-reply"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.total_tokens == 15
        assert not http_client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_error"),
    (
        (400, litellm.BadRequestError),
        (429, litellm.RateLimitError),
        (503, litellm.ServiceUnavailableError),
        (None, litellm.Timeout),
    ),
)
async def test_acompletion_preserves_public_errors_for_both_http_clients(
    http_backend: ModuleType,
    sdk_provider: str,
    status: int | None,
    expected_error: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(request: _Request) -> httpx.Response | SDKResponse:
        if status is None:
            raise http_backend.ReadTimeout("upstream timeout", request=request)
        return http_backend.Response(status, json={"error": {"message": "upstream failure", "type": "api_error"}})

    async with http_backend.AsyncClient(transport=http_backend.MockTransport(respond)) as http_client:
        monkeypatch.setattr(litellm, "aclient_session", http_client)
        with pytest.raises(expected_error, match="timed out" if status is None else "upstream failure"):
            await litellm.acompletion(
                model=f"{sdk_provider}/sdk-compat",
                api_key=f"{sdk_provider}-key",
                api_base=f"https://{sdk_provider}.example/v1",
                messages=[{"role": "user", "content": "request"}],
                num_retries=0,
                max_retries=0,
            )
        assert not http_client.is_closed


@pytest.mark.asyncio
async def test_acompletion_streams_text_deltas_over_injected_transport(
    http_backend: ModuleType, sdk_provider: str, monkeypatch: pytest.MonkeyPatch
):
    outbound: Final = asyncio.Queue()

    def chunk(delta: dict, finish: str | None) -> bytes:
        body: Final = {
            "id": "chatcmpl-smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "sdk-compat",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body)}\n\n".encode()

    def respond(request: _Request) -> httpx.Response | SDKResponse:
        outbound.put_nowait(json.loads(request.content))
        usage: Final = {
            "id": "chatcmpl-smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "sdk-compat",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        content: Final = b"".join(
            (
                chunk({"role": "assistant", "content": "Hel"}, None),
                chunk({"content": "lo"}, "stop"),
                f"data: {json.dumps(usage)}\n\n".encode(),
                b"data: [DONE]\n\n",
            )
        )
        return http_backend.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    async with http_backend.AsyncClient(transport=http_backend.MockTransport(respond)) as http_client:
        monkeypatch.setattr(litellm, "aclient_session", http_client)
        stream: Final = await litellm.acompletion(
            model=f"{sdk_provider}/sdk-compat",
            api_key=f"{sdk_provider}-key",
            api_base=f"https://{sdk_provider}.example/v1",
            messages=[{"role": "user", "content": "smoke-stream-request"}],
            stream=True,
            num_retries=0,
            max_retries=0,
        )
        chunks: Final = []

        async def drain() -> None:
            async for part in stream:
                chunks.append(part)

        await asyncio.wait_for(drain(), timeout=10)
        request: Final = await asyncio.wait_for(outbound.get(), timeout=10)
        assert request["stream"] is True
        assert outbound.empty()
        assert (
            "".join(part.choices[0].delta.content or "" for part in chunks if part.choices and part.choices[0].delta)
            == "Hello"
        )
        last_finish: Final = next(
            part.choices[0].finish_reason for part in reversed(chunks) if part.choices and part.choices[0].finish_reason
        )
        assert last_finish == "stop"


@pytest.mark.asyncio
async def test_acompletion_streams_tool_call_arguments_over_injected_transport(http_backend: ModuleType):
    outbound: Final = asyncio.Queue()
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    def chunk(delta: dict, finish: str | None) -> bytes:
        body: Final = {
            "id": "chatcmpl-smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-5.6",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body)}\n\n".encode()

    def respond(request: _Request) -> httpx.Response | SDKResponse:
        assert request.extensions["timeout"] == {"connect": 1, "read": None, "write": 2, "pool": 3}
        outbound.put_nowait(json.loads(request.content))
        content: Final = b"".join(
            (
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                    None,
                ),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}, None),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}, "tool_calls"),
                b"data: [DONE]\n\n",
            )
        )
        return http_backend.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    async with http_backend.AsyncClient(transport=http_backend.MockTransport(respond)) as http_client:
        client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client)
        messages: Final = [{"role": "user", "content": "weather in Paris"}]
        stream: Final = await litellm.acompletion(
            model="openai/gpt-4o",
            api_key="transport-only",
            client=client,
            messages=messages,
            tools=tools,
            stream=True,
            timeout=http_backend.Timeout(connect=1, read=None, write=2, pool=3),
            num_retries=0,
            max_retries=0,
        )
        chunks: Final = []

        async def drain() -> None:
            async for part in stream:
                chunks.append(part)

        await asyncio.wait_for(drain(), timeout=10)
        request: Final = await asyncio.wait_for(outbound.get(), timeout=10)
        assert request["stream"] is True
        assert request["tools"][0]["function"]["name"] == "get_weather"
        assert outbound.empty()
        rebuilt: Final = litellm.stream_chunk_builder(chunks, messages=messages)
        tool_call: Final = rebuilt.choices[0].message.tool_calls[0]
        assert tool_call.id == "call-1"
        assert tool_call.function.name == "get_weather"
        assert json.loads(tool_call.function.arguments) == {"city": "Paris"}
        assert rebuilt.choices[0].finish_reason == "tool_calls"


_PROVIDER_HEADERS: Final = {"x-request-id": "req_openai", "x-ratelimit-remaining-requests": "41"}


def _image_generation_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"created": 1, "data": [{"b64_json": "abc"}]}, headers=_PROVIDER_HEADERS
        )
    )


def _speech_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=b"audio-bytes", headers={**_PROVIDER_HEADERS, "content-type": "audio/mpeg"}
        )
    )


def _assert_provider_headers_recorded(response) -> None:
    assert response._hidden_params["headers"]["x-request-id"] == "req_openai"
    assert response._hidden_params["additional_headers"]["llm_provider-x-request-id"] == "req_openai"
    assert response._hidden_params["additional_headers"]["x-ratelimit-remaining-requests"] == "41"


def _image_generation_kwargs() -> dict:
    return {
        "model": "gpt-image-2",
        "prompt": "a cat",
        "timeout": 10,
        "optional_params": {},
        "logging_obj": Mock(),
        "api_key": "transport-only",
        "model_response": ImageResponse(),
    }


def test_image_generation_records_provider_response_headers():
    with httpx.Client(transport=_image_generation_transport()) as http_client:
        response = OpenAIChatCompletion().image_generation(
            client=OpenAI(api_key="transport-only", http_client=http_client), **_image_generation_kwargs()
        )

    _assert_provider_headers_recorded(response)


@pytest.mark.asyncio
async def test_aimage_generation_records_provider_response_headers():
    async with httpx.AsyncClient(transport=_image_generation_transport()) as http_client:
        response = await OpenAIChatCompletion().image_generation(
            client=AsyncOpenAI(api_key="transport-only", http_client=http_client),
            aimg_generation=True,
            **_image_generation_kwargs(),
        )

    _assert_provider_headers_recorded(response)


def _audio_speech_kwargs() -> dict:
    return {
        "model": "gpt-4o-mini-tts",
        "input": "hello",
        "voice": "alloy",
        "optional_params": {},
        "api_key": "transport-only",
        "api_base": None,
        "organization": None,
        "project": None,
        "max_retries": 0,
        "timeout": 10,
        "logging_obj": Mock(),
    }


def test_audio_speech_records_provider_response_headers():
    with httpx.Client(transport=_speech_transport()) as http_client:
        response = OpenAIChatCompletion().audio_speech(
            client=OpenAI(api_key="transport-only", http_client=http_client), **_audio_speech_kwargs()
        )

    _assert_provider_headers_recorded(response)


@pytest.mark.asyncio
async def test_async_audio_speech_records_provider_response_headers():
    async with httpx.AsyncClient(transport=_speech_transport()) as http_client:
        response = await OpenAIChatCompletion().audio_speech(
            client=AsyncOpenAI(api_key="transport-only", http_client=http_client),
            aspeech=True,
            **_audio_speech_kwargs(),
        )

    _assert_provider_headers_recorded(response)


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function")
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm globals to their true defaults before each test and
    restores them afterward, so tests don't leak side effects.
    Works safely under pytest-xdist parallel execution.
    """
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in ("pre_call_rules", "post_call_rules"):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "pre_call_rules",
        "post_call_rules",
    ):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])
    for attr, default_val in _SCALAR_DEFAULTS.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, default_val)
    yield
    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    _invalidate_model_cost_lowercase_map()


_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "num_retries_per_request": getattr(litellm, "num_retries_per_request", None),
    "request_timeout": getattr(litellm, "request_timeout", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "default_fallbacks": getattr(litellm, "default_fallbacks", None),
    "enable_azure_ad_token_refresh": getattr(litellm, "enable_azure_ad_token_refresh", None),
    "tag_budget_config": getattr(litellm, "tag_budget_config", None),
    "model_cost": getattr(litellm, "model_cost", None),
    "token_counter": getattr(litellm, "token_counter", None),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
}


@pytest.fixture(scope="module")
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield


ASSISTANT_INSTRUCTIONS = (
    "You are a personal math tutor. When asked a question, write and run Python code to answer the question."
)

ASSISTANT_ID = "asst_test"

THREAD_ID = "thread_test"

MESSAGE_ID = "msg_test"

RUN_ID = "run_test"


def _assistant(**overrides):
    data = {
        "id": ASSISTANT_ID,
        "object": "assistant",
        "created_at": 1,
        "name": "Math Tutor",
        "description": None,
        "model": "gpt-4.1",
        "instructions": ASSISTANT_INSTRUCTIONS,
        "tools": [],
        "metadata": {},
        "top_p": 1.0,
        "temperature": 1.0,
        "response_format": "auto",
    }
    data.update(overrides)
    return Assistant(**data)


def _thread(thread_id=THREAD_ID):
    return Thread(id=thread_id, object="thread", created_at=1, metadata={})


def _message(thread_id=THREAD_ID):
    return Message(
        id=MESSAGE_ID,
        object="thread.message",
        created_at=1,
        thread_id=thread_id,
        role="user",
        content=[
            {
                "type": "text",
                "text": {"value": "Hey, how's it going?", "annotations": []},
            }
        ],
        assistant_id=None,
        run_id=None,
        attachments=[],
        metadata={},
        status="completed",
    )


def _run(thread_id=THREAD_ID, assistant_id=ASSISTANT_ID):
    return Run(
        id=RUN_ID,
        object="thread.run",
        created_at=1,
        assistant_id=assistant_id,
        thread_id=thread_id,
        status="completed",
        started_at=1,
        expires_at=None,
        cancelled_at=None,
        failed_at=None,
        completed_at=1,
        last_error=None,
        model="gpt-4.1",
        instructions=ASSISTANT_INSTRUCTIONS,
        tools=[],
        metadata={},
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        required_action=None,
        incomplete_details=None,
        temperature=1.0,
        top_p=1.0,
        max_prompt_tokens=None,
        max_completion_tokens=None,
        truncation_strategy={"type": "auto", "last_messages": None},
        response_format="auto",
        tool_choice="auto",
        parallel_tool_calls=True,
    )


def _sync_page(data):
    first_id = data[0].id if data else None
    return SyncCursorPage(
        data=data,
        object="list",
        first_id=first_id,
        last_id=first_id,
        has_more=False,
    )


def _async_page(data):
    first_id = data[0].id if data else None
    return AsyncCursorPage(
        data=data,
        object="list",
        first_id=first_id,
        last_id=first_id,
        has_more=False,
    )


class _FakeAssistantEventHandler(AssistantEventHandler):
    def until_done(self):
        return None


class _FakeAsyncAssistantEventHandler(AsyncAssistantEventHandler):
    async def until_done(self):
        return None


class _FakeAssistantStream:
    def __enter__(self):
        return _FakeAssistantEventHandler()

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeAsyncAssistantStream:
    async def __aenter__(self):
        return _FakeAsyncAssistantEventHandler()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _SyncAssistants:
    def list(self, **_kwargs):
        return _sync_page([_assistant()])

    def create(self, **kwargs):
        return _assistant(**kwargs)

    def delete(self, assistant_id):
        return AssistantDeleted(id=assistant_id, object="assistant.deleted", deleted=True)


class _AsyncAssistants:
    async def list(self, **_kwargs):
        return _async_page([_assistant()])

    async def create(self, **kwargs):
        return _assistant(**kwargs)

    async def delete(self, assistant_id):
        return AssistantDeleted(id=assistant_id, object="assistant.deleted", deleted=True)


class _SyncMessages:
    def create(self, thread_id, **_kwargs):
        return _message(thread_id)

    def list(self, thread_id):
        return _sync_page([_message(thread_id)])


class _AsyncMessages:
    async def create(self, thread_id, **_kwargs):
        return _message(thread_id)

    async def list(self, thread_id):
        return _async_page([_message(thread_id)])


class _SyncRuns:
    def create_and_poll(self, thread_id, assistant_id, **_kwargs):
        return _run(thread_id=thread_id, assistant_id=assistant_id)

    def stream(self, **_kwargs):
        return _FakeAssistantStream()


class _AsyncRuns:
    async def create_and_poll(self, thread_id, assistant_id, **_kwargs):
        return _run(thread_id=thread_id, assistant_id=assistant_id)

    def stream(self, **_kwargs):
        return _FakeAsyncAssistantStream()


class _SyncThreads:
    def __init__(self):
        self.messages = _SyncMessages()
        self.runs = _SyncRuns()

    def create(self, **_kwargs):
        return _thread()

    def retrieve(self, thread_id):
        return _thread(thread_id)


class _AsyncThreads:
    def __init__(self):
        self.messages = _AsyncMessages()
        self.runs = _AsyncRuns()

    async def create(self, **_kwargs):
        return _thread()

    async def retrieve(self, thread_id):
        return _thread(thread_id)


class _FakeBeta:
    def __init__(self, *, async_mode):
        self.assistants = _AsyncAssistants() if async_mode else _SyncAssistants()
        self.threads = _AsyncThreads() if async_mode else _SyncThreads()


class _FakeAssistantClient:
    def __init__(self, *, async_mode):
        self.beta = _FakeBeta(async_mode=async_mode)


@pytest.fixture
def assistant_client(sync_mode):
    return _FakeAssistantClient(async_mode=not sync_mode)


def _request_data(provider, assistant_client, **kwargs):
    data = {"custom_llm_provider": provider, "client": assistant_client, **kwargs}
    if provider == "azure":
        data.update(
            {
                "api_version": "2024-02-15-preview",
                "api_base": "https://example.azure.test",
                "api_key": "test-key",
            }
        )
    return data


@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_get_assistants(provider, sync_mode, assistant_client):
    data = _request_data(provider, assistant_client)

    if sync_mode:
        assistants = litellm.get_assistants(**data)
        assert isinstance(assistants, SyncCursorPage)
    else:
        assistants = await litellm.aget_assistants(**data)
        assert isinstance(assistants, AsyncCursorPage)


@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.parametrize("provider", ["azure", "openai"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio()
async def test_create_delete_assistants(provider, sync_mode, assistant_client):
    data = _request_data(
        provider,
        assistant_client,
        model="gpt-4.1",
        instructions=ASSISTANT_INSTRUCTIONS,
        name="Math Tutor",
        tools=[{"type": "code_interpreter"}],
    )

    if sync_mode:
        assistant = litellm.create_assistants(**data)
        assert isinstance(assistant, Assistant)
        assert assistant.instructions == ASSISTANT_INSTRUCTIONS
        assert assistant.id is not None

        response = litellm.delete_assistant(
            **_request_data(
                provider,
                assistant_client,
                assistant_id=assistant.id,
            )
        )
        assert response.id == assistant.id
    else:
        assistant = await litellm.acreate_assistants(**data)
        assert isinstance(assistant, Assistant)
        assert assistant.instructions == ASSISTANT_INSTRUCTIONS
        assert assistant.id is not None

        response = await litellm.adelete_assistant(
            **_request_data(
                provider,
                assistant_client,
                assistant_id=assistant.id,
            )
        )
        assert response.id == assistant.id


async def _create_thread_litellm(sync_mode, provider, assistant_client) -> Thread:
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    data = _request_data(provider, assistant_client, message=[message])

    if sync_mode:
        new_thread = create_thread(**data)
    else:
        new_thread = await litellm.acreate_thread(**data)

    assert isinstance(new_thread, Thread)
    return new_thread


@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_create_thread_litellm(sync_mode, provider, assistant_client):
    await _create_thread_litellm(sync_mode, provider, assistant_client)


@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_get_thread_litellm(provider, sync_mode, assistant_client):
    new_thread = await _create_thread_litellm(sync_mode, provider, assistant_client)
    data = _request_data(provider, assistant_client, thread_id=new_thread.id)

    if sync_mode:
        received_thread = get_thread(**data)
    else:
        received_thread = await litellm.aget_thread(**data)

    assert isinstance(received_thread, Thread)


@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_add_message_litellm(sync_mode, provider, assistant_client):
    new_thread = await _create_thread_litellm(sync_mode, provider, assistant_client)
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    data = _request_data(provider, assistant_client, thread_id=new_thread.id, **message)

    if sync_mode:
        added_message = litellm.add_message(**data)
    else:
        added_message = await litellm.a_add_message(**data)

    assert isinstance(added_message, Message)


@pytest.mark.usefixtures("_vcr_outcome_gate", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.parametrize("provider", ["azure", "openai"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("is_streaming", [True, False])
@pytest.mark.asyncio
async def test_aarun_thread_litellm(sync_mode, provider, is_streaming, assistant_client):
    get_assistants_data = _request_data(provider, assistant_client)
    if sync_mode:
        assistants = litellm.get_assistants(**get_assistants_data)
    else:
        assistants = await litellm.aget_assistants(**get_assistants_data)

    assistant_id = assistants.data[0].id
    new_thread = await _create_thread_litellm(sync_mode, provider, assistant_client)
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    thread_data = _request_data(provider, assistant_client, thread_id=new_thread.id)
    message_data = _request_data(provider, assistant_client, thread_id=new_thread.id, **message)

    if sync_mode:
        added_message = litellm.add_message(**message_data)
        assert isinstance(added_message, Message)

        if is_streaming:
            run = litellm.run_thread_stream(assistant_id=assistant_id, **thread_data)
            with run as run:
                assert isinstance(run, AssistantEventHandler)
                run.until_done()
        else:
            run = litellm.run_thread(assistant_id=assistant_id, stream=is_streaming, **thread_data)
            assert run.status == "completed"
            messages = litellm.get_messages(**thread_data)
            assert isinstance(messages.data[0], Message)
    else:
        added_message = await litellm.a_add_message(**message_data)
        assert isinstance(added_message, Message)

        if is_streaming:
            run = litellm.arun_thread_stream(assistant_id=assistant_id, **thread_data)
            async with run as run:
                assert isinstance(run, AsyncAssistantEventHandler)
                await run.until_done()
        else:
            run = await litellm.arun_thread(
                custom_llm_provider=provider,
                thread_id=new_thread.id,
                assistant_id=assistant_id,
                client=assistant_client,
            )
            assert run.status == "completed"
            messages = await litellm.aget_messages(**thread_data)
            assert isinstance(messages.data[0], Message)


@pytest.mark.asyncio
async def test_openai_prediction_param_mock():
    """
    Tests that prediction parameter is correctly passed to the API
    """
    litellm.set_verbose = True

    code = """
    /// <summary>
    /// Represents a user with a first name, last name, and username.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user's first name.
        /// </summary>
        public string FirstName { get; set; }

        /// <summary>
        /// Gets or sets the user's last name.
        /// </summary>
        public string LastName { get; set; }

        /// <summary>
        /// Gets or sets the user's username.
        /// </summary>
        public string Username { get; set; }
    }
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": "Replace the Username property with an Email property. Respond only with code, and with no markdown formatting.",
                    },
                    {"role": "user", "content": code},
                ],
                prediction={"type": "content", "content": code},
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the prediction parameter
        assert "prediction" in request_body
        # verify prediction is correctly sent to the API
        assert request_body["prediction"] == {"type": "content", "content": code}


@patch("litellm.main.openai_chat_completions._get_openai_client")
def test_openai_max_retries_0(mock_get_openai_client):
    import litellm

    mock_get_openai_client.return_value.chat.completions.with_raw_response.create.return_value.headers = {}
    mock_get_openai_client.return_value.chat.completions.with_raw_response.create.return_value.parse.return_value = (
        ModelResponse(choices=[{"message": {"role": "assistant", "content": "Hello"}}])
    )
    litellm.set_verbose = True
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        max_retries=0,
        api_key="fake-key",
    )

    mock_get_openai_client.assert_called_once()
    assert mock_get_openai_client.call_args.kwargs["max_retries"] == 0
    assert response.choices[0].message.content == "Hello"


@patch("litellm.main.openai_chat_completions._get_openai_client")
def test_openai_image_generation_forwards_organization(mock_get_openai_client):
    """Ensure organization flows to OpenAI client for image generation."""

    class _DummyRawImages:
        def generate(self, **kwargs):  # type: ignore
            class _Resp:
                def model_dump(self_inner):  # minimal OpenAI ImagesResponse shape
                    return {
                        "created": 123,
                        "data": [{"url": "http://example.com/image.png"}],
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                    }

            class _RawResp:
                headers = {}

                def parse(self_inner):
                    return _Resp()

            return _RawResp()

    class _DummyImages:
        with_raw_response = _DummyRawImages()

    class _DummyClient:
        def __init__(self):
            self.api_key = "sk-test"

            class _BaseURL:
                _uri_reference = "https://api.openai.com/v1"

            self._base_url = _BaseURL()
            self.images = _DummyImages()

    mock_get_openai_client.return_value = _DummyClient()

    org = "org_test_123"
    resp = litellm.image_generation(
        model="gpt-image-1",
        prompt="A cute baby sea otter",
        organization=org,
    )

    # Assert organization forwarded into OpenAI client factory
    assert mock_get_openai_client.call_args.kwargs.get("organization") == org

    # Basic sanity on response shape
    assert hasattr(resp, "data") and len(resp.data) == 1


def test_openai_chat_completion_streaming_handler_reasoning_content():
    from litellm.llms.openai.chat.gpt_transformation import (
        OpenAIChatCompletionStreamingHandler,
    )
    from unittest.mock import MagicMock

    streaming_handler = OpenAIChatCompletionStreamingHandler(
        streaming_response=MagicMock(),
        sync_stream=True,
    )
    response = streaming_handler.chunk_parser(
        chunk={
            "id": "e89b6501-8ac2-464c-9550-7cd3daf94350",
            "object": "chat.completion.chunk",
            "created": 1741037890,
            "model": "deepseek-reasoner",
            "system_fingerprint": "fp_5417b77867_prod0225",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": None, "reasoning_content": "."},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
    )

    assert response.choices[0].delta.reasoning_content == "."


@pytest.mark.asyncio
async def test_openai_safety_identifier_parameter():
    """Test that safety_identifier parameter is correctly passed to the OpenAI API."""
    from openai import AsyncOpenAI

    litellm.set_verbose = True
    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                safety_identifier="user_code_123456",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the safety_identifier parameter
        assert "safety_identifier" in request_body
        # Verify safety_identifier is correctly sent to the API
        assert request_body["safety_identifier"] == "user_code_123456"


def test_openai_safety_identifier_parameter_sync():
    """Test that safety_identifier parameter is correctly passed to the OpenAI API."""
    from openai import OpenAI

    litellm.set_verbose = True
    client = OpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                safety_identifier="user_code_123456",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the safety_identifier parameter
        assert "safety_identifier" in request_body
        # Verify safety_identifier is correctly sent to the API
        assert request_body["safety_identifier"] == "user_code_123456"


@pytest.mark.asyncio
async def test_openai_service_tier_parameter():
    """Test that service_tier parameter is correctly passed to the OpenAI API."""
    from openai import AsyncOpenAI

    litellm.set_verbose = True
    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                service_tier="priority",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the service_tier parameter
        assert "service_tier" in request_body, "service_tier should be in request body"
        # Verify service_tier is correctly sent to the API
        assert (
            request_body["service_tier"] == "priority"
        ), "service_tier should be 'priority'"


def test_openai_service_tier_parameter_sync():
    """Test that service_tier parameter is correctly passed to the OpenAI API."""
    from openai import OpenAI

    litellm.set_verbose = True
    client = OpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                service_tier="priority",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        # Verify the request contains the service_tier parameter
        assert "service_tier" in request_body, "service_tier should be in request body"
        # Verify service_tier is correctly sent to the API
        assert (
            request_body["service_tier"] == "priority"
        ), "service_tier should be 'priority'"


def test_responses_gpt54_with_xhigh_reasoning():
    """
    Ensure chat->responses bridge sends the correct request payload for
    openai/responses/gpt-5.4 with reasoning_effort="xhigh".
    """
    with patch("litellm.responses") as mock_responses:
        mock_responses.side_effect = RuntimeError("stop_after_request_build")

        with pytest.raises(litellm.APIConnectionError):
            litellm.completion(
                model="openai/responses/gpt-5.4",
                messages=[{"role": "user", "content": "What is 2+2?"}],
                reasoning_effort="xhigh",
                max_tokens=100,
            )

        mock_responses.assert_called_once()
        request_body = mock_responses.call_args.kwargs

        assert request_body["model"] == "openai/gpt-5.4"

        assert request_body["reasoning"] == {"effort": "xhigh"}
