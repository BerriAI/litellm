import asyncio
import importlib
import json
import os
from typing import Final
from unittest.mock import Mock

import httpx
import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from openai.types.beta.assistant import Assistant
from openai.types.beta.assistant_deleted import AssistantDeleted

import litellm
from litellm import create_thread, get_thread
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.openai.openai import (
    AssistantEventHandler,
    AsyncAssistantEventHandler,
    AsyncCursorPage,
    MessageData,
    OpenAIChatCompletion,
    Run,
    SyncCursorPage,
    Thread,
)
from litellm.llms.openai.openai import (
    OpenAIMessage as Message,
)
from litellm.types.utils import ImageResponse
from litellm.utils import _invalidate_model_cost_lowercase_map
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from tests.fake_openai_endpoint import ensure_fake_openai_endpoint


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


@pytest.mark.asyncio
async def test_acompletion_returns_json_reply_over_injected_transport():
    outbound: Final = asyncio.Queue()

    def respond(request: httpx.Request) -> httpx.Response:
        outbound.put_nowait(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-smoke",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5.6",
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

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client)
        response: Final = await asyncio.wait_for(
            litellm.acompletion(
                model="openai/gpt-5.6",
                api_key="transport-only",
                client=client,
                messages=[{"role": "user", "content": "smoke-json-request"}],
                num_retries=0,
                max_retries=0,
            ),
            timeout=10,
        )
        request: Final = await asyncio.wait_for(outbound.get(), timeout=10)
        assert request["model"] == "gpt-5.6"
        assert request["messages"] == [{"role": "user", "content": "smoke-json-request"}]
        assert not request.get("stream")
        assert outbound.empty()
        assert response.choices[0].message.content == "smoke-json-reply"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_acompletion_streams_text_deltas_over_injected_transport():
    outbound: Final = asyncio.Queue()

    def chunk(delta: dict, finish: str | None) -> bytes:
        body: Final = {
            "id": "chatcmpl-smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-5.6",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body)}\n\n".encode()

    def respond(request: httpx.Request) -> httpx.Response:
        outbound.put_nowait(json.loads(request.content))
        usage: Final = {
            "id": "chatcmpl-smoke",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-5.6",
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
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client)
        stream: Final = await litellm.acompletion(
            model="openai/gpt-5.6",
            api_key="transport-only",
            client=client,
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
async def test_acompletion_streams_tool_call_arguments_over_injected_transport():
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

    def respond(request: httpx.Request) -> httpx.Response:
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
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client)
        messages: Final = [{"role": "user", "content": "weather in Paris"}]
        stream: Final = await litellm.acompletion(
            model="openai/gpt-4o",
            api_key="transport-only",
            client=client,
            messages=messages,
            tools=tools,
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


@pytest.fixture(scope="session")
def fake_openai_endpoint():
    ensure_fake_openai_endpoint()
    yield


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


load_dotenv()


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


@pytest.mark.usefixtures("_vcr_outcome_gate", "fake_openai_endpoint", "isolate_litellm_state", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "fake_openai_endpoint", "isolate_litellm_state", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "fake_openai_endpoint", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_create_thread_litellm(sync_mode, provider, assistant_client):
    await _create_thread_litellm(sync_mode, provider, assistant_client)


@pytest.mark.usefixtures("_vcr_outcome_gate", "fake_openai_endpoint", "isolate_litellm_state", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "fake_openai_endpoint", "isolate_litellm_state", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "fake_openai_endpoint", "isolate_litellm_state", "setup_and_teardown")
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
