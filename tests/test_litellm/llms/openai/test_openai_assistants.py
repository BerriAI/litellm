import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from litellm.llms.openai.openai import OpenAIAssistantsAPI
from litellm.types.llms.openai import OpenAIMessage, Thread

_THREAD_PAYLOAD = {
    "id": "thread_123",
    "object": "thread",
    "created_at": 1700000000,
    "metadata": {"origin": "unit-test"},
    "unexpected_upstream_field": "kept",
}

_MESSAGE_PAYLOAD = {
    "id": "msg_123",
    "object": "thread.message",
    "created_at": 1700000000,
    "thread_id": "thread_123",
    "role": "assistant",
    "status": "in_progress",
    "content": [{"type": "text", "text": {"value": "hi", "annotations": []}}],
    "metadata": {"origin": "unit-test"},
    "run_id": "run_123",
    "assistant_id": "asst_123",
    "unexpected_upstream_field": "kept",
}

_COMMON_ARGS = {
    "api_key": "test-key",
    "api_base": "https://api.openai.com/v1",
    "timeout": 60.0,
    "max_retries": 2,
    "organization": None,
}


def _transport(payload: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def _async_client(payload: dict = _THREAD_PAYLOAD) -> AsyncOpenAI:
    """A real AsyncOpenAI wired to a mock transport, so the SDK's own response parsing
    runs and the handler under test receives exactly what production would."""
    return AsyncOpenAI(
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=_transport(payload)),
    )


def _sync_client(payload: dict = _THREAD_PAYLOAD) -> OpenAI:
    return OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=_transport(payload)),
    )


def _assert_thread(thread: object) -> None:
    assert isinstance(thread, Thread)
    assert thread.id == "thread_123"
    assert thread.created_at == 1700000000
    assert thread.object == "thread"
    assert thread.metadata == {"origin": "unit-test"}
    # LiteLLM's Thread declares its own fields, so anything the provider adds on top
    # is dropped rather than carried through.
    assert "unexpected_upstream_field" not in thread.model_dump()


@pytest.mark.asyncio
async def test_async_thread_responses_preserve_declared_fields():
    api = OpenAIAssistantsAPI()

    created = await api.async_create_thread(
        metadata={"origin": "unit-test"},
        messages=None,
        client=_async_client(),
        **_COMMON_ARGS,
    )
    retrieved = await api.async_get_thread(
        thread_id="thread_123",
        client=_async_client(),
        **_COMMON_ARGS,
    )

    _assert_thread(created)
    _assert_thread(retrieved)


def _assert_message(result: object) -> None:
    assert isinstance(result, OpenAIMessage)
    assert result.id == "msg_123"
    assert result.thread_id == "thread_123"
    assert result.role == "assistant"
    assert result.metadata == {"origin": "unit-test"}
    assert result.model_dump()["unexpected_upstream_field"] == "kept"


@pytest.mark.asyncio
async def test_a_add_message_preserves_fields_and_defaults_status():
    api = OpenAIAssistantsAPI()

    result = await api.a_add_message(
        thread_id="thread_123",
        message_data={"role": "user", "content": "hi"},
        client=_async_client(_MESSAGE_PAYLOAD),
        **_COMMON_ARGS,
    )

    _assert_message(result)
    assert result.status == "in_progress"

    without_status = {k: v for k, v in _MESSAGE_PAYLOAD.items() if k != "status"}
    defaulted = await api.a_add_message(
        thread_id="thread_123",
        message_data={"role": "user", "content": "hi"},
        client=_async_client(without_status),
        **_COMMON_ARGS,
    )

    assert defaulted.status == "completed"


def test_sync_add_message_preserves_fields_and_defaults_status():
    api = OpenAIAssistantsAPI()

    result = api.add_message(
        thread_id="thread_123",
        message_data={"role": "user", "content": "hi"},
        client=_sync_client(_MESSAGE_PAYLOAD),
        **_COMMON_ARGS,
    )

    _assert_message(result)
    assert result.status == "in_progress"

    without_status = {k: v for k, v in _MESSAGE_PAYLOAD.items() if k != "status"}
    defaulted = api.add_message(
        thread_id="thread_123",
        message_data={"role": "user", "content": "hi"},
        client=_sync_client(without_status),
        **_COMMON_ARGS,
    )

    assert defaulted.status == "completed"


def test_sync_thread_responses_preserve_declared_fields():
    api = OpenAIAssistantsAPI()

    created = api.create_thread(
        metadata={"origin": "unit-test"},
        messages=None,
        client=_sync_client(),
        **_COMMON_ARGS,
    )
    retrieved = api.get_thread(
        thread_id="thread_123",
        client=_sync_client(),
        **_COMMON_ARGS,
    )

    _assert_thread(created)
    _assert_thread(retrieved)
