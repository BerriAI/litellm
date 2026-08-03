import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from litellm.llms.openai.openai import OpenAIAssistantsAPI
from litellm.types.llms.openai import Thread

_THREAD_PAYLOAD = {
    "id": "thread_123",
    "object": "thread",
    "created_at": 1700000000,
    "metadata": {"origin": "unit-test"},
    "unexpected_upstream_field": "kept",
}

_COMMON_ARGS = {
    "api_key": "test-key",
    "api_base": "https://api.openai.com/v1",
    "timeout": 60.0,
    "max_retries": 2,
    "organization": None,
}


def _handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_THREAD_PAYLOAD)


def _async_client() -> AsyncOpenAI:
    """A real AsyncOpenAI wired to a mock transport, so the SDK's own response parsing
    runs and the handler under test receives exactly what production would."""
    return AsyncOpenAI(
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )


def _sync_client() -> OpenAI:
    return OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
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
