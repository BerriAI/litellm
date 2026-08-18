import httpx
import pytest
from openai import AsyncAzureOpenAI, AzureOpenAI

from litellm.llms.azure.assistants import AzureAssistantsAPI
from litellm.types.llms.openai import OpenAIMessage, Thread

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

_THREAD_PAYLOAD = {
    "id": "thread_123",
    "object": "thread",
    "created_at": 1700000000,
    "metadata": {"origin": "unit-test"},
    "unexpected_upstream_field": "kept",
}

_COMMON_ARGS = {
    "api_key": "test-key",
    "api_base": "https://test.openai.azure.com",
    "api_version": "2024-05-01-preview",
    "azure_ad_token": None,
    "timeout": 60.0,
    "max_retries": 2,
}


def _client(payload: dict) -> AsyncAzureOpenAI:
    """A real AsyncAzureOpenAI wired to a mock transport, so the SDK's own response
    parsing runs and the handler under test receives exactly what production would."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return AsyncAzureOpenAI(
        api_key="test-key",
        api_version="2024-05-01-preview",
        azure_endpoint="https://test.openai.azure.com",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _sync_client(payload: dict) -> AzureOpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return AzureOpenAI(
        api_key="test-key",
        api_version="2024-05-01-preview",
        azure_endpoint="https://test.openai.azure.com",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_a_add_message_preserves_upstream_message_fields():
    """The returned message is rebuilt from the upstream one, so every field the
    provider sent (including ones the SDK model does not declare) must survive."""
    result = await AzureAssistantsAPI().a_add_message(
        thread_id="thread_123",
        message_data={"role": "user", "content": "hi"},
        client=_client(_MESSAGE_PAYLOAD),
        **_COMMON_ARGS,
    )

    assert isinstance(result, OpenAIMessage)
    assert result.id == "msg_123"
    assert result.thread_id == "thread_123"
    assert result.status == "in_progress"
    assert result.role == "assistant"
    assert result.metadata == {"origin": "unit-test"}
    assert result.model_dump()["unexpected_upstream_field"] == "kept"


@pytest.mark.asyncio
async def test_a_add_message_defaults_missing_status_to_completed():
    """Some deployments omit `status` on the created message; it is filled in before
    the message is handed back so callers never see a status-less message."""
    payload = {k: v for k, v in _MESSAGE_PAYLOAD.items() if k != "status"}

    result = await AzureAssistantsAPI().a_add_message(
        thread_id="thread_123",
        message_data={"role": "user", "content": "hi"},
        client=_client(payload),
        **_COMMON_ARGS,
    )

    assert result.status == "completed"
    assert result.id == "msg_123"
    assert result.metadata == {"origin": "unit-test"}


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
async def test_thread_responses_preserve_declared_fields():
    api = AzureAssistantsAPI()

    created = await api.async_create_thread(
        metadata={"origin": "unit-test"},
        messages=None,
        client=_client(_THREAD_PAYLOAD),
        **_COMMON_ARGS,
    )
    retrieved = await api.async_get_thread(
        thread_id="thread_123",
        client=_client(_THREAD_PAYLOAD),
        **_COMMON_ARGS,
    )

    _assert_thread(created)
    _assert_thread(retrieved)


def test_sync_thread_responses_preserve_declared_fields():
    api = AzureAssistantsAPI()

    created = api.create_thread(
        metadata={"origin": "unit-test"},
        messages=None,
        client=_sync_client(_THREAD_PAYLOAD),
        **_COMMON_ARGS,
    )
    retrieved = api.get_thread(
        thread_id="thread_123",
        client=_sync_client(_THREAD_PAYLOAD),
        **_COMMON_ARGS,
    )

    _assert_thread(created)
    _assert_thread(retrieved)


def test_sync_add_message_preserves_fields_and_defaults_status():
    api = AzureAssistantsAPI()

    result = api.add_message(
        thread_id="thread_123",
        message_data={"role": "user", "content": "hi"},
        client=_sync_client(_MESSAGE_PAYLOAD),
        **_COMMON_ARGS,
    )

    assert isinstance(result, OpenAIMessage)
    assert result.id == "msg_123"
    assert result.status == "in_progress"
    assert result.metadata == {"origin": "unit-test"}
    assert result.model_dump()["unexpected_upstream_field"] == "kept"

    without_status = {k: v for k, v in _MESSAGE_PAYLOAD.items() if k != "status"}
    defaulted = api.add_message(
        thread_id="thread_123",
        message_data={"role": "user", "content": "hi"},
        client=_sync_client(without_status),
        **_COMMON_ARGS,
    )

    assert defaulted.status == "completed"
