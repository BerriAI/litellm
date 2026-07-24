import os
import sys

import pytest
from dotenv import load_dotenv
from openai.types.beta.assistant import Assistant
from openai.types.beta.assistant_deleted import AssistantDeleted

load_dotenv()
sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import create_thread, get_thread
from litellm.llms.openai.openai import (
    AssistantEventHandler,
    AsyncAssistantEventHandler,
    AsyncCursorPage,
    MessageData,
    OpenAIMessage as Message,
    Run,
    SyncCursorPage,
    Thread,
)

ASSISTANT_INSTRUCTIONS = (
    "You are a personal math tutor. When asked a question, write and run Python "
    "code to answer the question."
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
        return AssistantDeleted(
            id=assistant_id, object="assistant.deleted", deleted=True
        )


class _AsyncAssistants:
    async def list(self, **_kwargs):
        return _async_page([_assistant()])

    async def create(self, **kwargs):
        return _assistant(**kwargs)

    async def delete(self, assistant_id):
        return AssistantDeleted(
            id=assistant_id, object="assistant.deleted", deleted=True
        )


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


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_create_thread_litellm(sync_mode, provider, assistant_client):
    await _create_thread_litellm(sync_mode, provider, assistant_client)


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


@pytest.mark.parametrize("provider", ["azure", "openai"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("is_streaming", [True, False])
@pytest.mark.asyncio
async def test_aarun_thread_litellm(
    sync_mode, provider, is_streaming, assistant_client
):
    get_assistants_data = _request_data(provider, assistant_client)
    if sync_mode:
        assistants = litellm.get_assistants(**get_assistants_data)
    else:
        assistants = await litellm.aget_assistants(**get_assistants_data)

    assistant_id = assistants.data[0].id
    new_thread = await _create_thread_litellm(sync_mode, provider, assistant_client)
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    thread_data = _request_data(provider, assistant_client, thread_id=new_thread.id)
    message_data = _request_data(
        provider, assistant_client, thread_id=new_thread.id, **message
    )

    if sync_mode:
        added_message = litellm.add_message(**message_data)
        assert isinstance(added_message, Message)

        if is_streaming:
            run = litellm.run_thread_stream(assistant_id=assistant_id, **thread_data)
            with run as run:
                assert isinstance(run, AssistantEventHandler)
                run.until_done()
        else:
            run = litellm.run_thread(
                assistant_id=assistant_id, stream=is_streaming, **thread_data
            )
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
