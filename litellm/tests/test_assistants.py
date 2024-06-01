# What is this?
## Unit Tests for OpenAI Assistants API
import sys, os, json
import traceback
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest, logging, asyncio
import litellm
from litellm import create_thread, get_thread
from litellm.llms.openai import (
    OpenAIAssistantsAPI,
    MessageData,
    Thread,
    OpenAIMessage as Message,
    AsyncCursorPage,
)

"""
V0 Scope:

- Add Message -> `/v1/threads/{thread_id}/messages`
- Run Thread -> `/v1/threads/{thread_id}/run`
"""


@pytest.mark.asyncio
async def test_async_get_assistants():
    assistants = await litellm.aget_assistants(custom_llm_provider="openai")
    assert isinstance(assistants, AsyncCursorPage)


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_create_thread_litellm(sync_mode) -> Thread:
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore

    if sync_mode:
        new_thread = create_thread(
            custom_llm_provider="openai",
            messages=[message],  # type: ignore
        )
    else:
        new_thread = await litellm.acreate_thread(
            custom_llm_provider="openai",
            messages=[message],  # type: ignore
        )

    assert isinstance(
        new_thread, Thread
    ), f"type of thread={type(new_thread)}. Expected Thread-type"

    return new_thread


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_get_thread_litellm(sync_mode):
    new_thread = test_create_thread_litellm(sync_mode)

    if asyncio.iscoroutine(new_thread):
        _new_thread = await new_thread
    else:
        _new_thread = new_thread

    if sync_mode:
        received_thread = get_thread(
            custom_llm_provider="openai",
            thread_id=_new_thread.id,
        )
    else:
        received_thread = await litellm.aget_thread(
            custom_llm_provider="openai",
            thread_id=_new_thread.id,
        )

    assert isinstance(
        received_thread, Thread
    ), f"type of thread={type(received_thread)}. Expected Thread-type"
    return new_thread


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_add_message_litellm(sync_mode):
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    new_thread = test_create_thread_litellm(sync_mode)

    if asyncio.iscoroutine(new_thread):
        _new_thread = await new_thread
    else:
        _new_thread = new_thread
    # add message to thread
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    if sync_mode:
        added_message = litellm.add_message(
            thread_id=_new_thread.id, custom_llm_provider="openai", **message
        )
    else:
        added_message = await litellm.a_add_message(
            thread_id=_new_thread.id, custom_llm_provider="openai", **message
        )

    print(f"added message: {added_message}")

    assert isinstance(added_message, Message)


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_run_thread_litellm(sync_mode):
    """
    - Get Assistants
    - Create thread
    - Create run w/ Assistants + Thread
    """
    if sync_mode:
        assistants = litellm.get_assistants(custom_llm_provider="openai")
    else:
        assistants = await litellm.aget_assistants(custom_llm_provider="openai")

    ## get the first assistant ###
    assistant_id = assistants.data[0].id

    new_thread = test_create_thread_litellm(sync_mode=sync_mode)

    if asyncio.iscoroutine(new_thread):
        _new_thread = await new_thread
    else:
        _new_thread = new_thread

    thread_id = _new_thread.id

    # add message to thread
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore

    if sync_mode:
        added_message = litellm.add_message(
            thread_id=_new_thread.id, custom_llm_provider="openai", **message
        )

        run = litellm.run_thread(
            custom_llm_provider="openai", thread_id=thread_id, assistant_id=assistant_id
        )

        if run.status == "completed":
            messages = litellm.get_messages(
                thread_id=_new_thread.id, custom_llm_provider="openai"
            )
            assert isinstance(messages.data[0], Message)
        else:
            pytest.fail("An unexpected error occurred when running the thread")

    else:
        added_message = await litellm.a_add_message(
            thread_id=_new_thread.id, custom_llm_provider="openai", **message
        )

        run = await litellm.arun_thread(
            custom_llm_provider="openai", thread_id=thread_id, assistant_id=assistant_id
        )

        if run.status == "completed":
            messages = await litellm.aget_messages(
                thread_id=_new_thread.id, custom_llm_provider="openai"
            )
            assert isinstance(messages.data[0], Message)
        else:
            pytest.fail("An unexpected error occurred when running the thread")
