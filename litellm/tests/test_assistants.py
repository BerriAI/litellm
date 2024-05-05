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
)

"""
V0 Scope:

- Add Message -> `/v1/threads/{thread_id}/messages`
- Run Thread -> `/v1/threads/{thread_id}/run`
"""


def test_create_thread_litellm() -> Thread:
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    new_thread = create_thread(
        custom_llm_provider="openai",
        messages=[message],  # type: ignore
    )

    assert isinstance(
        new_thread, Thread
    ), f"type of thread={type(new_thread)}. Expected Thread-type"
    return new_thread


def test_get_thread_litellm():
    new_thread = test_create_thread_litellm()

    received_thread = get_thread(
        custom_llm_provider="openai",
        thread_id=new_thread.id,
    )

    assert isinstance(
        received_thread, Thread
    ), f"type of thread={type(received_thread)}. Expected Thread-type"
    return new_thread


def test_add_message_litellm():
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    new_thread = test_create_thread_litellm()

    # add message to thread
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    added_message = litellm.add_message(
        thread_id=new_thread.id, custom_llm_provider="openai", **message
    )

    print(f"added message: {added_message}")

    assert isinstance(added_message, Message)


def test_run_thread_litellm():
    """
    - Get Assistants
    - Create thread
    - Create run w/ Assistants + Thread
    """
    assistants = litellm.get_assistants(custom_llm_provider="openai")

    ## get the first assistant ###
    assistant_id = assistants.data[0].id

    new_thread = test_create_thread_litellm()

    thread_id = new_thread.id

    # add message to thread
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    added_message = litellm.add_message(
        thread_id=new_thread.id, custom_llm_provider="openai", **message
    )

    run = litellm.run_thread(
        custom_llm_provider="openai", thread_id=thread_id, assistant_id=assistant_id
    )

    if run.status == "completed":
        messages = litellm.get_messages(
            thread_id=new_thread.id, custom_llm_provider="openai"
        )
        assert isinstance(messages.data[0], Message)
    else:
        pytest.fail("An unexpected error occurred when running the thread")
