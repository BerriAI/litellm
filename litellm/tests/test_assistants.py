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
from litellm import create_thread
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


def test_create_thread_openai_direct() -> Thread:
    openai_api = OpenAIAssistantsAPI()

    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    new_thread = openai_api.create_thread(
        messages=[message],  # type: ignore
        api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore
        metadata={},
        api_base=None,
        timeout=600,
        max_retries=2,
        organization=None,
        client=None,
    )

    print(f"new_thread: {new_thread}")
    print(f"type of thread: {type(new_thread)}")
    assert isinstance(
        new_thread, Thread
    ), f"type of thread={type(new_thread)}. Expected Thread-type"
    return new_thread


def test_add_message_openai_direct():
    openai_api = OpenAIAssistantsAPI()
    # create thread
    new_thread = test_create_thread_openai_direct()
    # add message to thread
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    added_message = openai_api.add_message(
        thread_id=new_thread.id,
        message_data=message,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=None,
        timeout=600,
        max_retries=2,
        organization=None,
        client=None,
    )

    print(f"added message: {added_message}")

    assert isinstance(added_message, Message)


def test_get_thread_openai_direct():
    openai_api = OpenAIAssistantsAPI()

    ## create a thread w/ message ###
    new_thread = test_create_thread()

    retrieved_thread = openai_api.get_thread(
        thread_id=new_thread.id,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=None,
        timeout=600,
        max_retries=2,
        organization=None,
        client=None,
    )

    assert isinstance(
        retrieved_thread, Thread
    ), f"type of thread={type(retrieved_thread)}. Expected Thread-type"
    return new_thread


def test_run_thread_openai_direct():
    """
    - Get Assistants
    - Create thread
    - Create run w/ Assistants + Thread
    """
    openai_api = OpenAIAssistantsAPI()

    assistants = openai_api.get_assistants(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=None,
        timeout=600,
        max_retries=2,
        organization=None,
        client=None,
    )

    ## get the first assistant ###
    assistant_id = assistants.data[0].id

    ## create a thread w/ message ###
    new_thread = test_create_thread()

    thread_id = new_thread.id

    # add message to thread
    message: MessageData = {"role": "user", "content": "Hey, how's it going?"}  # type: ignore
    added_message = openai_api.add_message(
        thread_id=new_thread.id,
        message_data=message,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=None,
        timeout=600,
        max_retries=2,
        organization=None,
        client=None,
    )

    run = openai_api.run_thread(
        thread_id=thread_id,
        assistant_id=assistant_id,
        additional_instructions=None,
        instructions=None,
        metadata=None,
        model=None,
        stream=None,
        tools=None,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=None,
        timeout=600,
        max_retries=2,
        organization=None,
        client=None,
    )

    print(f"run: {run}")

    if run.status == "completed":
        messages = openai_api.get_messages(
            thread_id=new_thread.id,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=None,
            timeout=600,
            max_retries=2,
            organization=None,
            client=None,
        )
        assert isinstance(messages.data[0], Message)
    else:
        pytest.fail("An unexpected error occurred when running the thread")
