import pytest
import openai
import aiohttp
import asyncio
import tempfile
from typing_extensions import override
from openai import AssistantEventHandler


client = openai.OpenAI(base_url="http://0.0.0.0:4000/openai", api_key="sk-1234")


def test_pass_through_file_operations():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".txt", delete=False
    ) as temp_file:
        temp_file.write("This is a test file for the OpenAI Assistants API.")
        temp_file.flush()

        # create a file
        file = client.files.create(
            file=open(temp_file.name, "rb"),
            purpose="assistants",
        )
        print("file created", file)

        # delete the file
        delete_file = client.files.delete(file.id)
        print("file deleted", delete_file)


def test_openai_assistants_e2e_operations():
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4o",
    )
    print("assistant created", assistant)

    get_assistant = client.beta.assistants.retrieve(assistant.id)
    print(get_assistant)

    delete_assistant = client.beta.assistants.delete(assistant.id)
    print(delete_assistant)


class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)

    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == "code_interpreter":
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)


def test_openai_assistants_e2e_operations_stream():

    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4o",
    )
    print("assistant created", assistant)

    thread = client.beta.threads.create()
    print("thread created", thread)

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
    )
    print("message created", message)

    # Then, we use the `stream` SDK helper
    # with the `EventHandler` class to create the Run
    # and stream the response.

    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


def test_azure_openai_assistants_e2e_operations_stream():
    from openai import AzureOpenAI

    client = AzureOpenAI(
        base_url="http://0.0.0.0:4000/azure-config-passthrough/openai",
        api_key="sk-1234",
        api_version="2025-01-01-preview",
    )
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4o",
    )
    print("assistant created", assistant)

    thread = client.beta.threads.create()
    print("thread created", thread)

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
    )
    print("message created", message)

    # Then, we use the `stream` SDK helper
    # with the `EventHandler` class to create the Run
    # and stream the response.

    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()
