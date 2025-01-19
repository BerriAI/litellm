import asyncio
import copy
import json
import logging
import os
import sys
from typing import Any, Optional
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import completion
from litellm.caching import InMemoryCache
import logging
from litellm._logging import verbose_logger

logging.basicConfig(level=logging.DEBUG)
litellm.num_retries = 3
litellm.success_callback = ["langfuse"]
os.environ["LANGFUSE_DEBUG"] = "True"
import time

import pytest


# Get the current directory of the file being-run
pwd = os.path.dirname(os.path.realpath(__file__))
print(pwd)

file_path = os.path.join(pwd, "gettysburg.wav")

audio_file = open(file_path, "rb")


def assert_langfuse_request_matches_expected(
    actual_request_body: dict,
    expected_file_name: str,
    trace_id: Optional[str] = None,
):
    """
    Helper function to compare actual Langfuse request body with expected JSON file.

    Args:
        actual_request_body (dict): The actual request body received from the API call
        expected_file_name (str): Name of the JSON file containing expected request body (e.g., "transcription.json")
    """
    # Get the current directory and read the expected request body
    pwd = os.path.dirname(os.path.realpath(__file__))
    expected_body_path = os.path.join(
        pwd, "langfuse_expected_request_body", expected_file_name
    )

    with open(expected_body_path, "r") as f:
        expected_request_body = json.load(f)

    # Replace dynamic values in actual request body
    for item in actual_request_body["batch"]:
        # Replace IDs with expected IDs
        if item["type"] == "trace-create":
            item["id"] = expected_request_body["batch"][0]["id"]
            item["body"]["id"] = expected_request_body["batch"][0]["body"]["id"]
            item["timestamp"] = expected_request_body["batch"][0]["timestamp"]
            item["body"]["timestamp"] = expected_request_body["batch"][0]["body"][
                "timestamp"
            ]
        elif item["type"] == "generation-create":
            item["id"] = expected_request_body["batch"][1]["id"]
            item["body"]["id"] = expected_request_body["batch"][1]["body"]["id"]
            item["timestamp"] = expected_request_body["batch"][1]["timestamp"]
            item["body"]["startTime"] = expected_request_body["batch"][1]["body"][
                "startTime"
            ]
            item["body"]["endTime"] = expected_request_body["batch"][1]["body"][
                "endTime"
            ]
            item["body"]["completionStartTime"] = expected_request_body["batch"][1][
                "body"
            ]["completionStartTime"]
            if trace_id is None:
                print("popping traceId")
                item["body"].pop("traceId")
            else:
                item["body"]["traceId"] = trace_id
                expected_request_body["batch"][1]["body"]["traceId"] = trace_id

    # Replace SDK version with expected version
    actual_request_body["metadata"]["sdk_version"] = expected_request_body["metadata"][
        "sdk_version"
    ]
    # Assert the entire request body matches
    assert (
        actual_request_body == expected_request_body
    ), f"Difference in request bodies: {json.dumps(actual_request_body, indent=2)} != {json.dumps(expected_request_body, indent=2)}"


class TestLangfuseLogging:
    @pytest.fixture
    async def mock_setup(self):
        """Common setup for Langfuse logging tests"""
        import uuid
        from unittest.mock import AsyncMock, patch
        import httpx

        # Create a mock Response object
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}

        # Create mock for httpx.Client.post
        mock_post = AsyncMock()
        mock_post.return_value = mock_response

        litellm.set_verbose = True
        litellm.success_callback = ["langfuse"]

        return {"trace_id": f"litellm-test-{str(uuid.uuid4())}", "mock_post": mock_post}

    async def _verify_langfuse_call(
        self,
        mock_post,
        expected_file_name,
        trace_id: Optional[str] = None,
    ):
        """Helper method to verify Langfuse API calls"""
        await asyncio.sleep(1)

        # Verify the call
        assert mock_post.call_count >= 1
        url = mock_post.call_args[0][0]
        request_body = mock_post.call_args[1].get("content")

        # Parse the JSON string into a dict for assertions
        actual_request_body = json.loads(request_body)

        print("\nMocked Request Details:")
        print(f"URL: {url}")
        print(f"Request Body: {json.dumps(actual_request_body, indent=4)}")

        assert url == "https://us.cloud.langfuse.com/api/public/ingestion"
        assert_langfuse_request_matches_expected(
            actual_request_body, expected_file_name, trace_id
        )

    @pytest.mark.asyncio
    async def test_langfuse_logging_transcription(self, mock_setup):
        """Test Langfuse logging for audio transcription"""
        setup = await mock_setup  # Await the fixture
        with patch("httpx.Client.post", setup["mock_post"]):
            await litellm.atranscription(
                model="whisper-1",
                file=audio_file,
                metadata={"trace_id": setup["trace_id"]},
            )
            await self._verify_langfuse_call(
                setup["mock_post"], "transcription.json", setup["trace_id"]
            )

    @pytest.mark.asyncio
    async def test_langfuse_logging_completion(self, mock_setup):
        """Test Langfuse logging for text completion"""
        setup = await mock_setup  # Await the fixture
        with patch("httpx.Client.post", setup["mock_post"]):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                mock_response="Hello! How can I assist you today?",
                metadata={"trace_id": setup["trace_id"]},
            )
            await self._verify_langfuse_call(
                setup["mock_post"], "completion.json", setup["trace_id"]
            )

    @pytest.mark.asyncio
    async def test_langfuse_logging_streaming_completion(self, mock_setup):
        """Test Langfuse logging for streaming completion"""
        setup = await mock_setup  # Await the fixture
        with patch("httpx.Client.post", setup["mock_post"]):
            async for chunk in await litellm.acompletion(  # type: ignore
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}],
                stream=True,
                mock_response="Hello! How can I assist you today?",
                metadata={"trace_id": setup["trace_id"]},
            ):
                pass  # Process chunks if needed
            await self._verify_langfuse_call(
                setup["mock_post"], "streaming_completion.json", setup["trace_id"]
            )

    @pytest.mark.asyncio
    async def test_langfuse_logging_embedding(self, mock_setup):
        """Test Langfuse logging for embeddings"""
        setup = await mock_setup  # Await the fixture
        with patch("httpx.Client.post", setup["mock_post"]):
            await litellm.aembedding(
                model="text-embedding-ada-002",
                input=["Hello world"],
            )
            await self._verify_langfuse_call(setup["mock_post"], "embedding.json")

    @pytest.mark.asyncio
    async def test_langfuse_logging_custom_generation_name(self, mock_setup):
        """Test Langfuse logging with custom generation name and metadata"""
        setup = await mock_setup  # Await the fixture
        with patch("httpx.Client.post", setup["mock_post"]):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi 👋 - i'm claude"}],
                max_tokens=10,
                metadata={
                    "langfuse/foo": "bar",
                    "langsmith/fizz": "buzz",
                    "prompt_hash": "asdf98u0j9131123",
                    "generation_name": "ishaan-test-generation",
                    "generation_id": "gen-id22",
                    "trace_id": setup["trace_id"],
                    "trace_user_id": "user-id2",
                },
                mock_response="Hello! I'm an AI assistant.",
            )
            await self._verify_langfuse_call(
                setup["mock_post"], "custom_generation.json", setup["trace_id"]
            )

    @pytest.mark.asyncio
    async def test_langfuse_masked_input_output(self, mock_setup):
        """Test Langfuse logging with masked input and output"""
        setup = await mock_setup  # Await the fixture
        with patch("httpx.Client.post", setup["mock_post"]):
            import uuid

            for mask_value in [True, False]:
                await litellm.acompletion(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "This is a test"}],
                    mock_response="This is a test",
                    metadata={
                        "mask_input": mask_value,
                        "mask_output": mask_value,
                    },
                )
                if mask_value is True:
                    await self._verify_langfuse_call(
                        setup["mock_post"],
                        "completion_redacted.json",
                        setup["trace_id"],
                    )
                else:
                    await self._verify_langfuse_call(
                        setup["mock_post"],
                        "completion_non_redacted.json",
                        setup["trace_id"],
                    )


@pytest.mark.asyncio
@pytest.mark.flaky(retries=12, delay=2)
async def test_aaalangfuse_logging_metadata():
    """
    Test that creates multiple traces, with a varying number of generations and sets various metadata fields
    Confirms that no metadata that is standard within Langfuse is duplicated in the respective trace or generation metadata
    For trace continuation certain metadata of the trace is overriden with metadata from the last generation based on the update_trace_keys field
    Version is set for both the trace and the generation
    Release is just set for the trace
    Tags is just set for the trace
    """
    import uuid

    litellm.set_verbose = True
    litellm.success_callback = ["langfuse"]

    trace_identifiers = {}
    expected_filtered_metadata_keys = {
        "trace_name",
        "trace_id",
        "existing_trace_id",
        "trace_user_id",
        "session_id",
        "tags",
        "generation_name",
        "generation_id",
        "prompt",
    }
    trace_metadata = {
        "trace_actual_metadata_key": "trace_actual_metadata_value"
    }  # Allows for setting the metadata on the trace
    run_id = str(uuid.uuid4())
    session_id = f"litellm-test-session-{run_id}"
    trace_common_metadata = {
        "session_id": session_id,
        "tags": ["litellm-test-tag1", "litellm-test-tag2"],
        "update_trace_keys": [
            "output",
            "trace_metadata",
        ],  # Overwrite the following fields in the trace with the last generation's output and the trace_user_id
        "trace_metadata": trace_metadata,
        "gen_metadata_key": "gen_metadata_value",  # Metadata key that should not be filtered in the generation
        "trace_release": "litellm-test-release",
        "version": "litellm-test-version",
    }
    for trace_num in range(1, 3):  # Two traces
        metadata = copy.deepcopy(trace_common_metadata)
        trace_id = f"litellm-test-trace{trace_num}-{run_id}"
        metadata["trace_id"] = trace_id
        metadata["trace_name"] = trace_id
        trace_identifiers[trace_id] = []
        print(f"Trace: {trace_id}")
        for generation_num in range(
            1, trace_num + 1
        ):  # Each trace has a number of generations equal to its trace number
            metadata["trace_user_id"] = f"litellm-test-user{generation_num}-{run_id}"
            generation_id = (
                f"litellm-test-trace{trace_num}-generation-{generation_num}-{run_id}"
            )
            metadata["generation_id"] = generation_id
            metadata["generation_name"] = generation_id
            metadata["trace_metadata"][
                "generation_id"
            ] = generation_id  # Update to test if trace_metadata is overwritten by update trace keys
            trace_identifiers[trace_id].append(generation_id)
            print(f"Generation: {generation_id}")
            response = await create_async_task(
                model="gpt-3.5-turbo",
                mock_response=f"{session_id}:{trace_id}:{generation_id}",
                messages=[
                    {
                        "role": "user",
                        "content": f"{session_id}:{trace_id}:{generation_id}",
                    }
                ],
                max_tokens=100,
                temperature=0.2,
                metadata=copy.deepcopy(
                    metadata
                ),  # Every generation needs its own metadata, langfuse is not async/thread safe without it
            )
            print(response)
            metadata["existing_trace_id"] = trace_id

            await asyncio.sleep(2)
    await asyncio.sleep(4)


@pytest.mark.skip(reason="Need to address this on main")
def test_aaalangfuse_existing_trace_id():
    """
    When existing trace id is passed, don't set trace params -> prevents overwriting the trace

    Pass 1 logging object with a trace

    Pass 2nd logging object with the trace id

    Assert no changes to the trace
    """
    # Test - if the logs were sent to the correct team on langfuse
    import datetime

    import litellm
    from litellm.integrations.langfuse.langfuse import LangFuseLogger

    langfuse_Logger = LangFuseLogger(
        langfuse_public_key=os.getenv("LANGFUSE_PROJECT2_PUBLIC"),
        langfuse_secret=os.getenv("LANGFUSE_PROJECT2_SECRET"),
    )
    litellm.success_callback = ["langfuse"]

    # langfuse_args = {'kwargs': { 'start_time':  'end_time': datetime.datetime(2024, 5, 1, 7, 31, 29, 903685), 'user_id': None, 'print_verbose': <function print_verbose at 0x109d1f420>, 'level': 'DEFAULT', 'status_message': None}
    response_obj = litellm.ModelResponse(
        id="chatcmpl-9K5HUAbVRqFrMZKXL0WoC295xhguY",
        choices=[
            litellm.Choices(
                finish_reason="stop",
                index=0,
                message=litellm.Message(
                    content="I'm sorry, I am an AI assistant and do not have real-time information. I recommend checking a reliable weather website or app for the most up-to-date weather information in Boston.",
                    role="assistant",
                ),
            )
        ],
        created=1714573888,
        model="gpt-3.5-turbo-0125",
        object="chat.completion",
        system_fingerprint="fp_3b956da36b",
        usage=litellm.Usage(completion_tokens=37, prompt_tokens=14, total_tokens=51),
    )

    ### NEW TRACE ###
    message = [{"role": "user", "content": "what's the weather in boston"}]
    langfuse_args = {
        "response_obj": response_obj,
        "kwargs": {
            "model": "gpt-3.5-turbo",
            "litellm_params": {
                "acompletion": False,
                "api_key": None,
                "force_timeout": 600,
                "logger_fn": None,
                "verbose": False,
                "custom_llm_provider": "openai",
                "api_base": "https://api.openai.com/v1/",
                "litellm_call_id": None,
                "model_alias_map": {},
                "completion_call_id": None,
                "metadata": None,
                "model_info": None,
                "proxy_server_request": None,
                "preset_cache_key": None,
                "no-log": False,
                "stream_response": {},
            },
            "messages": message,
            "optional_params": {"temperature": 0.1, "extra_body": {}},
            "start_time": "2024-05-01 07:31:27.986164",
            "stream": False,
            "user": None,
            "call_type": "completion",
            "litellm_call_id": None,
            "completion_start_time": "2024-05-01 07:31:29.903685",
            "temperature": 0.1,
            "extra_body": {},
            "input": [{"role": "user", "content": "what's the weather in boston"}],
            "api_key": "my-api-key",
            "additional_args": {
                "complete_input_dict": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": "what's the weather in boston"}
                    ],
                    "temperature": 0.1,
                    "extra_body": {},
                }
            },
            "log_event_type": "successful_api_call",
            "end_time": "2024-05-01 07:31:29.903685",
            "cache_hit": None,
            "response_cost": 6.25e-05,
        },
        "start_time": datetime.datetime(2024, 5, 1, 7, 31, 27, 986164),
        "end_time": datetime.datetime(2024, 5, 1, 7, 31, 29, 903685),
        "user_id": None,
        "print_verbose": litellm.print_verbose,
        "level": "DEFAULT",
        "status_message": None,
    }

    langfuse_response_object = langfuse_Logger.log_event(**langfuse_args)

    import langfuse

    langfuse_client = langfuse.Langfuse(
        public_key=os.getenv("LANGFUSE_PROJECT2_PUBLIC"),
        secret_key=os.getenv("LANGFUSE_PROJECT2_SECRET"),
    )

    trace_id = langfuse_response_object["trace_id"]

    assert trace_id is not None

    langfuse_client.flush()

    time.sleep(2)

    print(langfuse_client.get_trace(id=trace_id))

    initial_langfuse_trace = langfuse_client.get_trace(id=trace_id)

    ### EXISTING TRACE ###

    new_metadata = {"existing_trace_id": trace_id}
    new_messages = [{"role": "user", "content": "What do you know?"}]
    new_response_obj = litellm.ModelResponse(
        id="chatcmpl-9K5HUAbVRqFrMZKXL0WoC295xhguY",
        choices=[
            litellm.Choices(
                finish_reason="stop",
                index=0,
                message=litellm.Message(
                    content="What do I know?",
                    role="assistant",
                ),
            )
        ],
        created=1714573888,
        model="gpt-3.5-turbo-0125",
        object="chat.completion",
        system_fingerprint="fp_3b956da36b",
        usage=litellm.Usage(completion_tokens=37, prompt_tokens=14, total_tokens=51),
    )
    langfuse_args = {
        "response_obj": new_response_obj,
        "kwargs": {
            "model": "gpt-3.5-turbo",
            "litellm_params": {
                "acompletion": False,
                "api_key": None,
                "force_timeout": 600,
                "logger_fn": None,
                "verbose": False,
                "custom_llm_provider": "openai",
                "api_base": "https://api.openai.com/v1/",
                "litellm_call_id": "508113a1-c6f1-48ce-a3e1-01c6cce9330e",
                "model_alias_map": {},
                "completion_call_id": None,
                "metadata": new_metadata,
                "model_info": None,
                "proxy_server_request": None,
                "preset_cache_key": None,
                "no-log": False,
                "stream_response": {},
            },
            "messages": new_messages,
            "optional_params": {"temperature": 0.1, "extra_body": {}},
            "start_time": "2024-05-01 07:31:27.986164",
            "stream": False,
            "user": None,
            "call_type": "completion",
            "litellm_call_id": "508113a1-c6f1-48ce-a3e1-01c6cce9330e",
            "completion_start_time": "2024-05-01 07:31:29.903685",
            "temperature": 0.1,
            "extra_body": {},
            "input": [{"role": "user", "content": "what's the weather in boston"}],
            "api_key": "my-api-key",
            "additional_args": {
                "complete_input_dict": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "user", "content": "what's the weather in boston"}
                    ],
                    "temperature": 0.1,
                    "extra_body": {},
                }
            },
            "log_event_type": "successful_api_call",
            "end_time": "2024-05-01 07:31:29.903685",
            "cache_hit": None,
            "response_cost": 6.25e-05,
        },
        "start_time": datetime.datetime(2024, 5, 1, 7, 31, 27, 986164),
        "end_time": datetime.datetime(2024, 5, 1, 7, 31, 29, 903685),
        "user_id": None,
        "print_verbose": litellm.print_verbose,
        "level": "DEFAULT",
        "status_message": None,
    }

    langfuse_response_object = langfuse_Logger.log_event(**langfuse_args)

    new_trace_id = langfuse_response_object["trace_id"]

    assert new_trace_id == trace_id

    langfuse_client.flush()

    time.sleep(2)

    print(langfuse_client.get_trace(id=trace_id))

    new_langfuse_trace = langfuse_client.get_trace(id=trace_id)

    initial_langfuse_trace_dict = dict(initial_langfuse_trace)
    initial_langfuse_trace_dict.pop("updatedAt")
    initial_langfuse_trace_dict.pop("timestamp")

    new_langfuse_trace_dict = dict(new_langfuse_trace)
    new_langfuse_trace_dict.pop("updatedAt")
    new_langfuse_trace_dict.pop("timestamp")

    assert initial_langfuse_trace_dict == new_langfuse_trace_dict


@pytest.mark.skipif(
    condition=not os.environ.get("OPENAI_API_KEY", False),
    reason="Authentication missing for openai",
)
def test_langfuse_logging_tool_calling():
    litellm.set_verbose = True

    def get_current_weather(location, unit="fahrenheit"):
        """Get the current weather in a given location"""
        if "tokyo" in location.lower():
            return json.dumps(
                {"location": "Tokyo", "temperature": "10", "unit": "celsius"}
            )
        elif "san francisco" in location.lower():
            return json.dumps(
                {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
            )
        elif "paris" in location.lower():
            return json.dumps(
                {"location": "Paris", "temperature": "22", "unit": "celsius"}
            )
        else:
            return json.dumps({"location": location, "temperature": "unknown"})

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = litellm.completion(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    print("\nLLM Response1:\n", response)
    response_message = response.choices[0].message
    tool_calls = response.choices[0].message.tool_calls


# test_langfuse_logging_tool_calling()


def get_langfuse_prompt(name: str):
    import langfuse
    from langfuse import Langfuse

    try:
        langfuse = Langfuse(
            public_key=os.environ["LANGFUSE_DEV_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_DEV_SK_KEY"],
            host=os.environ["LANGFUSE_HOST"],
        )

        # Get current production version of a text prompt
        prompt = langfuse.get_prompt(name=name)
        return prompt
    except Exception as e:
        raise Exception(f"Error getting prompt: {e}")


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="local only test, use this to verify if we can send request to litellm proxy server"
)
async def test_make_request():
    response = await litellm.acompletion(
        model="openai/llama3",
        api_key="sk-1234",
        base_url="http://localhost:4000",
        messages=[{"role": "user", "content": "Hi 👋 - i'm claude"}],
        extra_body={
            "metadata": {
                "tags": ["openai"],
                "prompt": get_langfuse_prompt("test-chat"),
            }
        },
    )


@pytest.mark.skip(
    reason="local only test, use this to verify if dynamic langfuse logging works as expected"
)
def test_aaalangfuse_dynamic_logging():
    """
    pass in langfuse credentials via completion call

    assert call is logged.

    Covers the team-logging scenario.
    """
    import uuid

    import langfuse

    trace_id = str(uuid.uuid4())
    _ = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey"}],
        mock_response="Hey! how's it going?",
        langfuse_public_key=os.getenv("LANGFUSE_PROJECT2_PUBLIC"),
        langfuse_secret_key=os.getenv("LANGFUSE_PROJECT2_SECRET"),
        metadata={"trace_id": trace_id},
        success_callback=["langfuse"],
    )

    time.sleep(3)

    langfuse_client = langfuse.Langfuse(
        public_key=os.getenv("LANGFUSE_PROJECT2_PUBLIC"),
        secret_key=os.getenv("LANGFUSE_PROJECT2_SECRET"),
    )

    langfuse_client.get_trace(id=trace_id)


import datetime

generation_params = {
    "name": "litellm-acompletion",
    "id": "time-10-35-32-316778_chatcmpl-ABQDEzVJS8fziPdvkeTA3tnQaxeMX",
    "start_time": datetime.datetime(2024, 9, 25, 10, 35, 32, 316778),
    "end_time": datetime.datetime(2024, 9, 25, 10, 35, 32, 897141),
    "model": "gpt-4o",
    "model_parameters": {
        "stream": False,
        "max_retries": 0,
        "extra_body": "{}",
        "system_fingerprint": "fp_52a7f40b0b",
    },
    "input": {
        "messages": [
            {"content": "<>", "role": "system"},
            {"content": "<>", "role": "user"},
        ]
    },
    "output": {
        "content": "Hello! It looks like your message might have been sent by accident. How can I assist you today?",
        "role": "assistant",
        "tool_calls": None,
        "function_call": None,
    },
    "usage": {"prompt_tokens": 13, "completion_tokens": 21, "total_cost": 0.00038},
    "metadata": {
        "prompt": {
            "name": "conversational-service-answer_question_restricted_reply",
            "version": 9,
            "config": {},
            "labels": ["latest", "staging", "production"],
            "tags": ["conversational-service"],
            "prompt": [
                {"role": "system", "content": "<>"},
                {"role": "user", "content": "{{text}}"},
            ],
        },
        "requester_metadata": {
            "session_id": "e953a71f-e129-4cf5-ad11-ad18245022f1",
            "trace_name": "jess",
            "tags": ["conversational-service", "generative-ai-engine", "staging"],
            "prompt": {
                "name": "conversational-service-answer_question_restricted_reply",
                "version": 9,
                "config": {},
                "labels": ["latest", "staging", "production"],
                "tags": ["conversational-service"],
                "prompt": [
                    {"role": "system", "content": "<>"},
                    {"role": "user", "content": "{{text}}"},
                ],
            },
        },
        "user_api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
        "litellm_api_version": "0.0.0",
        "user_api_key_user_id": "default_user_id",
        "user_api_key_spend": 0.0,
        "user_api_key_metadata": {},
        "requester_ip_address": "127.0.0.1",
        "model_group": "gpt-4o",
        "model_group_size": 0,
        "deployment": "gpt-4o",
        "model_info": {
            "id": "5583ac0c3e38cfd381b6cc09bcca6e0db60af48d3f16da325f82eb9df1b6a1e4",
            "db_model": False,
        },
        "hidden_params": {
            "headers": {
                "date": "Wed, 25 Sep 2024 17:35:32 GMT",
                "content-type": "application/json",
                "transfer-encoding": "chunked",
                "connection": "keep-alive",
                "access-control-expose-headers": "X-Request-ID",
                "openai-organization": "reliablekeystest",
                "openai-processing-ms": "329",
                "openai-version": "2020-10-01",
                "strict-transport-security": "max-age=31536000; includeSubDomains; preload",
                "x-ratelimit-limit-requests": "10000",
                "x-ratelimit-limit-tokens": "30000000",
                "x-ratelimit-remaining-requests": "9999",
                "x-ratelimit-remaining-tokens": "29999980",
                "x-ratelimit-reset-requests": "6ms",
                "x-ratelimit-reset-tokens": "0s",
                "x-request-id": "req_fdff3bfa11c391545d2042d46473214f",
                "cf-cache-status": "DYNAMIC",
                "set-cookie": "__cf_bm=NWwOByRU5dQwDqLRYbbTT.ecfqvnWiBi8aF9rfp1QB8-1727285732-1.0.1.1-.Cm0UGMaQ4qZbY3ZU0F7trjSsNUcIBo04PetRMlCoyoTCTnKTbmwmDCWcHmqHOTuE_bNspSgfQoANswx4BSD.A; path=/; expires=Wed, 25-Sep-24 18:05:32 GMT; domain=.api.openai.com; HttpOnly; Secure; SameSite=None, _cfuvid=1b_nyqBtAs4KHRhFBV2a.8zic1fSRJxT.Jn1npl1_GY-1727285732915-0.0.1.1-604800000; path=/; domain=.api.openai.com; HttpOnly; Secure; SameSite=None",
                "x-content-type-options": "nosniff",
                "server": "cloudflare",
                "cf-ray": "8c8cc573becb232c-SJC",
                "content-encoding": "gzip",
                "alt-svc": 'h3=":443"; ma=86400',
            },
            "additional_headers": {
                "llm_provider-date": "Wed, 25 Sep 2024 17:35:32 GMT",
                "llm_provider-content-type": "application/json",
                "llm_provider-transfer-encoding": "chunked",
                "llm_provider-connection": "keep-alive",
                "llm_provider-access-control-expose-headers": "X-Request-ID",
                "llm_provider-openai-organization": "reliablekeystest",
                "llm_provider-openai-processing-ms": "329",
                "llm_provider-openai-version": "2020-10-01",
                "llm_provider-strict-transport-security": "max-age=31536000; includeSubDomains; preload",
                "llm_provider-x-ratelimit-limit-requests": "10000",
                "llm_provider-x-ratelimit-limit-tokens": "30000000",
                "llm_provider-x-ratelimit-remaining-requests": "9999",
                "llm_provider-x-ratelimit-remaining-tokens": "29999980",
                "llm_provider-x-ratelimit-reset-requests": "6ms",
                "llm_provider-x-ratelimit-reset-tokens": "0s",
                "llm_provider-x-request-id": "req_fdff3bfa11c391545d2042d46473214f",
                "llm_provider-cf-cache-status": "DYNAMIC",
                "llm_provider-set-cookie": "__cf_bm=NWwOByRU5dQwDqLRYbbTT.ecfqvnWiBi8aF9rfp1QB8-1727285732-1.0.1.1-.Cm0UGMaQ4qZbY3ZU0F7trjSsNUcIBo04PetRMlCoyoTCTnKTbmwmDCWcHmqHOTuE_bNspSgfQoANswx4BSD.A; path=/; expires=Wed, 25-Sep-24 18:05:32 GMT; domain=.api.openai.com; HttpOnly; Secure; SameSite=None, _cfuvid=1b_nyqBtAs4KHRhFBV2a.8zic1fSRJxT.Jn1npl1_GY-1727285732915-0.0.1.1-604800000; path=/; domain=.api.openai.com; HttpOnly; Secure; SameSite=None",
                "llm_provider-x-content-type-options": "nosniff",
                "llm_provider-server": "cloudflare",
                "llm_provider-cf-ray": "8c8cc573becb232c-SJC",
                "llm_provider-content-encoding": "gzip",
                "llm_provider-alt-svc": 'h3=":443"; ma=86400',
            },
            "litellm_call_id": "1fa31658-20af-40b5-9ac9-60fd7b5ad98c",
            "model_id": "5583ac0c3e38cfd381b6cc09bcca6e0db60af48d3f16da325f82eb9df1b6a1e4",
            "api_base": "https://api.openai.com",
            "optional_params": {
                "stream": False,
                "max_retries": 0,
                "extra_body": {},
            },
            "response_cost": 0.00038,
        },
        "litellm_response_cost": 0.00038,
        "api_base": "https://api.openai.com/v1/",
        "cache_hit": False,
    },
    "level": "DEFAULT",
    "version": None,
}


@pytest.mark.parametrize(
    "prompt",
    [
        [
            {"role": "system", "content": "<>"},
            {"role": "user", "content": "{{text}}"},
        ],
        "hello world",
    ],
)
def test_langfuse_prompt_type(prompt):

    from litellm.integrations.langfuse.langfuse import _add_prompt_to_generation_params
    from unittest.mock import patch, MagicMock, Mock

    clean_metadata = {
        "prompt": {
            "name": "conversational-service-answer_question_restricted_reply",
            "version": 9,
            "config": {},
            "labels": ["latest", "staging", "production"],
            "tags": ["conversational-service"],
            "prompt": prompt,
        },
        "requester_metadata": {
            "session_id": "e953a71f-e129-4cf5-ad11-ad18245022f1",
            "trace_name": "jess",
            "tags": ["conversational-service", "generative-ai-engine", "staging"],
            "prompt": {
                "name": "conversational-service-answer_question_restricted_reply",
                "version": 9,
                "config": {},
                "labels": ["latest", "staging", "production"],
                "tags": ["conversational-service"],
                "prompt": [
                    {"role": "system", "content": "<>"},
                    {"role": "user", "content": "{{text}}"},
                ],
            },
        },
        "user_api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
        "litellm_api_version": "0.0.0",
        "user_api_key_user_id": "default_user_id",
        "user_api_key_spend": 0.0,
        "user_api_key_metadata": {},
        "requester_ip_address": "127.0.0.1",
        "model_group": "gpt-4o",
        "model_group_size": 0,
        "deployment": "gpt-4o",
        "model_info": {
            "id": "5583ac0c3e38cfd381b6cc09bcca6e0db60af48d3f16da325f82eb9df1b6a1e4",
            "db_model": False,
        },
        "hidden_params": {
            "headers": {
                "date": "Wed, 25 Sep 2024 17:35:32 GMT",
                "content-type": "application/json",
                "transfer-encoding": "chunked",
                "connection": "keep-alive",
                "access-control-expose-headers": "X-Request-ID",
                "openai-organization": "reliablekeystest",
                "openai-processing-ms": "329",
                "openai-version": "2020-10-01",
                "strict-transport-security": "max-age=31536000; includeSubDomains; preload",
                "x-ratelimit-limit-requests": "10000",
                "x-ratelimit-limit-tokens": "30000000",
                "x-ratelimit-remaining-requests": "9999",
                "x-ratelimit-remaining-tokens": "29999980",
                "x-ratelimit-reset-requests": "6ms",
                "x-ratelimit-reset-tokens": "0s",
                "x-request-id": "req_fdff3bfa11c391545d2042d46473214f",
                "cf-cache-status": "DYNAMIC",
                "set-cookie": "__cf_bm=NWwOByRU5dQwDqLRYbbTT.ecfqvnWiBi8aF9rfp1QB8-1727285732-1.0.1.1-.Cm0UGMaQ4qZbY3ZU0F7trjSsNUcIBo04PetRMlCoyoTCTnKTbmwmDCWcHmqHOTuE_bNspSgfQoANswx4BSD.A; path=/; expires=Wed, 25-Sep-24 18:05:32 GMT; domain=.api.openai.com; HttpOnly; Secure; SameSite=None, _cfuvid=1b_nyqBtAs4KHRhFBV2a.8zic1fSRJxT.Jn1npl1_GY-1727285732915-0.0.1.1-604800000; path=/; domain=.api.openai.com; HttpOnly; Secure; SameSite=None",
                "x-content-type-options": "nosniff",
                "server": "cloudflare",
                "cf-ray": "8c8cc573becb232c-SJC",
                "content-encoding": "gzip",
                "alt-svc": 'h3=":443"; ma=86400',
            },
            "additional_headers": {
                "llm_provider-date": "Wed, 25 Sep 2024 17:35:32 GMT",
                "llm_provider-content-type": "application/json",
                "llm_provider-transfer-encoding": "chunked",
                "llm_provider-connection": "keep-alive",
                "llm_provider-access-control-expose-headers": "X-Request-ID",
                "llm_provider-openai-organization": "reliablekeystest",
                "llm_provider-openai-processing-ms": "329",
                "llm_provider-openai-version": "2020-10-01",
                "llm_provider-strict-transport-security": "max-age=31536000; includeSubDomains; preload",
                "llm_provider-x-ratelimit-limit-requests": "10000",
                "llm_provider-x-ratelimit-limit-tokens": "30000000",
                "llm_provider-x-ratelimit-remaining-requests": "9999",
                "llm_provider-x-ratelimit-remaining-tokens": "29999980",
                "llm_provider-x-ratelimit-reset-requests": "6ms",
                "llm_provider-x-ratelimit-reset-tokens": "0s",
                "llm_provider-x-request-id": "req_fdff3bfa11c391545d2042d46473214f",
                "llm_provider-cf-cache-status": "DYNAMIC",
                "llm_provider-set-cookie": "__cf_bm=NWwOByRU5dQwDqLRYbbTT.ecfqvnWiBi8aF9rfp1QB8-1727285732-1.0.1.1-.Cm0UGMaQ4qZbY3ZU0F7trjSsNUcIBo04PetRMlCoyoTCTnKTbmwmDCWcHmqHOTuE_bNspSgfQoANswx4BSD.A; path=/; expires=Wed, 25-Sep-24 18:05:32 GMT; domain=.api.openai.com; HttpOnly; Secure; SameSite=None, _cfuvid=1b_nyqBtAs4KHRhFBV2a.8zic1fSRJxT.Jn1npl1_GY-1727285732915-0.0.1.1-604800000; path=/; domain=.api.openai.com; HttpOnly; Secure; SameSite=None",
                "llm_provider-x-content-type-options": "nosniff",
                "llm_provider-server": "cloudflare",
                "llm_provider-cf-ray": "8c8cc573becb232c-SJC",
                "llm_provider-content-encoding": "gzip",
                "llm_provider-alt-svc": 'h3=":443"; ma=86400',
            },
            "litellm_call_id": "1fa31658-20af-40b5-9ac9-60fd7b5ad98c",
            "model_id": "5583ac0c3e38cfd381b6cc09bcca6e0db60af48d3f16da325f82eb9df1b6a1e4",
            "api_base": "https://api.openai.com",
            "optional_params": {"stream": False, "max_retries": 0, "extra_body": {}},
            "response_cost": 0.00038,
        },
        "litellm_response_cost": 0.00038,
        "api_base": "https://api.openai.com/v1/",
        "cache_hit": False,
    }
    _add_prompt_to_generation_params(
        generation_params=generation_params,
        clean_metadata=clean_metadata,
        prompt_management_metadata=None,
        langfuse_client=Mock(),
    )


def test_langfuse_logging_metadata():
    from litellm.integrations.langfuse.langfuse import log_requester_metadata

    metadata = {"key": "value", "requester_metadata": {"key": "value"}}

    got_metadata = log_requester_metadata(clean_metadata=metadata)
    expected_metadata = {"requester_metadata": {"key": "value"}}

    assert expected_metadata == got_metadata


def test_langfuse_logging_async():
    # this tests time added to make langfuse logging calls, vs just acompletion calls
    try:
        litellm.set_verbose = True

        # Make 5 calls with an empty success_callback
        litellm.success_callback = []
        start_time_empty_callback = asyncio.run(make_async_calls())
        print("done with no callback test")

        print("starting langfuse test")
        # Make 5 calls with success_callback set to "langfuse"
        litellm.success_callback = ["langfuse"]
        start_time_langfuse = asyncio.run(make_async_calls())
        print("done with langfuse test")

        # Compare the time for both scenarios
        print(f"Time taken with success_callback='langfuse': {start_time_langfuse}")
        print(f"Time taken with empty success_callback: {start_time_empty_callback}")

        # assert the diff is not more than 1 second - this was 5 seconds before the fix
        assert abs(start_time_langfuse - start_time_empty_callback) < 1

    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")


async def make_async_calls(metadata=None, **completion_kwargs):
    tasks = []
    for _ in range(5):
        tasks.append(create_async_task())

    # Measure the start time before running the tasks
    start_time = asyncio.get_event_loop().time()

    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)

    # Print the responses when tasks return
    for idx, response in enumerate(responses):
        print(f"Response from Task {idx + 1}: {response}")

    # Calculate the total time taken
    total_time = asyncio.get_event_loop().time() - start_time

    return total_time


def create_async_task(**completion_kwargs):
    """
    Creates an async task for the litellm.acompletion function.
    This is just the task, but it is not run here.
    To run the task it must be awaited or used in other asyncio coroutine execution functions like asyncio.gather.
    Any kwargs passed to this function will be passed to the litellm.acompletion function.
    By default a standard set of arguments are used for the litellm.acompletion function.
    """
    completion_args = {
        "model": "azure/chatgpt-v-2",
        "api_version": "2024-02-01",
        "messages": [{"role": "user", "content": "This is a test"}],
        "max_tokens": 5,
        "temperature": 0.7,
        "timeout": 5,
        "user": "langfuse_latency_test_user",
        "mock_response": "It's simple to use and easy to get started",
    }
    completion_args.update(completion_kwargs)
    return asyncio.create_task(litellm.acompletion(**completion_args))
