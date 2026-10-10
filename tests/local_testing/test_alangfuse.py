import asyncio
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

logging.basicConfig(level=logging.DEBUG)

import litellm
from litellm.integrations.langfuse.langfuse_sdk import resolve_trace_id

litellm.num_retries = 3
litellm.success_callback = ["langfuse"]
os.environ["LANGFUSE_DEBUG"] = "True"

import pytest


def search_logs(log_file_path, num_good_logs=1):
    """
    Searches the given log file for logs containing the "/api/public" string.

    Parameters:
    - log_file_path (str): The path to the log file to be searched.

    Returns:
    - None

    Raises:
    - Exception: If there are any bad logs found in the log file.
    """
    import re

    print("\n searching logs")
    bad_logs = []
    good_logs = []
    all_logs = []
    try:
        with open(log_file_path, "r") as log_file:
            lines = log_file.readlines()
            print(f"searching logslines: {lines}")
            for line in lines:
                all_logs.append(line.strip())
                if "/api/public" in line:
                    print("Found log with /api/public:")
                    print(line.strip())
                    print("\n\n")
                    match = re.search(
                        r'"POST /api/public/ingestion HTTP/1.1" (\d+) (\d+)',
                        line,
                    )
                    if match:
                        status_code = int(match.group(1))
                        print("STATUS CODE", status_code)
                        if (
                            status_code != 200
                            and status_code != 201
                            and status_code != 207
                        ):
                            print("got a BAD log")
                            bad_logs.append(line.strip())
                        else:
                            good_logs.append(line.strip())
        print("\nBad Logs")
        print(bad_logs)
        if len(bad_logs) > 0:
            raise Exception(f"bad logs, Bad logs = {bad_logs}")
        assert (
            len(good_logs) == num_good_logs
        ), f"Did not get expected number of good logs, expected {num_good_logs}, got {len(good_logs)}. All logs \n {all_logs}"
        print("\nGood Logs")
        print(good_logs)
        if len(good_logs) <= 0:
            raise Exception(
                f"There were no Good Logs from Langfuse. No logs with /api/public status 200. \nAll logs:{all_logs}"
            )

    except Exception as e:
        raise e


def pre_langfuse_setup():
    """
    Set up the logging for the 'pre_langfuse_setup' function.
    """
    # sends logs to langfuse.log
    import logging

    # Configure the logging to write to a file
    logging.basicConfig(filename="langfuse.log", level=logging.DEBUG)
    logger = logging.getLogger()

    # Add a FileHandler to the logger
    file_handler = logging.FileHandler("langfuse.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return


def test_langfuse_logging_async():
    # this tests time added to make langfuse logging calls, vs just acompletion calls
    try:
        pre_langfuse_setup()
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
        "model": "azure/gpt-4.1-mini",
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


def _otlp_capture(exports: list[bytes]) -> type[BaseHTTPRequestHandler]:
    class OtlpCapture(BaseHTTPRequestHandler):
        def do_POST(self):
            exports.append(self.rfile.read(int(self.headers.get("content-length", 0))))
            self.send_response(200)
            self.end_headers()

        def do_GET(self):
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    return OtlpCapture


@pytest.fixture
def local_langfuse():
    exports: list[bytes] = []
    server = HTTPServer(("127.0.0.1", 0), _otlp_capture(exports))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", exports
    server.shutdown()


def _exported_spans(exports: list[bytes]):
    for body in exports:
        for resource_spans in ExportTraceServiceRequest.FromString(body).resource_spans:
            for scope_spans in resource_spans.scope_spans:
                yield from scope_spans.spans


def _exported_attributes(exports: list[bytes], trace_id: str) -> list[dict[str, str]]:
    return [
        {attribute.key: attribute.value.string_value for attribute in span.attributes}
        for span in _exported_spans(list(exports))
        if span.trace_id.hex() == trace_id
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_langfuse_logging_without_request_response(stream, local_langfuse, monkeypatch):
    from litellm._uuid import uuid

    langfuse_host, exports = local_langfuse
    prompt = f"prompt-{uuid.uuid4()}"
    answer = f"answer-{uuid.uuid4()}"
    trace_name = f"litellm-test-{uuid.uuid4()}"
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    monkeypatch.setattr(litellm, "success_callback", ["langfuse"])
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        mock_response=answer,
        stream=stream,
        metadata={"trace_id": trace_name},
        langfuse_public_key=f"pk-lf-{trace_name}",
        langfuse_secret_key="sk-lf-local",
        langfuse_host=langfuse_host,
    )
    if stream:
        async for _ in response:
            pass

    generations: list[dict[str, str]] = []
    for _ in range(60):
        generations = [
            attributes
            for attributes in _exported_attributes(exports, resolve_trace_id(trace_name))
            if attributes.get("langfuse.observation.type") == "generation"
        ]
        if generations:
            break
        await asyncio.sleep(0.5)

    assert len(generations) == 1, generations
    assert json.loads(generations[0]["langfuse.observation.input"]) == {
        "messages": [{"content": "redacted-by-litellm", "role": "user"}]
    }
    assert json.loads(generations[0]["langfuse.observation.output"])["content"] == "redacted-by-litellm"
    assert all(prompt.encode() not in body and answer.encode() not in body for body in exports)


# Get the current directory of the file being run
pwd = os.path.dirname(os.path.realpath(__file__))
print(pwd)

file_path = os.path.join(pwd, "gettysburg.wav")

audio_file = open(file_path, "rb")








# test_langfuse_logging()




# test_langfuse_logging_stream()




# test_langfuse_logging_custom_generation_name()






# test_langfuse_logging_function_calling()


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
        model="gpt-6-luna",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    print("\nLLM Response1:\n", response)
    response_message = response.choices[0].message
    tool_calls = response.choices[0].message.tool_calls
    assert response.choices[0].message.tool_calls
    assert all(call.function.name == "get_current_weather" for call in response.choices[0].message.tool_calls)


# test_langfuse_logging_tool_calling()






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
        "user_api_key": "sk-test-mock-api-key-123",
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

    from unittest.mock import Mock

    from litellm.integrations.langfuse.langfuse import _add_prompt_to_generation_params

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
        "user_api_key": "sk-test-mock-api-key-123",
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
