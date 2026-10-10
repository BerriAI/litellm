#### What this tests ####
#    This tests streaming for the completion endpoint

import asyncio
import json
import os
import time
import traceback
from typing import Final, Tuple

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
from litellm._uuid import uuid
from litellm.types.utils import ModelResponseStream

load_dotenv()

import litellm
from litellm import (
    AuthenticationError,
    ModelResponse,
    RateLimitError,
    acompletion,
    completion,
)

litellm.logging = False
litellm.set_verbose = True
litellm.num_retries = 3
litellm.cache = None

score = 0


def logger_fn(model_call_object: dict):
    print(f"model call details: {model_call_object}")


user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]


first_openai_chunk_example = {
    "id": "chatcmpl-7zSKLBVXnX9dwgRuDYVqVVDsgh2yp",
    "object": "chat.completion.chunk",
    "created": 1694881253,
    "model": "gpt-4-0613",
    "choices": [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,  # it's null
        }
    ],
}


def validate_first_format(chunk):
    # write a test to make sure chunk follows the same format as first_openai_chunk_example
    assert isinstance(chunk, ModelResponseStream), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."
    assert getattr(chunk, "usage", None) is None, "Chunk cannot contain usage"

    for choice in chunk["choices"]:
        assert isinstance(choice["index"], int), "'index' should be an integer."
        assert isinstance(
            choice["delta"]["role"], str
        ), f"'role' should be a string. Got {choice['delta']['role']}"
        assert "messages" not in choice
        # openai v1.0.0 returns content as None
        assert (choice["finish_reason"] is None) or isinstance(
            choice["finish_reason"], str
        ), "'finish_reason' should be None or a string."


second_openai_chunk_example = {
    "id": "chatcmpl-7zSKLBVXnX9dwgRuDYVqVVDsgh2yp",
    "object": "chat.completion.chunk",
    "created": 1694881253,
    "model": "gpt-4-0613",
    "choices": [
        {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}  # it's null
    ],
}


def validate_second_format(chunk):
    assert isinstance(chunk, ModelResponseStream), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."
    assert getattr(chunk, "usage", None) is None, "Chunk cannot contain usage"

    for choice in chunk["choices"]:
        assert isinstance(choice["index"], int), "'index' should be an integer."
        assert hasattr(choice["delta"], "role"), "'role' should be a string."
        # openai v1.0.0 returns content as None
        assert (choice["finish_reason"] is None) or isinstance(
            choice["finish_reason"], str
        ), "'finish_reason' should be None or a string."


last_openai_chunk_example = {
    "id": "chatcmpl-7zSKLBVXnX9dwgRuDYVqVVDsgh2yp",
    "object": "chat.completion.chunk",
    "created": 1694881253,
    "model": "gpt-4-0613",
    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
}

"""
Final chunk (sdk):
chunk: ChatCompletionChunk(id='chatcmpl-96mM3oNBlxh2FDWVLKsgaFBBcULmI', 
choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, 
tool_calls=None), finish_reason='stop', index=0, logprobs=None)], 
created=1711402871, model='gpt-3.5-turbo-0125', object='chat.completion.chunk', system_fingerprint='fp_3bc1b5746c')
"""


def validate_last_format(chunk):
    """
    Ensure last chunk has no remaining content or tools
    """
    assert isinstance(chunk, ModelResponseStream), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."
    assert getattr(chunk, "usage", None) is None, "Chunk cannot contain usage"

    for choice in chunk["choices"]:
        assert isinstance(choice["index"], int), "'index' should be an integer."
        assert choice["delta"]["content"] is None
        assert choice["delta"]["function_call"] is None
        assert choice["delta"]["role"] is None
        assert choice["delta"]["tool_calls"] is None
        assert isinstance(
            choice["finish_reason"], str
        ), "'finish_reason' should be a string."


def streaming_format_tests(idx, chunk) -> Tuple[str, bool]:
    extracted_chunk = ""
    finished = False
    print(f"chunk: {chunk}")
    if idx == 0:  # ensure role assistant is set
        validate_first_format(chunk=chunk)
        role = chunk["choices"][0]["delta"]["role"]
        assert role == "assistant"
    elif idx == 1:  # second chunk
        validate_second_format(chunk=chunk)
    if idx != 0:  # ensure no role
        if "role" in chunk["choices"][0]["delta"]:
            pass  # openai v1.0.0+ passes role = None
    if chunk["choices"][0][
        "finish_reason"
    ]:  # ensure finish reason is only in last chunk
        validate_last_format(chunk=chunk)
        finished = True
    if (
        "content" in chunk["choices"][0]["delta"]
        and chunk["choices"][0]["delta"]["content"] is not None
    ):
        extracted_chunk = chunk["choices"][0]["delta"]["content"]
    print(f"extracted chunk: {extracted_chunk}")
    return extracted_chunk, finished


tools_schema = [
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


def test_completion_azure_stream_special_char():
    litellm.set_verbose = True
    messages = [{"role": "user", "content": "hi. respond with the <xml> tag only"}]
    response = completion(model="azure/gpt-4.1-mini", messages=messages, stream=True)
    response_str = ""
    for part in response:
        response_str += part.choices[0].delta.content or ""
    print(f"response_str: {response_str}")
    assert len(response_str) > 0




@pytest.mark.flaky(retries=5, delay=1)
def test_completion_azure_stream():
    try:
        litellm.set_verbose = False
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        response = completion(
            model="azure/gpt-4.1-mini", messages=messages, stream=True, max_tokens=50
        )
        complete_response = ""
        # Add any assertions here to check the response
        for idx, init_chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, init_chunk)
            complete_response += chunk
            custom_llm_provider = init_chunk._hidden_params["custom_llm_provider"]
            print(f"custom_llm_provider: {custom_llm_provider}")
            assert custom_llm_provider == "azure"
            if finished:
                assert isinstance(init_chunk.choices[0], litellm.utils.StreamingChoices)
                break
        if complete_response.strip() == "":
            raise Exception("Empty response received")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")




@pytest.mark.parametrize(
    "model",
    [
        # "claude-3-5-haiku-20241022",
        # "mistral/mistral-small-latest",
        "openrouter/openai/gpt-4o-mini",
    ],
)
def test_completion_model_stream(model):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        response = completion(
            model=model, messages=messages, stream=True, max_tokens=50
        )
        complete_response = ""
        # Add any assertions here to check the response
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            if finished:
                break
            complete_response += chunk
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"completion_response: {complete_response}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)  # ,
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_completion_gemini_stream(sync_mode):
    try:
        litellm.turn_on_debug()
        print("Streaming gemini response")
        function1 = [
            {
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
            }
        ]
        messages = [
            {
                "role": "user",
                "content": "What is the weather like in Boston, MA?. You must provide me with a tool call in your response.",
            }
        ]
        print("testing gemini streaming")
        complete_response = ""
        # Add any assertions here to check the response
        non_empty_chunks = 0
        chunks = []
        if sync_mode:
            response = completion(
                model="gemini/gemini-2.5-flash-lite",
                messages=messages,
                stream=True,
                functions=function1,
            )

            for idx, chunk in enumerate(response):
                print(chunk)
                chunks.append(chunk)
                # print(chunk.choices[0].delta)
                chunk, finished = streaming_format_tests(idx, chunk)
                if finished:
                    break
                non_empty_chunks += 1
                complete_response += chunk
        else:
            response = await litellm.acompletion(
                model="gemini/gemini-2.5-flash-lite",
                messages=messages,
                stream=True,
                functions=function1,
            )

            idx = 0
            async for chunk in response:
                print(chunk)
                chunks.append(chunk)
                # print(chunk.choices[0].delta)
                chunk, finished = streaming_format_tests(idx, chunk)
                if finished:
                    break
                non_empty_chunks += 1
                complete_response += chunk
                idx += 1

        # if complete_response.strip() == "":
        #     raise Exception("Empty response received")
        print(f"completion_response: {complete_response}")

        complete_response = litellm.stream_chunk_builder(
            chunks=chunks, messages=messages
        )

        assert complete_response.choices[0].message.function_call is not None

        # assert non_empty_chunks > 1
    except litellm.InternalServerError as e:
        pass
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        # if "429 Resource has been exhausted":
        #     return
        pytest.fail(f"Error occurred: {e}")






@pytest.mark.flaky(retries=3, delay=1)
def test_completion_mistral_api_mistral_large_function_call_with_streaming():
    litellm.set_verbose = True
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
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston today in fahrenheit?",
        }
    ]
    try:
        # test without max tokens
        response = completion(
            model="mistral/mistral-medium-latest",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )
        saw_function_call_chunk = False
        for chunk in response:
            print(f"chunk in response: {chunk}")
            assert chunk._hidden_params["custom_llm_provider"] == "mistral"
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].finish_reason is not None:  # last chunk
                validate_final_streaming_function_calling_chunk(chunk=chunk)
                break
            tool_calls = chunk.choices[0].delta.tool_calls
            if tool_calls is None:
                continue
            assert tool_calls[0].function.arguments is not None
            assert isinstance(tool_calls[0].function.arguments, str)
            if not saw_function_call_chunk:
                if chunk.choices[0].delta.role is not None:
                    validate_first_streaming_function_calling_chunk(chunk=chunk)
                else:
                    validate_second_streaming_function_calling_chunk(chunk=chunk)
                saw_function_call_chunk = True
            else:
                validate_second_streaming_function_calling_chunk(chunk=chunk)
        assert saw_function_call_chunk
    except litellm.RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")




def test_completion_claude_stream_bad_key():
    try:
        litellm.cache = None
        litellm.set_verbose = True
        api_key = "bad-key"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        response = completion(
            model="claude-haiku-4-5-20251001",
            messages=messages,
            stream=True,
            max_tokens=50,
            api_key=api_key,
        )
        complete_response = ""
        # Add any assertions here to check the response
        has_finish_reason = False
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            if finished:
                has_finish_reason = True
                break
            complete_response += chunk
        if has_finish_reason == False:
            raise Exception("finish reason not set")
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"1234completion_response: {complete_response}")
        raise Exception("Auth error not raised")
    except AuthenticationError as e:
        print("Auth Error raised")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("provider", ["vertex_ai_beta"])  # ""
def test_vertex_ai_stream(provider):
    from test_amazing_vertex_completion import (
        load_vertex_ai_credentials,
    )

    load_vertex_ai_credentials()
    litellm.set_verbose = True

    test_models = ["gemini-2.5-flash-lite"]
    for model in test_models:
        try:
            print("making request", model)
            response = completion(
                model="{}/{}".format(provider, model),
                messages=[
                    {"role": "user", "content": "Hey, how's it going?"},
                    {
                        "role": "assistant",
                        "content": "I'm doing well. Would like to hear the rest of the story?",
                    },
                    {"role": "user", "content": "Na"},
                    {
                        "role": "assistant",
                        "content": "No problem, is there anything else i can help you with today?",
                    },
                    {
                        "role": "user",
                        "content": "I think you're getting cut off sometimes",
                    },
                ],
                stream=True,
            )
            complete_response = ""
            is_finished = False
            for idx, chunk in enumerate(response):
                print(f"chunk in response: {chunk}")
                chunk, finished = streaming_format_tests(idx, chunk)
                if finished:
                    is_finished = True
                    break
                complete_response += chunk
            if complete_response.strip() == "":
                raise Exception("Empty response received")
            print(f"completion_response: {complete_response}")
            assert is_finished == True

        except litellm.RateLimitError as e:
            pass
        except litellm.exceptions.MidStreamFallbackError as e:
            # Streaming 429s are wrapped in MidStreamFallbackError so the
            # Router can fall back; treat as a transient rate-limit pass.
            if not isinstance(e.original_exception, litellm.RateLimitError):
                pytest.fail(f"Error occurred: {e}")
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")




@pytest.mark.parametrize("sync_mode", [True, False])  #
@pytest.mark.parametrize(
    "model, region",
    [
        # ["bedrock/ai21.jamba-instruct-v1:0", "us-east-1"],
        ["us.anthropic.claude-sonnet-4-5-20250929-v1:0", None],
        # ["mistral.mistral-7b-instruct-v0:2", None],
        # ["meta.llama3-8b-instruct-v1:0", None],
    ],
)
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_bedrock_httpx_streaming(sync_mode, model, region):
    try:
        litellm.set_verbose = True
        if sync_mode:
            final_chunk: Optional[litellm.ModelResponse] = None
            response: litellm.CustomStreamWrapper = completion(  # type: ignore
                model=model,
                messages=messages,
                max_tokens=10,  # type: ignore
                stream=True,
                aws_region_name=region,
            )
            complete_response = ""
            # Add any assertions here to check the response
            has_finish_reason = False
            for idx, chunk in enumerate(response):
                final_chunk = chunk
                chunk, finished = streaming_format_tests(idx, chunk)
                if finished:
                    has_finish_reason = True
                    break
                complete_response += chunk
            if has_finish_reason is False:
                raise Exception("finish reason not set")
            if complete_response.strip() == "":
                raise Exception("Empty response received")
        else:
            response: litellm.CustomStreamWrapper = await litellm.acompletion(  # type: ignore
                model=model,
                messages=messages,
                max_tokens=100,  # type: ignore
                stream=True,
                aws_region_name=region,
            )
            complete_response = ""
            # Add any assertions here to check the response
            has_finish_reason = False
            idx = 0
            final_chunk: Optional[litellm.ModelResponse] = None
            async for chunk in response:
                final_chunk = chunk
                chunk, finished = streaming_format_tests(idx, chunk)
                if finished:
                    has_finish_reason = True
                    break
                complete_response += chunk
                idx += 1
            if has_finish_reason == False:
                raise Exception("finish reason not set")
            if complete_response.strip() == "":
                raise Exception("Empty response received")
        print(f"completion_response: {complete_response}\n\nFinalChunk: {final_chunk}")
    except RateLimitError as e:
        print("got rate limit error=", e)
        pass
    except litellm.Timeout:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_bedrock_claude_3_streaming():
    try:
        litellm.set_verbose = True
        response: ModelResponse = completion(  # type: ignore
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=messages,
            max_tokens=10,  # type: ignore
            stream=True,
        )
        complete_response = ""
        # Add any assertions here to check the response
        has_finish_reason = False
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            if finished:
                has_finish_reason = True
                break
            complete_response += chunk
        if has_finish_reason == False:
            raise Exception("finish reason not set")
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"completion_response: {complete_response}")
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize(
    "model",
    [
        "claude-haiku-4-5-20251001",
        "bedrock/mistral.mistral-7b-instruct-v0:2",
        "gpt-3.5-turbo",
    ],
)
@pytest.mark.asyncio
async def test_parallel_streaming_requests(sync_mode, model):
    """
    Important prod test.
    """
    try:
        import threading

        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "What do you know?"},
        ]

        def sync_test_streaming():
            response: litellm.CustomStreamWrapper = litellm.completion(  # type: ignore
                model=model,
                messages=messages,
                stream=True,
                max_tokens=10,
                timeout=10,
            )
            complete_response = ""
            # Add any assertions here to-check the response
            num_finish_reason = 0
            for chunk in response:
                print(f"chunk: {chunk}")
                if isinstance(chunk, ModelResponseStream):
                    if chunk.choices[0].finish_reason is not None:
                        num_finish_reason += 1
            assert num_finish_reason == 1

        async def test_streaming():
            response: litellm.CustomStreamWrapper = await litellm.acompletion(  # type: ignore
                model=model,
                messages=messages,
                stream=True,
                max_tokens=10,
                timeout=10,
            )
            complete_response = ""
            # Add any assertions here to-check the response
            num_finish_reason = 0
            async for chunk in response:
                print(f"type of chunk: {type(chunk)}")
                if isinstance(chunk, ModelResponseStream):
                    print(f"OUTSIDE CHUNK: {chunk.choices[0]}")
                    if chunk.choices[0].finish_reason is not None:
                        num_finish_reason += 1
            assert num_finish_reason == 1

        tasks = []
        for _ in range(2):
            if sync_mode == False:
                tasks.append(test_streaming())
            else:
                thread = threading.Thread(target=sync_test_streaming)
                thread.start()
                tasks.append(thread)

        if sync_mode == False:
            await asyncio.gather(*tasks)
        else:
            # Wait for all threads to complete
            for thread in tasks:
                thread.join()

    except RateLimitError:
        pass
    except litellm.Timeout:
        pass
    except litellm.ServiceUnavailableError as e:
        if model == "predibase/llama-3-8b-instruct":
            pass
        else:
            pytest.fail(f"Service Unavailable Error got{str(e)}")
    except litellm.InternalServerError as e:
        if "predibase" in str(e).lower():
            # only skip internal server error from predibase - their endpoint seems quite unstable
            pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")






def test_completion_bedrock_mistral_stream():
    try:
        litellm.set_verbose = False
        response = completion(
            model="bedrock/mistral.mixtral-8x7b-instruct-v0:1",
            messages=[
                {
                    "role": "user",
                    "content": "Be as verbose as possible and give as many details as possible, how does a court case get to the Supreme Court?",
                }
            ],
            temperature=1,
            max_tokens=20,
            stream=True,
        )
        print(response)
        complete_response = ""
        has_finish_reason = False
        # Add any assertions here to check the response
        for idx, chunk in enumerate(response):
            # print
            chunk, finished = streaming_format_tests(idx, chunk)
            has_finish_reason = finished
            complete_response += chunk
            if finished:
                break
        if has_finish_reason is False:
            raise Exception("finish reason not set for last chunk")
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"completion_response: {complete_response}")
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")








# test on openai completion call
def test_openai_chat_completion_call():
    litellm.set_verbose = False
    litellm.return_response_headers = True
    response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
    assert isinstance(
        response._hidden_params["additional_headers"][
            "llm_provider-x-ratelimit-remaining-requests"
        ],
        str,
    )

    print(f"response._hidden_params: {response._hidden_params}")
    complete_response = ""
    start_time = time.time()
    for idx, chunk in enumerate(response):
        chunk, finished = streaming_format_tests(idx, chunk)
        print(f"outside chunk: {chunk}")
        if finished:
            break
        complete_response += chunk
        # print(f'complete_chunk: {complete_response}')
    if complete_response.strip() == "":
        raise Exception("Empty response received")
    print(f"complete response: {complete_response}")


@pytest.mark.parametrize(
    "model",
    [
        "gpt-3.5-turbo",
        "claude-haiku-4-5-20251001",
        "o1",
    ],
)
@pytest.mark.parametrize(
    "sync",
    [True, False],
)
@pytest.mark.asyncio
@pytest.mark.flaky(retries=6, delay=10)
async def test_openai_stream_options_call(model, sync):
    litellm.enable_preview_features = True
    litellm.set_verbose = True
    usage = None
    chunks = []
    if sync:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "user", "content": "say GM - we're going to make it "},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in response:
            print("chunk: ", chunk)
            chunks.append(chunk)
    else:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "say GM - we're going to make it "}],
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in response:
            print("chunk: ", chunk)
            chunks.append(chunk)

    last_chunk = chunks[-1]
    print("last chunk: ", last_chunk)

    """
    Assert that:
    - Last Chunk includes Usage
    - All chunks prior to last chunk have usage=None
    """

    assert last_chunk.usage is not None
    assert isinstance(last_chunk.usage, litellm.Usage)
    assert last_chunk.usage.total_tokens > 0
    assert last_chunk.usage.prompt_tokens > 0
    assert last_chunk.usage.completion_tokens > 0

    # assert all non last chunks have usage=None
    # Improved assertion with detailed error message
    non_last_chunks_with_usage = [
        chunk
        for chunk in chunks[:-1]
        if hasattr(chunk, "usage") and chunk.usage is not None
    ]
    assert (
        not non_last_chunks_with_usage
    ), f"Non-last chunks with usage not None:\n" + "\n".join(
        f"Chunk ID: {chunk.id}, Usage: {chunk.usage}, Content: {chunk.choices[0].delta.content}"
        for chunk in non_last_chunks_with_usage
    )


def test_openai_stream_options_call_text_completion() -> None:
    chunks: Final = tuple(
        litellm.text_completion(
            model="gpt-6-luna",
            reasoning_effort="none",
            prompt="say GM - we're going to make it ",
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=10,
        )
    )
    assert chunks
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens > 0
    assert chunks[-1].usage.prompt_tokens > 0
    assert chunks[-1].usage.completion_tokens > 0
    assert all(chunk.usage is None for chunk in chunks[:-1])
    assert any(chunk.choices[0].text for chunk in chunks)




#### Test Function calling + streaming ####


def test_completion_openai_with_functions() -> None:
    functions: Final = [
        {
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
        }
    ]
    messages: Final = [{"role": "user", "content": "what's the weather in SF"}]
    chunks: Final = tuple(
        completion(
            model="gpt-6-luna",
            reasoning_effort="none",
            messages=messages,
            functions=functions,
            function_call={"name": "get_current_weather"},
            stream=True,
            max_tokens=128,
        )
    )
    response: Final = litellm.stream_chunk_builder(chunks, messages=messages)
    assert response is not None
    function_call: Final = response.choices[0].message.function_call
    assert function_call is not None
    assert function_call.name == "get_current_weather"
    assert json.loads(function_call.arguments)["location"]
    assert sum(chunk.choices[0].finish_reason is not None for chunk in chunks) == 1


#### Test Async streaming ####


async def completion_call():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
            logger_fn=logger_fn,
            max_tokens=10,
        )
        print(f"response: {response}")
        complete_response = ""
        start_time = time.time()
        # Change for loop to async for loop
        idx = 0
        async for chunk in response:
            chunk, finished = streaming_format_tests(idx, chunk)
            if finished:
                break
            complete_response += chunk
            idx += 1
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"complete response: {complete_response}")
    except Exception:
        print(f"error occurred: {traceback.format_exc()}")
        pass


#### Test Function Calling + Streaming ####

final_openai_function_call_example = {
    "id": "chatcmpl-7zVNA4sXUftpIg6W8WlntCyeBj2JY",
    "object": "chat.completion",
    "created": 1694892960,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "get_current_weather",
                    "arguments": '{\n  "location": "Boston, MA"\n}',
                },
            },
            "finish_reason": "function_call",
        }
    ],
    "usage": {"prompt_tokens": 82, "completion_tokens": 18, "total_tokens": 100},
}

function_calling_output_structure = {
    "id": str,
    "object": str,
    "created": int,
    "model": str,
    "choices": [
        {
            "index": int,
            "message": {
                "role": str,
                "content": (type(None), str),
                "function_call": {"name": str, "arguments": str},
            },
            "finish_reason": str,
        }
    ],
    "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
}


def validate_final_structure(item, structure=function_calling_output_structure):
    if isinstance(item, list):
        if not all(validate_final_structure(i, structure[0]) for i in item):
            return Exception(
                "Function calling final output doesn't match expected output format"
            )
    elif isinstance(item, dict):
        if not all(
            k in item and validate_final_structure(item[k], v)
            for k, v in structure.items()
        ):
            return Exception(
                "Function calling final output doesn't match expected output format"
            )
    else:
        if not isinstance(item, structure):
            return Exception(
                "Function calling final output doesn't match expected output format"
            )
    return True


first_openai_function_call_example = {
    "id": "chatcmpl-7zVRoE5HjHYsCMaVSNgOjzdhbS3P0",
    "object": "chat.completion.chunk",
    "created": 1694893248,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "get_current_weather", "arguments": ""},
            },
            "finish_reason": None,
        }
    ],
}


def validate_first_function_call_chunk_structure(item):
    if not (isinstance(item, dict) or isinstance(item, litellm.ModelResponse)):
        raise Exception(f"Incorrect format, type of item: {type(item)}")

    required_keys = {"id", "object", "created", "model", "choices"}
    for key in required_keys:
        if key not in item:
            raise Exception("Incorrect format")

    if not isinstance(item["choices"], list) or not item["choices"]:
        raise Exception("Incorrect format")

    required_keys_in_choices_array = {"index", "delta", "finish_reason"}
    for choice in item["choices"]:
        if not (
            isinstance(choice, dict)
            or isinstance(choice, litellm.utils.StreamingChoices)
        ):
            raise Exception(f"Incorrect format, type of choice: {type(choice)}")
        for key in required_keys_in_choices_array:
            if key not in choice:
                raise Exception("Incorrect format")

        if not (
            isinstance(choice["delta"], dict)
            or isinstance(choice["delta"], litellm.utils.Delta)
        ):
            raise Exception(
                f"Incorrect format, type of choice: {type(choice['delta'])}"
            )

        required_keys_in_delta = {"role", "content", "function_call"}
        for key in required_keys_in_delta:
            if key not in choice["delta"]:
                raise Exception("Incorrect format")

        if not (
            isinstance(choice["delta"]["function_call"], dict)
            or isinstance(choice["delta"]["function_call"], BaseModel)
        ):
            raise Exception(
                f"Incorrect format, type of function call: {type(choice['delta']['function_call'])}"
            )

        required_keys_in_function_call = {"name", "arguments"}
        for key in required_keys_in_function_call:
            if not hasattr(choice["delta"]["function_call"], key):
                raise Exception(
                    f"Incorrect format, expected key={key};  actual keys: {choice['delta']['function_call']}, eval: {hasattr(choice['delta']['function_call'], key)}"
                )

    return True


second_function_call_chunk_format = {
    "id": "chatcmpl-7zVRoE5HjHYsCMaVSNgOjzdhbS3P0",
    "object": "chat.completion.chunk",
    "created": 1694893248,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "delta": {"function_call": {"arguments": "{\n"}},
            "finish_reason": None,
        }
    ],
}


def validate_second_function_call_chunk_structure(data):
    if not isinstance(data, dict):
        raise Exception("Incorrect format")

    required_keys = {"id", "object", "created", "model", "choices"}
    for key in required_keys:
        if key not in data:
            raise Exception("Incorrect format")

    if not isinstance(data["choices"], list) or not data["choices"]:
        raise Exception("Incorrect format")

    required_keys_in_choices_array = {"index", "delta", "finish_reason"}
    for choice in data["choices"]:
        if not isinstance(choice, dict):
            raise Exception("Incorrect format")
        for key in required_keys_in_choices_array:
            if key not in choice:
                raise Exception("Incorrect format")

        if (
            "function_call" not in choice["delta"]
            or "arguments" not in choice["delta"]["function_call"]
        ):
            raise Exception("Incorrect format")

    return True


final_function_call_chunk_example = {
    "id": "chatcmpl-7zVRoE5HjHYsCMaVSNgOjzdhbS3P0",
    "object": "chat.completion.chunk",
    "created": 1694893248,
    "model": "gpt-3.5-turbo",
    "choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}],
}


def validate_final_function_call_chunk_structure(data):
    if not (isinstance(data, dict) or isinstance(data, litellm.ModelResponse)):
        raise Exception("Incorrect format")

    required_keys = {"id", "object", "created", "model", "choices"}
    for key in required_keys:
        if key not in data:
            raise Exception("Incorrect format")

    if not isinstance(data["choices"], list) or not data["choices"]:
        raise Exception("Incorrect format")

    required_keys_in_choices_array = {"index", "delta", "finish_reason"}
    for choice in data["choices"]:
        if not (
            isinstance(choice, dict) or isinstance(choice["delta"], litellm.utils.Delta)
        ):
            raise Exception("Incorrect format")
        for key in required_keys_in_choices_array:
            if key not in choice:
                raise Exception("Incorrect format")

    return True


def streaming_and_function_calling_format_tests(idx, chunk):
    extracted_chunk = ""
    finished = False
    print(f"idx: {idx}")
    print(f"chunk: {chunk}")
    decision = False
    if idx == 0:  # ensure role assistant is set
        decision = validate_first_function_call_chunk_structure(chunk)
        role = chunk["choices"][0]["delta"]["role"]
        assert role == "assistant"
    elif idx != 0:  # second chunk
        try:
            decision = validate_second_function_call_chunk_structure(data=chunk)
        except Exception:  # check if it's the last chunk (returns an empty delta {} )
            decision = validate_final_function_call_chunk_structure(data=chunk)
            finished = True
    if "content" in chunk["choices"][0]["delta"]:
        extracted_chunk = chunk["choices"][0]["delta"]["content"]
    if decision == False:
        raise Exception("incorrect format")
    return extracted_chunk, finished


@pytest.mark.parametrize(
    "model",
    [
        # "gpt-3.5-turbo",
        # "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-haiku-4-5-20251001",
    ],
)
def test_streaming_and_function_calling(model):
    import json

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

    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    try:
        # litellm.set_verbose = True
        response: litellm.CustomStreamWrapper = completion(
            model=model,
            tools=tools,
            messages=messages,
            stream=True,
            tool_choice="required",
        )  # type: ignore
        # Add any assertions here to check the response
        json_str = ""
        for idx, chunk in enumerate(response):
            # continue
            # print("\n{}\n".format(chunk))
            if idx == 0:
                assert (
                    chunk.choices[0].delta.tool_calls[0].function.arguments is not None
                )
                assert isinstance(
                    chunk.choices[0].delta.tool_calls[0].function.arguments, str
                )
            if chunk.choices[0].delta.tool_calls is not None:
                json_str += chunk.choices[0].delta.tool_calls[0].function.arguments

        print(json.loads(json_str))
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        raise e


def test_success_callback_streaming():
    def success_callback(kwargs, completion_response, start_time, end_time):
        print(
            {
                "success": True,
                "input": kwargs,
                "output": completion_response,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

    litellm.success_callback = [success_callback]

    messages = [{"role": "user", "content": "hello"}]
    print("TESTING LITELLM COMPLETION CALL")
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
        max_tokens=5,
    )
    print(response)

    for chunk in response:
        print(chunk["choices"][0])


from typing import List, Optional

#### STREAMING + FUNCTION CALLING ###


class Function(BaseModel):
    name: str
    arguments: str


class ToolCalls(BaseModel):
    index: int
    id: str
    type: str
    function: Function


class Delta(BaseModel):
    role: str
    content: Optional[str]
    tool_calls: List[ToolCalls]


class Choices(BaseModel):
    index: int
    delta: Delta
    logprobs: Optional[str]
    finish_reason: Optional[str]


class Chunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    # system_fingerprint: str
    choices: List[Choices]


def validate_first_streaming_function_calling_chunk(chunk: ModelResponse):
    chunk_instance = Chunk(**chunk.model_dump())


### Chunk 1


# {
#     "id": "chatcmpl-8vdVjtzxc0JqGjq93NxC79dMp6Qcs",
#     "object": "chat.completion.chunk",
#     "created": 1708747267,
#     "model": "gpt-3.5-turbo-0125",
#     "system_fingerprint": "fp_86156a94a0",
#     "choices": [
#         {
#             "index": 0,
#             "delta": {
#                 "role": "assistant",
#                 "content": null,
#                 "tool_calls": [
#                     {
#                         "index": 0,
#                         "id": "call_oN10vaaC9iA8GLFRIFwjCsN7",
#                         "type": "function",
#                         "function": {
#                             "name": "get_current_weather",
#                             "arguments": ""
#                         }
#                     }
#                 ]
#             },
#             "logprobs": null,
#             "finish_reason": null
#         }
#     ]
# }
class Function2(BaseModel):
    arguments: str


class ToolCalls2(BaseModel):
    index: int
    function: Optional[Function2]


class Delta2(BaseModel):
    tool_calls: List[ToolCalls2]


class Choices2(BaseModel):
    index: int
    delta: Delta2
    logprobs: Optional[str]
    finish_reason: Optional[str]


class Chunk2(BaseModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: Optional[str]
    choices: List[Choices2]


## Chunk 2

# {
#     "id": "chatcmpl-8vdVjtzxc0JqGjq93NxC79dMp6Qcs",
#     "object": "chat.completion.chunk",
#     "created": 1708747267,
#     "model": "gpt-3.5-turbo-0125",
#     "system_fingerprint": "fp_86156a94a0",
#     "choices": [
#         {
#             "index": 0,
#             "delta": {
#                 "tool_calls": [
#                     {
#                         "index": 0,
#                         "function": {
#                             "arguments": "{\""
#                         }
#                     }
#                 ]
#             },
#             "logprobs": null,
#             "finish_reason": null
#         }
#     ]
# }


def validate_second_streaming_function_calling_chunk(chunk: ModelResponse):
    chunk_instance = Chunk2(**chunk.model_dump())


class Delta3(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None
    function_call: Optional[dict] = None
    tool_calls: Optional[List] = None


class Choices3(BaseModel):
    index: int
    delta: Delta3
    logprobs: Optional[str]
    finish_reason: str


class Chunk3(BaseModel):
    id: str
    object: str
    created: int
    model: str
    # system_fingerprint: str
    choices: List[Choices3]


def validate_final_streaming_function_calling_chunk(chunk: ModelResponse):
    chunk_instance = Chunk3(**chunk.model_dump())


def test_azure_streaming_and_function_calling():
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
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    try:
        response = completion(
            model="azure/gpt-4.1-mini",
            tools=tools,
            tool_choice="auto",
            messages=messages,
            stream=True,
            api_base=os.getenv("AZURE_AI_API_BASE"),
            api_key=os.getenv("AZURE_AI_API_KEY"),
            api_version="2024-02-15-preview",
        )
        # Add any assertions here to check the response
        for idx, chunk in enumerate(response):
            print(f"chunk: {chunk}")
            if idx == 0:
                assert (
                    chunk.choices[0].delta.tool_calls[0].function.arguments is not None
                )
                assert isinstance(
                    chunk.choices[0].delta.tool_calls[0].function.arguments, str
                )
                validate_first_streaming_function_calling_chunk(chunk=chunk)
            elif idx == 1:
                validate_second_streaming_function_calling_chunk(chunk=chunk)
            elif chunk.choices[0].finish_reason is not None:  # last chunk
                validate_final_streaming_function_calling_chunk(chunk=chunk)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        raise e


@pytest.mark.asyncio
async def test_azure_astreaming_and_function_calling():

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
    messages = [
        {
            "role": "user",
            "content": f"What is the weather like in Boston? {uuid.uuid4()}",
        }
    ]
    from litellm.caching.caching import Cache

    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
    )
    try:
        litellm.set_verbose = True
        response = await litellm.acompletion(
            model="azure/gpt-4.1-mini",
            tools=tools,
            tool_choice="auto",
            messages=messages,
            stream=True,
            api_base=os.getenv("AZURE_AI_API_BASE"),
            api_key=os.getenv("AZURE_AI_API_KEY"),
            api_version="2024-02-15-preview",
            caching=True,
        )
        # Add any assertions here to check the response
        idx = 0
        async for chunk in response:
            print(f"chunk: {chunk}")
            if idx == 0:
                assert (
                    chunk.choices[0].delta.tool_calls[0].function.arguments is not None
                )
                assert isinstance(
                    chunk.choices[0].delta.tool_calls[0].function.arguments, str
                )
                validate_first_streaming_function_calling_chunk(chunk=chunk)
            elif idx == 1:
                validate_second_streaming_function_calling_chunk(chunk=chunk)
            elif chunk.choices[0].finish_reason is not None:  # last chunk
                validate_final_streaming_function_calling_chunk(chunk=chunk)
            idx += 1

        ## CACHING TEST
        print("\n\nCACHING TESTS\n\n")
        response = await litellm.acompletion(
            model="azure/gpt-4.1-mini",
            tools=tools,
            tool_choice="auto",
            messages=messages,
            stream=True,
            api_base=os.getenv("AZURE_AI_API_BASE"),
            api_key=os.getenv("AZURE_AI_API_KEY"),
            api_version="2024-02-15-preview",
            caching=True,
        )
        # Add any assertions here to check the response
        idx = 0
        async for chunk in response:
            print(f"chunk: {chunk}")
            if idx == 0:
                assert (
                    chunk.choices[0].delta.tool_calls[0].function.arguments is not None
                )
                assert isinstance(
                    chunk.choices[0].delta.tool_calls[0].function.arguments, str
                )
                validate_first_streaming_function_calling_chunk(chunk=chunk)
            elif idx == 1 and chunk.choices[0].finish_reason is None:
                validate_second_streaming_function_calling_chunk(chunk=chunk)
            elif chunk.choices[0].finish_reason is not None:  # last chunk
                validate_final_streaming_function_calling_chunk(chunk=chunk)
            idx += 1
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        raise e


def test_completion_claude_3_function_call_with_streaming():
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
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston today in fahrenheit?",
        }
    ]
    try:
        # test without max tokens
        response = completion(
            model="claude-haiku-4-5-20251001",
            messages=messages,
            tools=tools,
            tool_choice="required",
            stream=True,
        )
        idx = 0
        for chunk in response:
            print(f"chunk in response: {chunk}")
            if idx == 0:
                assert (
                    chunk.choices[0].delta.tool_calls[0].function.arguments is not None
                )
                assert isinstance(
                    chunk.choices[0].delta.tool_calls[0].function.arguments, str
                )
                validate_first_streaming_function_calling_chunk(chunk=chunk)
            elif idx == 1 and chunk.choices[0].finish_reason is None:
                validate_second_streaming_function_calling_chunk(chunk=chunk)
            elif chunk.choices[0].finish_reason is not None:  # last chunk
                assert "usage" in chunk._hidden_params
                validate_final_streaming_function_calling_chunk(chunk=chunk)
            idx += 1
        # raise Exception("it worked!")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "model",
    [
        "gemini/gemini-2.5-flash-lite",
    ],
)  #
@pytest.mark.asyncio
async def test_acompletion_function_call_with_streaming(model):
    litellm.set_verbose = True
    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_series_of_questions",
                "description": "Generate a series of questions, given a topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "description": "The questions to be generated.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["questions"],
                },
            },
        },
    ]
    SYSTEM_PROMPT = "You are an AI assistant"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Generate 3 questions about civil engineering.",
        },
    ]
    try:
        # test without max tokens
        response = await acompletion(
            model=model,
            # model="claude-3-5-sonnet-20240620",
            messages=messages,
            stream=True,
            temperature=0.75,
            tools=tools,
            stream_options={"include_usage": True},
        )
        idx = 0
        print(f"response: {response}")
        async for chunk in response:
            print(f"chunk in test: {chunk}")
            if idx == 0:
                assert (
                    chunk.choices[0].delta.tool_calls[0].function.arguments is not None
                )
                assert isinstance(
                    chunk.choices[0].delta.tool_calls[0].function.arguments, str
                )
                validate_first_streaming_function_calling_chunk(chunk=chunk)
            elif idx == 1 and chunk.choices[0].finish_reason is None:
                validate_second_streaming_function_calling_chunk(chunk=chunk)
            elif chunk.choices[0].finish_reason is not None:  # last chunk
                validate_final_streaming_function_calling_chunk(chunk=chunk)
            idx += 1
        # raise Exception("it worked! ")
    except litellm.InternalServerError as e:
        pytest.skip(f"InternalServerError - {str(e)}")
    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


















@pytest.mark.parametrize(
    "model",
    [
        "gpt-3.5-turbo",
        "claude-sonnet-4-5-20250929",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        # "vertex_ai/claude-3-5-sonnet@20240620",
    ],
)
@pytest.mark.flaky(retries=3, delay=1)
def test_aastreaming_tool_calls_valid_json_str(model):
    if "vertex_ai" in model:
        from test_amazing_vertex_completion import (
            load_vertex_ai_credentials,
        )

        load_vertex_ai_credentials()
        vertex_location = "us-east5"
    else:
        vertex_location = None
    litellm.set_verbose = False
    messages = [
        {"role": "user", "content": "Hit the snooze button."},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "snooze",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]

    stream = litellm.completion(
        model, messages, tools=tools, stream=True, vertex_location=vertex_location
    )
    chunks = [*stream]
    print(f"chunks: {chunks}")
    tool_call_id_arg_map = {}
    curr_tool_call_id = None
    curr_tool_call_str = ""
    for chunk in chunks:
        if chunk.choices[0].delta.tool_calls is not None:
            if chunk.choices[0].delta.tool_calls[0].id is not None:
                # flush prev tool call
                if curr_tool_call_id is not None:
                    tool_call_id_arg_map[curr_tool_call_id] = curr_tool_call_str
                    curr_tool_call_str = ""
                curr_tool_call_id = chunk.choices[0].delta.tool_calls[0].id
                tool_call_id_arg_map[curr_tool_call_id] = ""
            if chunk.choices[0].delta.tool_calls[0].function.arguments is not None:
                curr_tool_call_str += (
                    chunk.choices[0].delta.tool_calls[0].function.arguments
                )
    # flush prev tool call
    if curr_tool_call_id is not None:
        tool_call_id_arg_map[curr_tool_call_id] = curr_tool_call_str

    for k, v in tool_call_id_arg_map.items():
        print("k={}, v={}".format(k, v))
        json.loads(v)  # valid json str


def test_streaming_api_base():
    litellm.set_verbose = False
    stream = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey"}],
        stream=True,
    )
    assert "https://api.openai.com" in stream._hidden_params["api_base"]




@pytest.mark.parametrize(
    "model",
    [
        # "deepseek/deepseek-reasoner",
        # "anthropic/claude-3-7-sonnet-20250219",
        "openrouter/anthropic/claude-sonnet-4.5",
    ],
)
def test_reasoning_content_completion(model):
    # litellm.set_verbose = True
    try:
        # litellm.turn_on_debug()
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Tell me a joke."}],
            stream=True,
            # thinking={"type": "enabled", "budget_tokens": 1024},
            reasoning={"effort": "high"},
            drop_params=True,
        )

        reasoning_content_exists = False
        for chunk in resp:
            print(f"chunk 2: {chunk}")
            if (
                hasattr(chunk.choices[0].delta, "reasoning_content")
                and chunk.choices[0].delta.reasoning_content is not None
            ):
                reasoning_content_exists = True
                break
        assert reasoning_content_exists
    except litellm.Timeout:
        pytest.skip("Model is timing out")




def test_streaming_with_cost_calculation():
    from typing import Optional

    from litellm.types.utils import Usage

    litellm.include_cost_in_streaming_usage = True

    ## Test 1: check if usage object can handle 'cost' field
    usage_object = Usage(
        prompt_tokens=100,
        completion_tokens=100,
        total_tokens=200,
        cost=1.0,
    )
    assert usage_object.cost is not None

    print(f"usage_object: {usage_object}")

    ## Test 2: check if usage object has 'cost' field when streaming

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    usage_object: Optional[Usage] = None
    for chunk in response:
        _usage_obj = getattr(chunk, "usage", None)
        if _usage_obj is not None:
            usage_object = _usage_obj
            break

    assert usage_object is not None
    assert usage_object.total_tokens is not None
    assert usage_object.total_tokens > 0
    assert usage_object.prompt_tokens is not None
    assert usage_object.prompt_tokens > 0
    assert usage_object.cost is not None
    assert usage_object.cost > 0


def test_streaming_finish_reason():
    litellm.set_verbose = False

    openai_finish_reason_idx: Optional[int] = None
    openai_last_chunk_idx: Optional[int] = None
    anthropic_finish_reason_idx: Optional[int] = None
    anthropic_last_chunk_idx: Optional[int] = None

    ## OpenAI
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    for idx, chunk in enumerate(response):
        print(f"OPENAI CHUNK: {chunk}")
        if chunk.choices[0].finish_reason is not None:
            openai_finish_reason_idx = idx
        openai_last_chunk_idx = idx

    assert openai_finish_reason_idx is not None
    assert openai_finish_reason_idx > 0

    ## Anthropic
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    for idx, chunk in enumerate(response):
        print(f"ANTHROPIC CHUNK: {chunk}")
        if chunk.choices[0].finish_reason is not None:
            anthropic_finish_reason_idx = idx
        anthropic_last_chunk_idx = idx

    assert anthropic_finish_reason_idx is not None
    assert anthropic_finish_reason_idx > 0

    relative_anthropic_idx = anthropic_finish_reason_idx - anthropic_last_chunk_idx
    relative_openai_idx = openai_finish_reason_idx - openai_last_chunk_idx
    assert relative_anthropic_idx == relative_openai_idx
