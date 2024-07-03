#### What this tests ####
#    This tests streaming for the completion endpoint

import asyncio
import os
import sys
import time
import traceback
import uuid
from typing import Tuple

import pytest
from pydantic import BaseModel

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from dotenv import load_dotenv

load_dotenv()
import litellm
from litellm import (
    AuthenticationError,
    BadRequestError,
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
    assert isinstance(chunk, ModelResponse), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."
    assert not hasattr(chunk, "usage"), "Chunk cannot contain usage"

    for choice in chunk["choices"]:
        assert isinstance(choice["index"], int), "'index' should be an integer."
        assert isinstance(choice["delta"]["role"], str), "'role' should be a string."
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
    assert isinstance(chunk, ModelResponse), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."
    assert not hasattr(chunk, "usage"), "Chunk cannot contain usage"

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
    assert isinstance(chunk, ModelResponse), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."
    assert not hasattr(chunk, "usage"), "Chunk cannot contain usage"

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

# def test_completion_cohere_stream():
# # this is a flaky test due to the cohere API endpoint being unstable
#     try:
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {
#                 "role": "user",
#                 "content": "how does a court case get to the Supreme Court?",
#             },
#         ]
#         response = completion(
#             model="command-nightly", messages=messages, stream=True, max_tokens=50,
#         )
#         complete_response = ""
#         # Add any assertions here to check the response
#         has_finish_reason = False
#         for idx, chunk in enumerate(response):
#             chunk, finished = streaming_format_tests(idx, chunk)
#             has_finish_reason = finished
#             if finished:
#                 break
#             complete_response += chunk
#         if has_finish_reason is False:
#             raise Exception("Finish reason not in final chunk")
#         if complete_response.strip() == "":
#             raise Exception("Empty response received")
#         print(f"completion_response: {complete_response}")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_cohere_stream()


def test_completion_azure_stream_special_char():
    litellm.set_verbose = True
    messages = [{"role": "user", "content": "hi. respond with the <xml> tag only"}]
    response = completion(model="azure/chatgpt-v-2", messages=messages, stream=True)
    response_str = ""
    for part in response:
        response_str += part.choices[0].delta.content or ""
    print(f"response_str: {response_str}")
    assert len(response_str) > 0


def test_completion_azure_stream_content_filter_no_delta():
    """
    Tests streaming from Azure when the chunks have no delta because they represent the filtered content
    """
    try:
        chunks = [
            {
                "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
                "choices": [
                    {
                        "delta": {"content": "", "role": "assistant"},
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                "created": 1716563849,
                "model": "gpt-4o-2024-05-13",
                "object": "chat.completion.chunk",
                "system_fingerprint": "fp_5f4bad809a",
            },
            {
                "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
                "choices": [
                    {"delta": {"content": "This"}, "finish_reason": None, "index": 0}
                ],
                "created": 1716563849,
                "model": "gpt-4o-2024-05-13",
                "object": "chat.completion.chunk",
                "system_fingerprint": "fp_5f4bad809a",
            },
            {
                "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
                "choices": [
                    {"delta": {"content": " is"}, "finish_reason": None, "index": 0}
                ],
                "created": 1716563849,
                "model": "gpt-4o-2024-05-13",
                "object": "chat.completion.chunk",
                "system_fingerprint": "fp_5f4bad809a",
            },
            {
                "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
                "choices": [
                    {"delta": {"content": " a"}, "finish_reason": None, "index": 0}
                ],
                "created": 1716563849,
                "model": "gpt-4o-2024-05-13",
                "object": "chat.completion.chunk",
                "system_fingerprint": "fp_5f4bad809a",
            },
            {
                "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
                "choices": [
                    {"delta": {"content": " dummy"}, "finish_reason": None, "index": 0}
                ],
                "created": 1716563849,
                "model": "gpt-4o-2024-05-13",
                "object": "chat.completion.chunk",
                "system_fingerprint": "fp_5f4bad809a",
            },
            {
                "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
                "choices": [
                    {
                        "delta": {"content": " response"},
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                "created": 1716563849,
                "model": "gpt-4o-2024-05-13",
                "object": "chat.completion.chunk",
                "system_fingerprint": "fp_5f4bad809a",
            },
            {
                "id": "",
                "choices": [
                    {
                        "finish_reason": None,
                        "index": 0,
                        "content_filter_offsets": {
                            "check_offset": 35159,
                            "start_offset": 35159,
                            "end_offset": 36150,
                        },
                        "content_filter_results": {
                            "hate": {"filtered": False, "severity": "safe"},
                            "self_harm": {"filtered": False, "severity": "safe"},
                            "sexual": {"filtered": False, "severity": "safe"},
                            "violence": {"filtered": False, "severity": "safe"},
                        },
                    }
                ],
                "created": 0,
                "model": "",
                "object": "",
            },
            {
                "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
                "choices": [
                    {"delta": {"content": "."}, "finish_reason": None, "index": 0}
                ],
                "created": 1716563849,
                "model": "gpt-4o-2024-05-13",
                "object": "chat.completion.chunk",
                "system_fingerprint": "fp_5f4bad809a",
            },
            {
                "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "created": 1716563849,
                "model": "gpt-4o-2024-05-13",
                "object": "chat.completion.chunk",
                "system_fingerprint": "fp_5f4bad809a",
            },
            {
                "id": "",
                "choices": [
                    {
                        "finish_reason": None,
                        "index": 0,
                        "content_filter_offsets": {
                            "check_offset": 36150,
                            "start_offset": 36060,
                            "end_offset": 37029,
                        },
                        "content_filter_results": {
                            "hate": {"filtered": False, "severity": "safe"},
                            "self_harm": {"filtered": False, "severity": "safe"},
                            "sexual": {"filtered": False, "severity": "safe"},
                            "violence": {"filtered": False, "severity": "safe"},
                        },
                    }
                ],
                "created": 0,
                "model": "",
                "object": "",
            },
        ]

        chunk_list = []
        for chunk in chunks:
            new_chunk = litellm.ModelResponse(stream=True, id=chunk["id"])
            if "choices" in chunk and isinstance(chunk["choices"], list):
                new_choices = []
                for choice in chunk["choices"]:
                    if isinstance(choice, litellm.utils.StreamingChoices):
                        _new_choice = choice
                    elif isinstance(choice, dict):
                        _new_choice = litellm.utils.StreamingChoices(**choice)
                    new_choices.append(_new_choice)
                new_chunk.choices = new_choices
            chunk_list.append(new_chunk)

        completion_stream = ModelResponseListIterator(model_responses=chunk_list)

        litellm.set_verbose = True

        response = litellm.CustomStreamWrapper(
            completion_stream=completion_stream,
            model="gpt-4-0613",
            custom_llm_provider="cached_response",
            logging_obj=litellm.Logging(
                model="gpt-4-0613",
                messages=[{"role": "user", "content": "Hey"}],
                stream=True,
                call_type="completion",
                start_time=time.time(),
                litellm_call_id="12345",
                function_id="1245",
            ),
        )

        for idx, chunk in enumerate(response):
            complete_response = ""
            for idx, chunk in enumerate(response):
                # print
                delta = chunk.choices[0].delta
                content = delta.content if delta else None
                complete_response += content or ""
                if chunk.choices[0].finish_reason is not None:
                    break
            assert len(complete_response) > 0

    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


def test_completion_cohere_stream_bad_key():
    try:
        litellm.cache = None
        api_key = "bad-key"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        response = completion(
            model="command-nightly",
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
            has_finish_reason = finished
            if finished:
                break
            complete_response += chunk
        if has_finish_reason is False:
            raise Exception("Finish reason not in final chunk")
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"completion_response: {complete_response}")
    except AuthenticationError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_cohere_stream_bad_key()


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
            model="azure/chatgpt-v-2", messages=messages, stream=True, max_tokens=50
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


# test_completion_azure_stream()
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_completion_predibase_streaming(sync_mode):
    try:
        litellm.set_verbose = True

        if sync_mode:
            response = completion(
                model="predibase/llama-3-8b-instruct",
                tenant_id="c4768f95",
                api_base="https://serving.app.predibase.com",
                api_key=os.getenv("PREDIBASE_API_KEY"),
                messages=[{"role": "user", "content": "What is the meaning of life?"}],
                stream=True,
            )

            complete_response = ""
            for idx, init_chunk in enumerate(response):
                chunk, finished = streaming_format_tests(idx, init_chunk)
                complete_response += chunk
                custom_llm_provider = init_chunk._hidden_params["custom_llm_provider"]
                print(f"custom_llm_provider: {custom_llm_provider}")
                assert custom_llm_provider == "predibase"
                if finished:
                    assert isinstance(
                        init_chunk.choices[0], litellm.utils.StreamingChoices
                    )
                    break
            if complete_response.strip() == "":
                raise Exception("Empty response received")
        else:
            response = await litellm.acompletion(
                model="predibase/llama-3-8b-instruct",
                tenant_id="c4768f95",
                api_base="https://serving.app.predibase.com",
                api_key=os.getenv("PREDIBASE_API_KEY"),
                messages=[{"role": "user", "content": "What is the meaning of life?"}],
                stream=True,
            )

            # await response

            complete_response = ""
            idx = 0
            async for init_chunk in response:
                chunk, finished = streaming_format_tests(idx, init_chunk)
                complete_response += chunk
                custom_llm_provider = init_chunk._hidden_params["custom_llm_provider"]
                print(f"custom_llm_provider: {custom_llm_provider}")
                assert custom_llm_provider == "predibase"
                idx += 1
                if finished:
                    assert isinstance(
                        init_chunk.choices[0], litellm.utils.StreamingChoices
                    )
                    break
            if complete_response.strip() == "":
                raise Exception("Empty response received")

        print(f"complete_response: {complete_response}")
    except litellm.Timeout as e:
        pass
    except litellm.InternalServerError as e:
        pass
    except Exception as e:
        print("ERROR class", e.__class__)
        print("ERROR message", e)
        print("ERROR traceback", traceback.format_exc())

        pytest.fail(f"Error occurred: {e}")


def test_completion_azure_function_calling_stream():
    try:
        litellm.set_verbose = False
        user_message = "What is the current weather in Boston?"
        messages = [{"content": user_message, "role": "user"}]
        response = completion(
            model="azure/chatgpt-functioncalling",
            messages=messages,
            stream=True,
            tools=tools_schema,
        )
        # Add any assertions here to check the response
        for chunk in response:
            print(chunk)
            if chunk["choices"][0]["finish_reason"] == "stop":
                break
            print(chunk["choices"][0]["finish_reason"])
            print(chunk["choices"][0]["delta"]["content"])
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure_function_calling_stream()


@pytest.mark.skip("Flaky ollama test - needs to be fixed")
def test_completion_ollama_hosted_stream():
    try:
        litellm.set_verbose = True
        response = completion(
            model="ollama/phi",
            messages=messages,
            max_tokens=10,
            num_retries=3,
            timeout=20,
            api_base="https://test-ollama-endpoint.onrender.com",
            stream=True,
        )
        # Add any assertions here to check the response
        complete_response = ""
        # Add any assertions here to check the response
        for idx, init_chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, init_chunk)
            complete_response += chunk
            if finished:
                assert isinstance(init_chunk.choices[0], litellm.utils.StreamingChoices)
                break
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"complete_response: {complete_response}")
    except Exception as e:
        if "try pulling it first" in str(e):
            return
        pytest.fail(f"Error occurred: {e}")


# test_completion_ollama_hosted_stream()


def test_completion_claude_stream():
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        response = completion(
            model="claude-instant-1.2", messages=messages, stream=True, max_tokens=50
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


# test_completion_claude_stream()
def test_completion_claude_2_stream():
    litellm.set_verbose = True
    response = completion(
        model="claude-2",
        messages=[{"role": "user", "content": "hello from litellm"}],
        stream=True,
    )
    complete_response = ""
    # Add any assertions here to check the response
    idx = 0
    for chunk in response:
        print(chunk)
        # print(chunk.choices[0].delta)
        chunk, finished = streaming_format_tests(idx, chunk)
        if finished:
            break
        complete_response += chunk
        idx += 1
    if complete_response.strip() == "":
        raise Exception("Empty response received")
    print(f"completion_response: {complete_response}")


@pytest.mark.asyncio
async def test_acompletion_claude_2_stream():
    litellm.set_verbose = True
    response = await litellm.acompletion(
        model="claude-2",
        messages=[{"role": "user", "content": "hello from litellm"}],
        stream=True,
    )
    complete_response = ""
    # Add any assertions here to check the response
    idx = 0
    async for chunk in response:
        print(chunk)
        # print(chunk.choices[0].delta)
        chunk, finished = streaming_format_tests(idx, chunk)
        if finished:
            break
        complete_response += chunk
        idx += 1
    if complete_response.strip() == "":
        raise Exception("Empty response received")
    print(f"completion_response: {complete_response}")


def test_completion_palm_stream():
    try:
        litellm.set_verbose = False
        print("Streaming palm response")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        print("testing palm streaming")
        response = completion(model="palm/chat-bison", messages=messages, stream=True)

        complete_response = ""
        # Add any assertions here to check the response
        for idx, chunk in enumerate(response):
            print(chunk)
            # print(chunk.choices[0].delta)
            chunk, finished = streaming_format_tests(idx, chunk)
            if finished:
                break
            complete_response += chunk
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"completion_response: {complete_response}")
    except litellm.Timeout as e:
        pass
    except litellm.APIError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_palm_stream()


@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)  # ,
@pytest.mark.asyncio
async def test_completion_gemini_stream(sync_mode):
    try:
        litellm.set_verbose = True
        print("Streaming gemini response")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Who was Alexander?",
            },
        ]
        print("testing gemini streaming")
        complete_response = ""
        # Add any assertions here to check the response
        non_empty_chunks = 0

        if sync_mode:
            response = completion(
                model="gemini/gemini-1.5-flash",
                messages=messages,
                stream=True,
            )

            for idx, chunk in enumerate(response):
                print(chunk)
                # print(chunk.choices[0].delta)
                chunk, finished = streaming_format_tests(idx, chunk)
                if finished:
                    break
                non_empty_chunks += 1
                complete_response += chunk
        else:
            response = await litellm.acompletion(
                model="gemini/gemini-1.5-flash",
                messages=messages,
                stream=True,
            )

            idx = 0
            async for chunk in response:
                print(chunk)
                # print(chunk.choices[0].delta)
                chunk, finished = streaming_format_tests(idx, chunk)
                if finished:
                    break
                non_empty_chunks += 1
                complete_response += chunk
                idx += 1

        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"completion_response: {complete_response}")
        assert non_empty_chunks > 1
    except litellm.InternalServerError as e:
        pass
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        # if "429 Resource has been exhausted":
        #     return
        pytest.fail(f"Error occurred: {e}")


# asyncio.run(test_acompletion_gemini_stream())


def test_completion_mistral_api_stream():
    try:
        litellm.set_verbose = True
        print("Testing streaming mistral api response")
        response = completion(
            model="mistral/mistral-medium",
            messages=[
                {
                    "role": "user",
                    "content": "Hey, how's it going?",
                }
            ],
            max_tokens=10,
            stream=True,
        )
        complete_response = ""
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
    except litellm.APIError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


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
            model="mistral/mistral-large-latest",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )
        idx = 0
        for chunk in response:
            print(f"chunk in response: {chunk}")
            assert chunk._hidden_params["custom_llm_provider"] == "mistral"
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


# test_completion_mistral_api_stream()


def test_completion_deep_infra_stream():
    # deep infra,currently includes role in the 2nd chunk
    # waiting for them to make a fix on this
    litellm.set_verbose = True
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        print("testing deep infra streaming")
        response = completion(
            model="deepinfra/meta-llama/Llama-2-70b-chat-hf",
            messages=messages,
            stream=True,
            max_tokens=80,
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
    except Exception as e:
        if "Model busy, retry later" in str(e):
            pass
        pytest.fail(f"Error occurred: {e}")


# test_completion_deep_infra_stream()


@pytest.mark.skip()
def test_completion_nlp_cloud_stream():
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        print("testing nlp cloud streaming")
        response = completion(
            model="nlp_cloud/finetuned-llama-2-70b",
            messages=messages,
            stream=True,
            max_tokens=20,
        )

        complete_response = ""
        # Add any assertions here to check the response
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            complete_response += chunk
            if finished:
                break
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"completion_response: {complete_response}")
    except Exception as e:
        print(f"Error occurred: {e}")
        pytest.fail(f"Error occurred: {e}")


# test_completion_nlp_cloud_stream()


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
            model="claude-instant-1",
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


# test_completion_claude_stream_bad_key()
# test_completion_replicate_stream()


@pytest.mark.parametrize("provider", ["vertex_ai_beta"])  # ""
def test_vertex_ai_stream(provider):
    from litellm.tests.test_amazing_vertex_completion import load_vertex_ai_credentials

    load_vertex_ai_credentials()
    litellm.set_verbose = True
    litellm.vertex_project = "adroit-crow-413218"
    import random

    test_models = ["gemini-1.5-pro"]
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

            assert False
        except litellm.RateLimitError as e:
            pass
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


# def test_completion_vertexai_stream():
#     try:
#         import os
#         os.environ["VERTEXAI_PROJECT"] = "pathrise-convert-1606954137718"
#         os.environ["VERTEXAI_LOCATION"] = "us-central1"
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {
#                 "role": "user",
#                 "content": "how does a court case get to the Supreme Court?",
#             },
#         ]
#         response = completion(
#             model="vertex_ai/chat-bison", messages=messages, stream=True, max_tokens=50
#         )
#         complete_response = ""
#         has_finish_reason = False
#         # Add any assertions here to check the response
#         for idx, chunk in enumerate(response):
#             chunk, finished = streaming_format_tests(idx, chunk)
#             has_finish_reason = finished
#             if finished:
#                 break
#             complete_response += chunk
#         if has_finish_reason is False:
#             raise Exception("finish reason not set for last chunk")
#         if complete_response.strip() == "":
#             raise Exception("Empty response received")
#         print(f"completion_response: {complete_response}")
#     except InvalidRequestError as e:
#         pass
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_vertexai_stream()


# def test_completion_vertexai_stream_bad_key():
#     try:
#         import os
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {
#                 "role": "user",
#                 "content": "how does a court case get to the Supreme Court?",
#             },
#         ]
#         response = completion(
#             model="vertex_ai/chat-bison", messages=messages, stream=True, max_tokens=50
#         )
#         complete_response = ""
#         has_finish_reason = False
#         # Add any assertions here to check the response
#         for idx, chunk in enumerate(response):
#             chunk, finished = streaming_format_tests(idx, chunk)
#             has_finish_reason = finished
#             if finished:
#                 break
#             complete_response += chunk
#         if has_finish_reason is False:
#             raise Exception("finish reason not set for last chunk")
#         if complete_response.strip() == "":
#             raise Exception("Empty response received")
#         print(f"completion_response: {complete_response}")
#     except InvalidRequestError as e:
#         pass
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_vertexai_stream_bad_key()


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_completion_databricks_streaming(sync_mode):
    litellm.set_verbose = True
    model_name = "databricks/databricks-dbrx-instruct"
    try:
        if sync_mode:
            final_chunk: Optional[litellm.ModelResponse] = None
            response: litellm.CustomStreamWrapper = completion(  # type: ignore
                model=model_name,
                messages=messages,
                max_tokens=10,  # type: ignore
                stream=True,
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
            if has_finish_reason == False:
                raise Exception("finish reason not set")
            if complete_response.strip() == "":
                raise Exception("Empty response received")
        else:
            response: litellm.CustomStreamWrapper = await litellm.acompletion(  # type: ignore
                model=model_name,
                messages=messages,
                max_tokens=100,  # type: ignore
                stream=True,
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
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [False, True])
@pytest.mark.asyncio
async def test_completion_replicate_llama3_streaming(sync_mode):
    litellm.set_verbose = True
    model_name = "replicate/meta/meta-llama-3-8b-instruct"
    try:
        if sync_mode:
            final_chunk: Optional[litellm.ModelResponse] = None
            response: litellm.CustomStreamWrapper = completion(  # type: ignore
                model=model_name,
                messages=messages,
                max_tokens=10,  # type: ignore
                stream=True,
                num_retries=3,
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
            if has_finish_reason == False:
                raise Exception("finish reason not set")
            if complete_response.strip() == "":
                raise Exception("Empty response received")
        else:
            response: litellm.CustomStreamWrapper = await litellm.acompletion(  # type: ignore
                model=model_name,
                messages=messages,
                max_tokens=100,  # type: ignore
                stream=True,
                num_retries=3,
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
    except litellm.UnprocessableEntityError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# TEMP Commented out - replicate throwing an auth error
#     try:
#         litellm.set_verbose = True
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {
#                 "role": "user",
#                 "content": "how does a court case get to the Supreme Court?",
#             },
#         ]
#         response = completion(
#             model="replicate/meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3", messages=messages, stream=True, max_tokens=50
#         )
#         complete_response = ""
#         has_finish_reason = False
#         # Add any assertions here to check the response
#         for idx, chunk in enumerate(response):
#             chunk, finished = streaming_format_tests(idx, chunk)
#             has_finish_reason = finished
#             if finished:
#                 break
#             complete_response += chunk
#         if has_finish_reason is False:
#             raise Exception("finish reason not set for last chunk")
#         if complete_response.strip() == "":
#             raise Exception("Empty response received")
#         print(f"completion_response: {complete_response}")
#     except InvalidRequestError as e:
#         pass
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [True])  # False
@pytest.mark.parametrize(
    "model",
    [
        "bedrock/cohere.command-r-plus-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-instant-v1",
        "bedrock/ai21.j2-mid",
        "mistral.mistral-7b-instruct-v0:2",
        "bedrock/amazon.titan-tg1-large",
        "meta.llama3-8b-instruct-v1:0",
        "cohere.command-text-v14",
    ],
)
@pytest.mark.asyncio
async def test_bedrock_httpx_streaming(sync_mode, model):
    try:
        litellm.set_verbose = True
        if sync_mode:
            final_chunk: Optional[litellm.ModelResponse] = None
            response: litellm.CustomStreamWrapper = completion(  # type: ignore
                model=model,
                messages=messages,
                max_tokens=10,  # type: ignore
                stream=True,
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
            if has_finish_reason == False:
                raise Exception("finish reason not set")
            if complete_response.strip() == "":
                raise Exception("Empty response received")
        else:
            response: litellm.CustomStreamWrapper = await litellm.acompletion(  # type: ignore
                model=model,
                messages=messages,
                max_tokens=100,  # type: ignore
                stream=True,
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
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_bedrock_claude_3_streaming():
    try:
        litellm.set_verbose = True
        response: ModelResponse = completion(  # type: ignore
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
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
        "claude-3-opus-20240229",
        "cohere.command-r-plus-v1:0",  # bedrock
        "gpt-3.5-turbo",
        "databricks/databricks-dbrx-instruct",  # databricks
        "predibase/llama-3-8b-instruct",  # predibase
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
            )
            complete_response = ""
            # Add any assertions here to-check the response
            num_finish_reason = 0
            for chunk in response:
                print(f"chunk: {chunk}")
                if isinstance(chunk, ModelResponse):
                    if chunk.choices[0].finish_reason is not None:
                        num_finish_reason += 1
            assert num_finish_reason == 1

        async def test_streaming():
            response: litellm.CustomStreamWrapper = await litellm.acompletion(  # type: ignore
                model=model,
                messages=messages,
                stream=True,
                max_tokens=10,
            )
            complete_response = ""
            # Add any assertions here to-check the response
            num_finish_reason = 0
            async for chunk in response:
                print(f"type of chunk: {type(chunk)}")
                if isinstance(chunk, ModelResponse):
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
    except litellm.InternalServerError as e:
        if "predibase" in str(e).lower():
            # only skip internal server error from predibase - their endpoint seems quite unstable
            pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="Replicate changed exceptions")
def test_completion_replicate_stream_bad_key():
    try:
        api_key = "bad-key"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        response = completion(
            model="replicate/meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            messages=messages,
            stream=True,
            max_tokens=50,
            api_key=api_key,
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
    except AuthenticationError as e:
        # this is an auth error with a bad key
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_replicate_stream_bad_key()


def test_completion_bedrock_claude_stream():
    try:
        litellm.set_verbose = False
        response = completion(
            model="bedrock/anthropic.claude-instant-v1",
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
        first_chunk_id = None
        for idx, chunk in enumerate(response):
            # print
            if idx == 0:
                first_chunk_id = chunk.id
            else:
                assert (
                    chunk.id == first_chunk_id
                ), f"chunk ids do not match: {chunk.id} != first chunk id{first_chunk_id}"
            chunk, finished = streaming_format_tests(idx, chunk)
            has_finish_reason = finished
            complete_response += chunk
            if finished:
                break
        if has_finish_reason is False:
            raise Exception("finish reason not set for last chunk")
        if complete_response.strip() == "":
            raise Exception("Empty response received")
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_bedrock_claude_stream()


def test_completion_bedrock_ai21_stream():
    try:
        litellm.set_verbose = False
        response = completion(
            model="bedrock/ai21.j2-mid-v1",
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


# test_completion_bedrock_ai21_stream()


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


def test_sagemaker_weird_response():
    """
    When the stream ends, flush any remaining holding chunks.
    """
    try:
        import json

        from litellm.llms.sagemaker import TokenIterator

        chunk = """<s>[INST] Hey, how's it going? [/INST],
        I'm doing well, thanks for asking! How about you? Is there anything you'd like to chat about or ask? I'm here to help with any questions you might have."""

        data = "\n".join(
            map(
                lambda x: f"data: {json.dumps({'token': {'text': x.strip()}})}",
                chunk.strip().split(","),
            )
        )
        stream = bytes(data, encoding="utf8")

        # Modify the array to be a dictionary with "PayloadPart" and "Bytes" keys.
        stream_iterator = iter([{"PayloadPart": {"Bytes": stream}}])

        token_iter = TokenIterator(stream_iterator)

        # for token in token_iter:
        #     print(token)
        litellm.set_verbose = True

        logging_obj = litellm.Logging(
            model="berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=messages,
            stream=True,
            litellm_call_id="1234",
            function_id="function_id",
            call_type="acompletion",
            start_time=time.time(),
        )
        response = litellm.CustomStreamWrapper(
            completion_stream=token_iter,
            model="berri-benchmarking-Llama-2-70b-chat-hf-4",
            custom_llm_provider="sagemaker",
            logging_obj=logging_obj,
        )
        complete_response = ""
        for idx, chunk in enumerate(response):
            # print
            chunk, finished = streaming_format_tests(idx, chunk)
            has_finish_reason = finished
            complete_response += chunk
            if finished:
                break
        assert len(complete_response) > 0
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# test_sagemaker_weird_response()


@pytest.mark.asyncio
async def test_sagemaker_streaming_async():
    try:
        messages = [{"role": "user", "content": "Hey, how's it going?"}]
        litellm.set_verbose = True
        response = await litellm.acompletion(
            model="sagemaker/jumpstart-dft-hf-llm-mistral-7b-ins-20240329-150233",
            model_id="huggingface-llm-mistral-7b-instruct-20240329-150233",
            messages=messages,
            temperature=0.2,
            max_tokens=80,
            aws_region_name=os.getenv("AWS_REGION_NAME_2"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_2"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_2"),
            stream=True,
        )
        # Add any assertions here to check the response
        print(response)
        complete_response = ""
        has_finish_reason = False
        # Add any assertions here to check the response
        idx = 0
        async for chunk in response:
            # print
            chunk, finished = streaming_format_tests(idx, chunk)
            has_finish_reason = finished
            complete_response += chunk
            if finished:
                break
            idx += 1
        if has_finish_reason is False:
            raise Exception("finish reason not set for last chunk")
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"completion_response: {complete_response}")
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# asyncio.run(test_sagemaker_streaming_async())


def test_completion_sagemaker_stream():
    try:
        response = completion(
            model="sagemaker/jumpstart-dft-hf-llm-mistral-7b-ins-20240329-150233",
            model_id="huggingface-llm-mistral-7b-instruct-20240329-150233",
            messages=messages,
            temperature=0.2,
            max_tokens=80,
            aws_region_name=os.getenv("AWS_REGION_NAME_2"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_2"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_2"),
            stream=True,
        )
        complete_response = ""
        has_finish_reason = False
        # Add any assertions here to check the response
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            has_finish_reason = finished
            if finished:
                break
            complete_response += chunk
        if has_finish_reason is False:
            raise Exception("finish reason not set for last chunk")
        if complete_response.strip() == "":
            raise Exception("Empty response received")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="Account deleted by IBM.")
def test_completion_watsonx_stream():
    litellm.set_verbose = True
    try:
        response = completion(
            model="watsonx/ibm/granite-13b-chat-v2",
            messages=messages,
            temperature=0.5,
            max_tokens=20,
            stream=True,
        )
        complete_response = ""
        has_finish_reason = False
        # Add any assertions here to check the response
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            has_finish_reason = finished
            if finished:
                break
            complete_response += chunk
        if has_finish_reason is False:
            raise Exception("finish reason not set for last chunk")
        if complete_response.strip() == "":
            raise Exception("Empty response received")
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_sagemaker_stream()


# def test_maritalk_streaming():
#     messages = [{"role": "user", "content": "Hey"}]
#     try:
#         response = completion("maritalk", messages=messages, stream=True)
#         complete_response = ""
#         start_time = time.time()
#         for idx, chunk in enumerate(response):
#             chunk, finished = streaming_format_tests(idx, chunk)
#             complete_response += chunk
#             if finished:
#                 break
#         if complete_response.strip() == "":
#             raise Exception("Empty response received")
#     except:
#         pytest.fail(f"error occurred: {traceback.format_exc()}")
# test_maritalk_streaming()
# test on openai completion call


# # test on ai21 completion call
def ai21_completion_call():
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an all-knowing oracle",
            },
            {"role": "user", "content": "What is the meaning of the Universe?"},
        ]
        response = completion(
            model="j2-ultra", messages=messages, stream=True, max_tokens=500
        )
        print(f"response: {response}")
        has_finished = False
        complete_response = ""
        start_time = time.time()
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            has_finished = finished
            complete_response += chunk
            if finished:
                break
        if has_finished is False:
            raise Exception("finished reason missing from final chunk")
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"completion_response: {complete_response}")
    except:
        pytest.fail(f"error occurred: {traceback.format_exc()}")


# ai21_completion_call()


def ai21_completion_call_bad_key():
    try:
        api_key = "bad-key"
        response = completion(
            model="j2-ultra", messages=messages, stream=True, api_key=api_key
        )
        print(f"response: {response}")
        complete_response = ""
        start_time = time.time()
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            if finished:
                break
            complete_response += chunk
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"completion_response: {complete_response}")
    except:
        pytest.fail(f"error occurred: {traceback.format_exc()}")


# ai21_completion_call_bad_key()


@pytest.mark.skip(reason="flaky test")
@pytest.mark.asyncio
async def test_hf_completion_tgi_stream():
    try:
        response = await acompletion(
            model="huggingface/HuggingFaceH4/zephyr-7b-beta",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            stream=True,
        )
        # Add any assertions here to check the response
        print(f"response: {response}")
        complete_response = ""
        start_time = time.time()
        idx = 0
        async for chunk in response:
            chunk, finished = streaming_format_tests(idx, chunk)
            complete_response += chunk
            if finished:
                break
            idx += 1
        print(f"completion_response: {complete_response}")
    except litellm.ServiceUnavailableError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# hf_test_completion_tgi_stream()

# def test_completion_aleph_alpha():
#     try:
#         response = completion(
#             model="luminous-base", messages=messages, stream=True
#         )
#         # Add any assertions here to check the response
#         has_finished = False
#         complete_response = ""
#         start_time = time.time()
#         for idx, chunk in enumerate(response):
#             chunk, finished = streaming_format_tests(idx, chunk)
#             has_finished = finished
#             complete_response += chunk
#             if finished:
#                 break
#         if has_finished is False:
#             raise Exception("finished reason missing from final chunk")
#         if complete_response.strip() == "":
#             raise Exception("Empty response received")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# # test_completion_aleph_alpha()

# def test_completion_aleph_alpha_bad_key():
#     try:
#         api_key = "bad-key"
#         response = completion(
#             model="luminous-base", messages=messages, stream=True, api_key=api_key
#         )
#         # Add any assertions here to check the response
#         has_finished = False
#         complete_response = ""
#         start_time = time.time()
#         for idx, chunk in enumerate(response):
#             chunk, finished = streaming_format_tests(idx, chunk)
#             has_finished = finished
#             complete_response += chunk
#             if finished:
#                 break
#         if has_finished is False:
#             raise Exception("finished reason missing from final chunk")
#         if complete_response.strip() == "":
#             raise Exception("Empty response received")
#     except InvalidRequestError as e:
#         pass
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_aleph_alpha_bad_key()


# test on openai completion call
def test_openai_chat_completion_call():
    try:
        litellm.set_verbose = False
        print(f"making openai chat completion call")
        response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
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
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass


# test_openai_chat_completion_call()


def test_openai_chat_completion_complete_response_call():
    try:
        complete_response = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
            complete_response=True,
        )
        print(f"complete response: {complete_response}")
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass


# test_openai_chat_completion_complete_response_call()
@pytest.mark.parametrize(
    "model",
    ["gpt-3.5-turbo", "azure/chatgpt-v-2"],
)
@pytest.mark.parametrize(
    "sync",
    [True, False],
)
@pytest.mark.asyncio
async def test_openai_stream_options_call(model, sync):
    litellm.set_verbose = False
    usage = None
    chunks = []
    if sync:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "say GM - we're going to make it "}
            ],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=10,
        )
        for chunk in response:
            print("chunk: ", chunk)
            chunks.append(chunk)
    else:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": "say GM - we're going to make it "}
            ],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=10,
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


def test_openai_stream_options_call_text_completion():
    litellm.set_verbose = False
    for idx in range(3):
        try:
            response = litellm.text_completion(
                model="gpt-3.5-turbo-instruct",
                prompt="say GM - we're going to make it ",
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=10,
            )
            usage = None
            chunks = []
            for chunk in response:
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
            assert last_chunk.usage.total_tokens > 0
            assert last_chunk.usage.prompt_tokens > 0
            assert last_chunk.usage.completion_tokens > 0

            # assert all non last chunks have usage=None
            assert all(chunk.usage is None for chunk in chunks[:-1])
            break
        except Exception as e:
            if idx < 2:
                pass
            else:
                raise e


def test_openai_text_completion_call():
    try:
        litellm.set_verbose = True
        response = completion(
            model="gpt-3.5-turbo-instruct", messages=messages, stream=True
        )
        complete_response = ""
        start_time = time.time()
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            print(f"chunk: {chunk}")
            complete_response += chunk
            if finished:
                break
            # print(f'complete_chunk: {complete_response}')
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"complete response: {complete_response}")
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass


# test_openai_text_completion_call()


# # test on together ai completion call - starcoder
def test_together_ai_completion_call_mistral():
    try:
        litellm.set_verbose = False
        start_time = time.time()
        response = completion(
            model="together_ai/mistralai/Mistral-7B-Instruct-v0.2",
            messages=messages,
            logger_fn=logger_fn,
            stream=True,
        )
        complete_response = ""
        print(f"returned response object: {response}")
        has_finish_reason = False
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            has_finish_reason = finished
            if finished:
                break
            complete_response += chunk
        if has_finish_reason is False:
            raise Exception("Finish reason not set for last chunk")
        if complete_response == "":
            raise Exception("Empty response received")
        print(f"complete response: {complete_response}")
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass


def test_together_ai_completion_call_starcoder_bad_key():
    try:
        api_key = "bad-key"
        start_time = time.time()
        response = completion(
            model="together_ai/bigcode/starcoder",
            messages=messages,
            stream=True,
            api_key=api_key,
        )
        complete_response = ""
        has_finish_reason = False
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            has_finish_reason = finished
            if finished:
                break
            complete_response += chunk
        if has_finish_reason is False:
            raise Exception("Finish reason not set for last chunk")
        if complete_response == "":
            raise Exception("Empty response received")
        print(f"complete response: {complete_response}")
    except BadRequestError as e:
        pass
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass


# test_together_ai_completion_call_starcoder_bad_key()
#### Test Function calling + streaming ####


def test_completion_openai_with_functions():
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
    try:
        litellm.set_verbose = False
        response = completion(
            model="gpt-3.5-turbo-1106",
            messages=[{"role": "user", "content": "what's the weather in SF"}],
            functions=function1,
            stream=True,
        )
        # Add any assertions here to check the response
        print(response)
        for chunk in response:
            print(chunk)
            if chunk["choices"][0]["finish_reason"] == "stop":
                break
            print(chunk["choices"][0]["finish_reason"])
            print(chunk["choices"][0]["delta"]["content"])
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_openai_with_functions()
#### Test Async streaming ####


# # test on ai21 completion call
async def ai21_async_completion_call():
    try:
        response = completion(
            model="j2-ultra", messages=messages, stream=True, logger_fn=logger_fn
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
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass


# asyncio.run(ai21_async_completion_call())


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
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass


# asyncio.run(completion_call())

#### Test Function Calling + Streaming ####

final_openai_function_call_example = {
    "id": "chatcmpl-7zVNA4sXUftpIg6W8WlntCyeBj2JY",
    "object": "chat.completion",
    "created": 1694892960,
    "model": "gpt-3.5-turbo-0613",
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
    "model": "gpt-3.5-turbo-0613",
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
    "model": "gpt-3.5-turbo-0613",
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
    "model": "gpt-3.5-turbo-0613",
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
        except:  # check if it's the last chunk (returns an empty delta {} )
            decision = validate_final_function_call_chunk_structure(data=chunk)
            finished = True
    if "content" in chunk["choices"][0]["delta"]:
        extracted_chunk = chunk["choices"][0]["delta"]["content"]
    if decision == False:
        raise Exception("incorrect format")
    return extracted_chunk, finished


@pytest.mark.parametrize(
    "model", ["gpt-3.5-turbo", "anthropic.claude-3-sonnet-20240229-v1:0"]
)
def test_streaming_and_function_calling(model):
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
        litellm.set_verbose = True
        response: litellm.CustomStreamWrapper = completion(
            model=model,
            tools=tools,
            messages=messages,
            stream=True,
            tool_choice="required",
        )  # type: ignore
        # Add any assertions here to check the response
        for idx, chunk in enumerate(response):
            # continue
            print("\n{}\n".format(chunk))
            if idx == 0:
                assert (
                    chunk.choices[0].delta.tool_calls[0].function.arguments is not None
                )
                assert isinstance(
                    chunk.choices[0].delta.tool_calls[0].function.arguments, str
                )
        # assert False
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        raise e


# test_azure_streaming_and_function_calling()


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


# test_success_callback_streaming()

from typing import List, Optional

#### STREAMING + FUNCTION CALLING ###
from pydantic import BaseModel


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
    system_fingerprint: str
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
            model="azure/gpt-4-nov-release",
            tools=tools,
            tool_choice="auto",
            messages=messages,
            stream=True,
            api_base=os.getenv("AZURE_FRANCE_API_BASE"),
            api_key=os.getenv("AZURE_FRANCE_API_KEY"),
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
    import uuid

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
    from litellm.caching import Cache

    litellm.cache = Cache(
        type="redis",
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        password=os.environ["REDIS_PASSWORD"],
    )
    try:
        litellm.set_verbose = True
        response = await litellm.acompletion(
            model="azure/gpt-4-nov-release",
            tools=tools,
            tool_choice="auto",
            messages=messages,
            stream=True,
            api_base=os.getenv("AZURE_FRANCE_API_BASE"),
            api_key=os.getenv("AZURE_FRANCE_API_KEY"),
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
            model="azure/gpt-4-nov-release",
            tools=tools,
            tool_choice="auto",
            messages=messages,
            stream=True,
            api_base=os.getenv("AZURE_FRANCE_API_BASE"),
            api_key=os.getenv("AZURE_FRANCE_API_KEY"),
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
            model="claude-3-opus-20240229",
            messages=messages,
            tools=tools,
            tool_choice="auto",
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
                validate_final_streaming_function_calling_chunk(chunk=chunk)
            idx += 1
        # raise Exception("it worked!")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_acompletion_claude_3_function_call_with_streaming():
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
        response = await acompletion(
            model="claude-3-opus-20240229",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
        )
        idx = 0
        print(f"response: {response}")
        async for chunk in response:
            # print(f"chunk: {chunk}")
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
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


class ModelResponseIterator:
    def __init__(self, model_response):
        self.model_response = model_response
        self.is_done = False

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        if self.is_done:
            raise StopIteration
        self.is_done = True
        return self.model_response

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.is_done:
            raise StopAsyncIteration
        self.is_done = True
        return self.model_response


class ModelResponseListIterator:
    def __init__(self, model_responses):
        self.model_responses = model_responses
        self.index = 0

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.model_responses):
            raise StopIteration
        model_response = self.model_responses[self.index]
        self.index += 1
        return model_response

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.model_responses):
            raise StopAsyncIteration
        model_response = self.model_responses[self.index]
        self.index += 1
        return model_response


def test_unit_test_custom_stream_wrapper():
    """
    Test if last streaming chunk ends with '?', if the message repeats itself.
    """
    litellm.set_verbose = False
    chunk = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1694268190,
        "model": "gpt-3.5-turbo-0125",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {"index": 0, "delta": {"content": "How are you?"}, "finish_reason": "stop"}
        ],
    }
    chunk = litellm.ModelResponse(**chunk, stream=True)

    completion_stream = ModelResponseIterator(model_response=chunk)

    response = litellm.CustomStreamWrapper(
        completion_stream=completion_stream,
        model="gpt-3.5-turbo",
        custom_llm_provider="cached_response",
        logging_obj=litellm.Logging(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey"}],
            stream=True,
            call_type="completion",
            start_time=time.time(),
            litellm_call_id="12345",
            function_id="1245",
        ),
    )

    freq = 0
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            if "How are you?" in chunk.choices[0].delta.content:
                freq += 1
    assert freq == 1


def test_aamazing_unit_test_custom_stream_wrapper_n():
    """
    Test if the translated output maps exactly to the received openai input

    Relevant issue: https://github.com/BerriAI/litellm/issues/3276
    """
    chunks = [
        {
            "id": "chatcmpl-9HzZIMCtVq7CbTmdwEZrktiTeoiYe",
            "object": "chat.completion.chunk",
            "created": 1714075272,
            "model": "gpt-4-0613",
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "It"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "It",
                                "logprob": -1.5952516,
                                "bytes": [73, 116],
                                "top_logprobs": [
                                    {
                                        "token": "Brown",
                                        "logprob": -0.7358765,
                                        "bytes": [66, 114, 111, 119, 110],
                                    }
                                ],
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-9HzZIMCtVq7CbTmdwEZrktiTeoiYe",
            "object": "chat.completion.chunk",
            "created": 1714075272,
            "model": "gpt-4-0613",
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 1,
                    "delta": {"content": "Brown"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "Brown",
                                "logprob": -0.7358765,
                                "bytes": [66, 114, 111, 119, 110],
                                "top_logprobs": [
                                    {
                                        "token": "Brown",
                                        "logprob": -0.7358765,
                                        "bytes": [66, 114, 111, 119, 110],
                                    }
                                ],
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-9HzZIMCtVq7CbTmdwEZrktiTeoiYe",
            "object": "chat.completion.chunk",
            "created": 1714075272,
            "model": "gpt-4-0613",
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "'s"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "'s",
                                "logprob": -0.006786893,
                                "bytes": [39, 115],
                                "top_logprobs": [
                                    {
                                        "token": "'s",
                                        "logprob": -0.006786893,
                                        "bytes": [39, 115],
                                    }
                                ],
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-9HzZIMCtVq7CbTmdwEZrktiTeoiYe",
            "object": "chat.completion.chunk",
            "created": 1714075272,
            "model": "gpt-4-0613",
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " impossible"},
                    "logprobs": {
                        "content": [
                            {
                                "token": " impossible",
                                "logprob": -0.06528423,
                                "bytes": [
                                    32,
                                    105,
                                    109,
                                    112,
                                    111,
                                    115,
                                    115,
                                    105,
                                    98,
                                    108,
                                    101,
                                ],
                                "top_logprobs": [
                                    {
                                        "token": " impossible",
                                        "logprob": -0.06528423,
                                        "bytes": [
                                            32,
                                            105,
                                            109,
                                            112,
                                            111,
                                            115,
                                            115,
                                            105,
                                            98,
                                            108,
                                            101,
                                        ],
                                    }
                                ],
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-9HzZIMCtVq7CbTmdwEZrktiTeoiYe",
            "object": "chat.completion.chunk",
            "created": 1714075272,
            "model": "gpt-4-0613",
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "—even"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "—even",
                                "logprob": -9999.0,
                                "bytes": [226, 128, 148, 101, 118, 101, 110],
                                "top_logprobs": [
                                    {
                                        "token": " to",
                                        "logprob": -0.12302828,
                                        "bytes": [32, 116, 111],
                                    }
                                ],
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-9HzZIMCtVq7CbTmdwEZrktiTeoiYe",
            "object": "chat.completion.chunk",
            "created": 1714075272,
            "model": "gpt-4-0613",
            "system_fingerprint": None,
            "choices": [
                {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "length"}
            ],
        },
        {
            "id": "chatcmpl-9HzZIMCtVq7CbTmdwEZrktiTeoiYe",
            "object": "chat.completion.chunk",
            "created": 1714075272,
            "model": "gpt-4-0613",
            "system_fingerprint": None,
            "choices": [
                {"index": 1, "delta": {}, "logprobs": None, "finish_reason": "stop"}
            ],
        },
    ]

    litellm.set_verbose = True

    chunk_list = []
    for chunk in chunks:
        new_chunk = litellm.ModelResponse(stream=True, id=chunk["id"])
        if "choices" in chunk and isinstance(chunk["choices"], list):
            print("INSIDE CHUNK CHOICES!")
            new_choices = []
            for choice in chunk["choices"]:
                if isinstance(choice, litellm.utils.StreamingChoices):
                    _new_choice = choice
                elif isinstance(choice, dict):
                    _new_choice = litellm.utils.StreamingChoices(**choice)
                new_choices.append(_new_choice)
            new_chunk.choices = new_choices
        chunk_list.append(new_chunk)

    completion_stream = ModelResponseListIterator(model_responses=chunk_list)

    response = litellm.CustomStreamWrapper(
        completion_stream=completion_stream,
        model="gpt-4-0613",
        custom_llm_provider="cached_response",
        logging_obj=litellm.Logging(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": "Hey"}],
            stream=True,
            call_type="completion",
            start_time=time.time(),
            litellm_call_id="12345",
            function_id="1245",
        ),
    )

    for idx, chunk in enumerate(response):
        chunk_dict = {}
        try:
            chunk_dict = chunk.model_dump(exclude_none=True)
        except:
            chunk_dict = chunk.dict(exclude_none=True)

        chunk_dict.pop("created")
        chunks[idx].pop("created")
        if chunks[idx]["system_fingerprint"] is None:
            chunks[idx].pop("system_fingerprint", None)
        if idx == 0:
            for choice in chunk_dict["choices"]:
                if "role" in choice["delta"]:
                    choice["delta"].pop("role")

        for choice in chunks[idx]["choices"]:
            # ignore finish reason None - since our pydantic object is set to exclude_none = true
            if "finish_reason" in choice and choice["finish_reason"] is None:
                choice.pop("finish_reason")
            if "logprobs" in choice and choice["logprobs"] is None:
                choice.pop("logprobs")

        assert (
            chunk_dict == chunks[idx]
        ), f"idx={idx} translated chunk = {chunk_dict} != openai chunk = {chunks[idx]}"
