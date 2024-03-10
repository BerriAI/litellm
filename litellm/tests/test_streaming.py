#### What this tests ####
#    This tests streaming for the completion endpoint

import sys, os, asyncio
import traceback
import time, pytest
from pydantic import BaseModel

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from dotenv import load_dotenv

load_dotenv()
import litellm
from litellm import (
    completion,
    acompletion,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ModelResponse,
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


def validate_last_format(chunk):
    assert isinstance(chunk, ModelResponse), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."

    for choice in chunk["choices"]:
        assert isinstance(choice["index"], int), "'index' should be an integer."
        assert isinstance(
            choice["finish_reason"], str
        ), "'finish_reason' should be a string."


def streaming_format_tests(idx, chunk):
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


def test_completion_gemini_stream():
    try:
        litellm.set_verbose = True
        print("Streaming gemini response")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "how does a court case get to the Supreme Court?",
            },
        ]
        print("testing gemini streaming")
        response = completion(model="gemini/gemini-pro", messages=messages, stream=True)
        print(f"type of response at the top: {response}")
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
    except litellm.APIError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_acompletion_gemini_stream():
    try:
        litellm.set_verbose = True
        print("Streaming gemini response")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What do you know?",
            },
        ]
        print("testing gemini streaming")
        response = await acompletion(
            model="gemini/gemini-pro", messages=messages, max_tokens=50, stream=True
        )
        print(f"type of response at the top: {response}")
        complete_response = ""
        idx = 0
        # Add any assertions here to check the response
        async for chunk in response:
            print(f"chunk in acompletion gemini: {chunk}")
            print(chunk.choices[0].delta)
            chunk, finished = streaming_format_tests(idx, chunk)
            if finished:
                break
            print(f"chunk: {chunk}")
            complete_response += chunk
            idx += 1
        print(f"completion_response: {complete_response}")
        if complete_response.strip() == "":
            raise Exception("Empty response received")
    except litellm.APIError as e:
        pass
    except Exception as e:
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
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            if finished:
                break
            complete_response += chunk
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

# def test_completion_replicate_stream():
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


def test_bedrock_claude_3_streaming():
    try:
        litellm.set_verbose = True
        response: ModelResponse = completion(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            messages=messages,
            max_tokens=10,
            stream=True,
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
    except RateLimitError:
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


def test_sagemaker_weird_response():
    """
    When the stream ends, flush any remaining holding chunks.
    """
    try:
        from litellm.llms.sagemaker import TokenIterator
        import json
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


@pytest.mark.skip(reason="AWS Suspended Account")
@pytest.mark.asyncio
async def test_sagemaker_streaming_async():
    try:
        messages = [{"role": "user", "content": "Hey, how's it going?"}]
        litellm.set_verbose = True
        response = await litellm.acompletion(
            model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
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


@pytest.mark.skip(reason="AWS Suspended Account")
def test_completion_sagemaker_stream():
    try:
        response = completion(
            model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=messages,
            temperature=0.2,
            max_tokens=80,
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


def test_openai_streaming_and_function_calling():
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
            model="gpt-3.5-turbo",
            tools=tools,
            messages=messages,
            stream=True,
        )
        # Add any assertions here to check the response
        for idx, chunk in enumerate(response):
            if idx == 0:
                assert (
                    chunk.choices[0].delta.tool_calls[0].function.arguments is not None
                )
                assert isinstance(
                    chunk.choices[0].delta.tool_calls[0].function.arguments, str
                )
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
        model="j2-light",
        messages=messages,
        stream=True,
        max_tokens=5,
    )
    print(response)

    for chunk in response:
        print(chunk["choices"][0])


# test_success_callback_streaming()

#### STREAMING + FUNCTION CALLING ###
from pydantic import BaseModel
from typing import List, Optional


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
    system_fingerprint: str
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
    system_fingerprint: str
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
            elif idx == 1:
                validate_second_streaming_function_calling_chunk(chunk=chunk)
            elif chunk.choices[0].finish_reason is not None:  # last chunk
                validate_final_streaming_function_calling_chunk(chunk=chunk)
            idx += 1
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        raise e
