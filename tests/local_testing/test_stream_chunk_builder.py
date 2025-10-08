import asyncio
import os
import sys
import time
import traceback

import pytest
from typing import List
from litellm.types.utils import StreamingChoices, ChatCompletionAudioResponse


def check_non_streaming_response(completion):
    assert completion.choices[0].message.audio is not None, "Audio response is missing"
    print("audio", completion.choices[0].message.audio)
    assert isinstance(
        completion.choices[0].message.audio, ChatCompletionAudioResponse
    ), "Invalid audio response type"
    assert len(completion.choices[0].message.audio.data) > 0, "Audio data is empty"


sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import os

import dotenv
from openai import OpenAI

import litellm
import stream_chunk_testdata
from litellm import completion, stream_chunk_builder

dotenv.load_dotenv()

user_message = "What is the current weather in Boston?"
messages = [{"content": user_message, "role": "user"}]

function_schema = {
    "name": "get_weather",
    "description": "gets the current weather",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        "required": ["location"],
    },
}


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

# def test_stream_chunk_builder_tools():
#     try:
#       litellm.set_verbose = False
#       response = client.chat.completions.create(
#           model="gpt-3.5-turbo",
#           messages=messages,
#           tools=tools_schema,
#           # stream=True,
#           # complete_response=True # runs stream_chunk_builder under-the-hood
#       )

#       print(f"response: {response}")
#       print(f"response usage: {response.usage}")
#     except Exception as e:
#        pytest.fail(f"An exception occurred - {str(e)}")

# test_stream_chunk_builder_tools()


def test_stream_chunk_builder_litellm_function_call():
    try:
        litellm.set_verbose = False
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=[function_schema],
            # stream=True,
            # complete_response=True # runs stream_chunk_builder under-the-hood
        )

        print(f"response: {response}")
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# test_stream_chunk_builder_litellm_function_call()


def test_stream_chunk_builder_litellm_tool_call():
    try:
        litellm.set_verbose = True
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=tools_schema,
            stream=True,
            complete_response=True,
        )

        print(f"complete response: {response}")
        print(f"complete response usage: {response.usage}")
        assert response.usage.completion_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.completion_tokens + response.usage.prompt_tokens
        )
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# test_stream_chunk_builder_litellm_tool_call()


def test_stream_chunk_builder_litellm_tool_call_regular_message():
    try:
        messages = [{"role": "user", "content": "Hey, how's it going?"}]
        # litellm.set_verbose = True
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=tools_schema,
            stream=True,
            complete_response=True,
        )

        print(f"complete response: {response}")
        print(f"complete response usage: {response.usage}")
        assert response.usage.completion_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.completion_tokens + response.usage.prompt_tokens
        )

        # check provider is in hidden params
        print("hidden params", response._hidden_params)
        assert response._hidden_params["custom_llm_provider"] == "openai"

    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# test_stream_chunk_builder_litellm_tool_call_regular_message()


def test_stream_chunk_builder_litellm_mixed_calls():
    response = stream_chunk_builder(stream_chunk_testdata.chunks)
    assert (
        response.choices[0].message.content
        == "To answer your question about how many rows are in the 'users' table, I'll need to run a SQL query. Let me do that for you."
    )

    print(response.choices[0].message.tool_calls[0].to_dict())

    assert len(response.choices[0].message.tool_calls) == 1
    assert response.choices[0].message.tool_calls[0].to_dict() == {
        "function": {
            "arguments": '{"query": "SELECT COUNT(*) FROM users;"}',
            "name": "sql_query",
        },
        "id": "toolu_01H3AjkLpRtGQrof13CBnWfK",
        "type": "function",
    }


def test_stream_chunk_builder_litellm_empty_chunks():
    with pytest.raises(litellm.APIError):
        response = stream_chunk_builder(chunks=None)

    response = stream_chunk_builder(chunks=[])
    assert response is None


def test_stream_chunk_builder_multiple_tool_calls():
    init_chunks = [
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_X9P9B6STj7ze8OsJCGkfoN94",
                                "function": {"arguments": "", "name": "exponentiate"},
                                "type": "function",
                                "index": 0,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": '{"ba'},
                                "type": "function",
                                "index": 0,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": 'se": '},
                                "type": "function",
                                "index": 0,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": '3, "ex'},
                                "type": "function",
                                "index": 0,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": "pone"},
                                "type": "function",
                                "index": 0,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": 'nt": '},
                                "type": "function",
                                "index": 0,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": "5}"},
                                "type": "function",
                                "index": 0,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_Qq8yDeRx7v276abRcLrYORdW",
                                "function": {"arguments": "", "name": "add"},
                                "type": "function",
                                "index": 1,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": '{"fi'},
                                "type": "function",
                                "index": 1,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": "rst_i"},
                                "type": "function",
                                "index": 1,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": 'nt": 1'},
                                "type": "function",
                                "index": 1,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": '2, "'},
                                "type": "function",
                                "index": 1,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": "secon"},
                                "type": "function",
                                "index": 1,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": 'd_int"'},
                                "type": "function",
                                "index": 1,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": ": 3}"},
                                "type": "function",
                                "index": 1,
                            }
                        ],
                    },
                }
            ],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
        {
            "id": "chatcmpl-A5kCnzaxRsknd6008552ZhDi71yPt",
            "choices": [{"finish_reason": "tool_calls", "index": 0, "delta": {}}],
            "created": 1725932618,
            "model": "gpt-4o-2024-08-06",
            "object": "chat.completion.chunk",
            "system_fingerprint": "fp_b2ffeb16ee",
        },
    ]

    chunks = []
    for chunk in init_chunks:
        chunks.append(litellm.ModelResponse(**chunk, stream=True))
    response = stream_chunk_builder(chunks=chunks)

    print(f"Returned response: {response}")
    completed_response = {
        "id": "chatcmpl-A61mXjvcRX0Xr2IiojN9TPiy1P3Fm",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"base": 3, "exponent": 5}',
                                "name": "exponentiate",
                            },
                            "id": "call_X9P9B6STj7ze8OsJCGkfoN94",
                            "type": "function",
                        },
                        {
                            "function": {
                                "arguments": '{"first_int": 12, "second_int": 3}',
                                "name": "add",
                            },
                            "id": "call_Qq8yDeRx7v276abRcLrYORdW",
                            "type": "function",
                        },
                    ],
                    "function_call": None,
                },
            }
        ],
        "created": 1726000181,
        "model": "gpt-4o-2024-05-13",
        "object": "chat.completion",
        "system_fingerprint": "fp_25624ae3a5",
        "usage": {"completion_tokens": 55, "prompt_tokens": 127, "total_tokens": 182},
        "service_tier": None,
    }

    expected_response = litellm.ModelResponse(**completed_response)

    print(f"\n\nexpected_response:\n{expected_response}\n\n")
    assert (
        expected_response.choices == response.choices
    ), "\nGot={}\n, Expected={}\n".format(response.choices, expected_response.choices)


def test_stream_chunk_builder_openai_prompt_caching():
    from openai import OpenAI
    from pydantic import BaseModel

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks: List[litellm.ModelResponse] = []
    usage_obj = None
    for chunk in chat_completion:
        chunks.append(litellm.ModelResponse(**chunk.model_dump(), stream=True))

    print(f"chunks: {chunks}")

    usage_obj: litellm.Usage = chunks[-1].usage  # type: ignore

    response = stream_chunk_builder(chunks=chunks)
    print(f"response: {response}")
    print(f"response usage: {response.usage}")
    for k, v in usage_obj.model_dump(exclude_none=True).items():
        print(k, v)
        response_usage_value = getattr(response.usage, k)  # type: ignore
        print(f"response_usage_value: {response_usage_value}")
        print(f"type: {type(response_usage_value)}")
        if isinstance(response_usage_value, BaseModel):
            assert response_usage_value.model_dump(exclude_none=True) == v
        else:
            assert response_usage_value == v


def test_stream_chunk_builder_openai_audio_output_usage():
    from pydantic import BaseModel
    from openai import OpenAI
    from typing import Optional

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "pcm16"},
            messages=[{"role": "user", "content": "response in 1 word - yes or no"}],
            stream=True,
            stream_options={"include_usage": True},
        )
    except Exception as e:
        if "openai-internal" in str(e):
            pytest.skip("Skipping test due to openai-internal error")

    chunks = []
    for chunk in completion:
        chunks.append(litellm.ModelResponse(**chunk.model_dump(), stream=True))

    usage_obj: Optional[litellm.Usage] = None

    for index, chunk in enumerate(chunks):
        if hasattr(chunk, "usage"):
            usage_obj = chunk.usage
            print(f"chunk usage: {chunk.usage}")
            print(f"index: {index}")
            print(f"len chunks: {len(chunks)}")

    print(f"usage_obj: {usage_obj}")
    response = stream_chunk_builder(chunks=chunks)
    print(f"response usage: {response.usage}")
    check_non_streaming_response(response)
    print(f"response: {response}")
    # Convert both usage objects to dictionaries for easier comparison
    usage_dict = usage_obj.model_dump(exclude_none=True)
    response_usage_dict = response.usage.model_dump(exclude_none=True)

    # Simple dictionary comparison
    assert (
        usage_dict == response_usage_dict
    ), f"\nExpected: {usage_dict}\nGot: {response_usage_dict}"


def test_stream_chunk_builder_empty_initial_chunk():
    from litellm.litellm_core_utils.streaming_chunk_builder_utils import (
        ChunkProcessor,
    )

    chunks = [
        {"id": ""},
        {"id": "1"},
        {"id": "1"},
    ]

    id = ChunkProcessor._get_chunk_id(chunks)
    assert id == "1"


def test_stream_chunk_builder_tool_calls_list():
    from litellm.litellm_core_utils.streaming_chunk_builder_utils import (
        ChunkProcessor,
    )
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Function,
        ModelResponseStream,
        Delta,
        StreamingChoices,
        ChatCompletionDeltaToolCall,
    )

    chunks = [
        ModelResponseStream(
            id="chatcmpl-f323f7a5-2da0-4f86-8ed7-c653c5a359d9",
            created=1736388417,
            model="llama-3.3-70b-versatile",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="",
                        role="assistant",
                        function_call=None,
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_9y79",
                                function=Function(
                                    arguments='{"location": "San Francisco", "unit": "celsius"}',
                                    name="get_current_weather",
                                ),
                                type="function",
                                index=0,
                            )
                        ],
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            stream_options=None,
        ),
        ModelResponseStream(
            id="chatcmpl-f323f7a5-2da0-4f86-8ed7-c653c5a359d9",
            created=1736388417,
            model="llama-3.3-70b-versatile",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="",
                        role=None,
                        function_call=None,
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_pfp7",
                                function=Function(
                                    arguments='{"location": "Tokyo", "unit": "celsius"}',
                                    name="get_current_weather",
                                ),
                                type="function",
                                index=1,
                            )
                        ],
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            stream_options=None,
        ),
        ModelResponseStream(
            id="chatcmpl-f323f7a5-2da0-4f86-8ed7-c653c5a359d9",
            created=1736388417,
            model="llama-3.3-70b-versatile",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="",
                        role=None,
                        function_call=None,
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_hyj5",
                                function=Function(
                                    arguments='{"location": "Paris", "unit": "celsius"}',
                                    name="get_current_weather",
                                ),
                                type="function",
                                index=2,
                            )
                        ],
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            stream_options=None,
        ),
    ]

    processor = ChunkProcessor(chunks=chunks)

    tool_calls = processor.get_combined_tool_content(tool_call_chunks=chunks)
    print(f"tool_calls: {tool_calls}")
    assert len(tool_calls) == 3


import json


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


@pytest.fixture(scope="module", autouse=True)
def load_env():
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": litellm.utils.function_to_dict(get_current_weather),
        }
    ]
    OPENAI_GPT4oMINI = {
        "messages": messages,
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "tools": tools,
        "stream": True,
    }
    LLAMA3_3 = {
        "messages": messages,
        "model": "groq/llama-3.3-70b-versatile",
        "api_base": "https://api.groq.com/openai/v1",
        "temperature": 0.0,
        "tools": tools,
        "stream": True,
    }
    return OPENAI_GPT4oMINI, LLAMA3_3


def execute_completion(opts: dict):
    partial_streaming_chunks = []
    response_gen = litellm.completion(**opts)
    for i, part in enumerate(response_gen):
        partial_streaming_chunks.append(part)
    print("\n\n")
    print(f"partial_streaming_chunks: {partial_streaming_chunks}")
    print("\n\n")
    assembly = litellm.stream_chunk_builder(partial_streaming_chunks)
    print(f"assembly.choices[0].message.tool_calls: {assembly.choices[0].message.tool_calls}")
    print(assembly.choices[0].message.tool_calls)
    for tool_call in assembly.choices[0].message.tool_calls:
        json.loads(tool_call.function.arguments) # assert valid json - https://github.com/BerriAI/litellm/issues/10034


def test_grok_bug(load_env):
    litellm.set_verbose = True
    _, LLAMA3_3 = load_env
    execute_completion(LLAMA3_3)
