#### What this tests ####
#    This tests litellm.token_counter() function

import os
import sys
import traceback

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
from unittest.mock import AsyncMock, MagicMock, patch

from litellm import (
    create_pretrained_tokenizer,
    decode,
    encode,
    get_modified_max_tokens,
    token_counter,
)
from litellm.tests.large_text import text


def test_token_counter_normal_plus_function_calling():
    try:
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "content1"},
            {"role": "assistant", "content": "content2"},
            {"role": "user", "content": "conten3"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_E0lOb1h6qtmflUyok4L06TgY",
                        "function": {
                            "arguments": '{"query":"search query","domain":"google.ca","gl":"ca","hl":"en"}',
                            "name": "SearchInternet",
                        },
                        "type": "function",
                    }
                ],
            },
            {
                "tool_call_id": "call_E0lOb1h6qtmflUyok4L06TgY",
                "role": "tool",
                "name": "SearchInternet",
                "content": "tool content",
            },
        ]
        tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
        print(f"tokens: {tokens}")
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# test_token_counter_normal_plus_function_calling()


def test_tokenizers():
    try:
        ### test the openai, claude, cohere and llama2 tokenizers.
        ### The tokenizer value should be different for all
        sample_text = "Hellö World, this is my input string! My name is ishaan CTO"

        # openai tokenizer
        openai_tokens = token_counter(model="gpt-3.5-turbo", text=sample_text)

        # claude tokenizer
        claude_tokens = token_counter(model="claude-instant-1", text=sample_text)

        # cohere tokenizer
        cohere_tokens = token_counter(model="command-nightly", text=sample_text)

        # llama2 tokenizer
        llama2_tokens = token_counter(
            model="meta-llama/Llama-2-7b-chat", text=sample_text
        )

        # llama3 tokenizer (also testing custom tokenizer)
        llama3_tokens_1 = token_counter(
            model="meta-llama/llama-3-70b-instruct", text=sample_text
        )

        llama3_tokenizer = create_pretrained_tokenizer("Xenova/llama-3-tokenizer")
        llama3_tokens_2 = token_counter(
            custom_tokenizer=llama3_tokenizer, text=sample_text
        )

        print(
            f"openai tokens: {openai_tokens}; claude tokens: {claude_tokens}; cohere tokens: {cohere_tokens}; llama2 tokens: {llama2_tokens}; llama3 tokens: {llama3_tokens_1}"
        )

        # assert that all token values are different
        assert (
            openai_tokens != llama2_tokens != llama3_tokens_1
        ), "Token values are not different."

        assert (
            llama3_tokens_1 == llama3_tokens_2
        ), "Custom tokenizer is not being used! It has been configured to use the same tokenizer as the built in llama3 tokenizer and the results should be the same."

        print("test tokenizer: It worked!")
    except Exception as e:
        pytest.fail(f"An exception occured: {e}")


# test_tokenizers()


def test_encoding_and_decoding():
    try:
        sample_text = "Hellö World, this is my input string!"
        # openai encoding + decoding
        openai_tokens = encode(model="gpt-3.5-turbo", text=sample_text)
        openai_text = decode(model="gpt-3.5-turbo", tokens=openai_tokens)

        assert openai_text == sample_text

        # claude encoding + decoding
        claude_tokens = encode(model="claude-instant-1", text=sample_text)
        claude_text = decode(model="claude-instant-1", tokens=claude_tokens.ids)

        assert claude_text == sample_text

        # cohere encoding + decoding
        cohere_tokens = encode(model="command-nightly", text=sample_text)
        cohere_text = decode(model="command-nightly", tokens=cohere_tokens)

        assert cohere_text == sample_text

        # llama2 encoding + decoding
        llama2_tokens = encode(model="meta-llama/Llama-2-7b-chat", text=sample_text)
        llama2_text = decode(
            model="meta-llama/Llama-2-7b-chat", tokens=llama2_tokens.ids
        )

        assert llama2_text == sample_text
    except Exception as e:
        pytest.fail(f"An exception occured: {e}")


# test_encoding_and_decoding()


def test_gpt_vision_token_counting():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
            ],
        }
    ]
    tokens = token_counter(model="gpt-4-vision-preview", messages=messages)
    print(f"tokens: {tokens}")


# test_gpt_vision_token_counting()


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4-vision-preview",
        "gpt-4o",
        "claude-3-opus-20240229",
        "command-nightly",
        "mistral/mistral-tiny",
    ],
)
def test_load_test_token_counter(model):
    """
    Token count large prompt 100 times.

    Assert time taken is < 1.5s.
    """
    import tiktoken

    messages = [{"role": "user", "content": text}] * 10

    start_time = time.time()
    for _ in range(10):
        _ = token_counter(model=model, messages=messages)
        # enc.encode("".join(m["content"] for m in messages))

    end_time = time.time()

    total_time = end_time - start_time
    print("model={}, total test time={}".format(model, total_time))
    assert total_time < 10, f"Total encoding time > 10s, {total_time}"


def test_openai_token_with_image_and_text():
    model = "gpt-4o"
    full_request = {
        "model": "gpt-4o",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "json",
                    "parameters": {
                        "type": "object",
                        "required": ["clause"],
                        "properties": {"clause": {"type": "string"}},
                    },
                    "description": "Respond with a JSON object.",
                },
            }
        ],
        "logprobs": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "\n    Just some long text, long long text, and you know it will be longer than 7 tokens definetly.",
                        "type": "text",
                    }
                ],
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "json"}},
        "exclude_models": [],
        "disable_fallback": False,
        "exclude_providers": [],
    }
    messages = full_request.get("messages", [])

    token_count = token_counter(model=model, messages=messages)
    print(token_count)


@pytest.mark.parametrize(
    "model, base_model, input_tokens, user_max_tokens, expected_value",
    [
        ("random-model", "random-model", 1024, 1024, 1024),
        ("command", "command", 1000000, None, None),  # model max = 4096
        ("command", "command", 4000, 256, 96),  # model max = 4096
        ("command", "command", 4000, 10, 10),  # model max = 4096
        ("gpt-3.5-turbo", "gpt-3.5-turbo", 4000, 5000, 4096),  # model max output = 4096
    ],
)
def test_get_modified_max_tokens(
    model, base_model, input_tokens, user_max_tokens, expected_value
):
    """
    - Test when max_output is not known => expect user_max_tokens
    - Test when max_output == max_input,
        - input > max_output, no max_tokens => expect None
        - input + max_tokens > max_output => expect remainder
        - input + max_tokens < max_output => expect max_tokens
    - Test when max_tokens > max_output => expect max_output
    """
    args = locals()
    import litellm

    litellm.token_counter = MagicMock()

    def _mock_token_counter(*args, **kwargs):
        return input_tokens

    litellm.token_counter.side_effect = _mock_token_counter
    print(f"_mock_token_counter: {_mock_token_counter()}")
    messages = [{"role": "user", "content": "Hello world!"}]

    calculated_value = get_modified_max_tokens(
        model=model,
        base_model=base_model,
        messages=messages,
        user_max_tokens=user_max_tokens,
        buffer_perc=0,
        buffer_num=0,
    )

    if expected_value is None:
        assert calculated_value is None
    else:
        assert (
            calculated_value == expected_value
        ), "Got={}, Expected={}, Params={}".format(
            calculated_value, expected_value, args
        )
