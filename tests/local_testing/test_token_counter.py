#### What this tests ####
#    This tests litellm.token_counter() function

import os
import sys
import time
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm import (
    create_pretrained_tokenizer,
    decode,
    encode,
    get_modified_max_tokens,
    token_counter,
)
from litellm.tests.large_text import text
from litellm.tests.messages_with_counts import (
    MESSAGES_TEXT,
    MESSAGES_WITH_IMAGES,
    MESSAGES_WITH_TOOLS,
)


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


@pytest.mark.parametrize(
    "message_count_pair",
    MESSAGES_TEXT,
)
def test_token_counter_textonly(message_count_pair):
    counted_tokens = token_counter(
        model="gpt-35-turbo", messages=[message_count_pair["message"]]
    )
    assert counted_tokens == message_count_pair["count"]


@pytest.mark.parametrize(
    "message_count_pair",
    MESSAGES_WITH_IMAGES,
)
def test_token_counter_with_images(message_count_pair):
    counted_tokens = token_counter(
        model="gpt-4o", messages=[message_count_pair["message"]]
    )
    assert counted_tokens == message_count_pair["count"]


@pytest.mark.parametrize(
    "message_count_pair",
    MESSAGES_WITH_TOOLS,
)
def test_token_counter_with_tools(message_count_pair):
    counted_tokens = token_counter(
        model="gpt-35-turbo",
        messages=[message_count_pair["system_message"]],
        tools=message_count_pair["tools"],
        tool_choice=message_count_pair["tool_choice"],
    )
    expected_tokens = message_count_pair["count"]
    diff = counted_tokens - expected_tokens
    assert (
        diff >= 0 and diff <= 3
    ), f"Expected {expected_tokens} tokens, got {counted_tokens}. Counted tokens is only allowed to be off by 3 in the over-counting direction."


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


def test_empty_tools():
    messages = [{"role": "user", "content": "hey, how's it going?", "tool_calls": None}]

    result = token_counter(
        messages=messages,
    )

    print(result)


def test_gpt_4o_token_counter():
    with patch.object(
        litellm.utils, "openai_token_counter", new=MagicMock()
    ) as mock_client:
        token_counter(
            model="gpt-4o-2024-05-13", messages=[{"role": "user", "content": "Hey!"}]
        )

        mock_client.assert_called()


@pytest.mark.parametrize(
    "img_url",
    [
        "https://blog.purpureus.net/assets/blog/personal_key_rotation/simplified-asset-graph.jpg",
        "data:image/png;base64,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",
    ],
)
def test_img_url_token_counter(img_url):

    from litellm.utils import get_image_dimensions

    width, height = get_image_dimensions(data=img_url)

    print(width, height)

    assert width is not None
    assert height is not None
