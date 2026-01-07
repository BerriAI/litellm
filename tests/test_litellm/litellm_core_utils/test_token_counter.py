#### What this tests ####
#    This tests litellm.token_counter.token_counter() function
import os
import sys
import time
import traceback
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm import create_pretrained_tokenizer, decode, encode, get_modified_max_tokens
from litellm import token_counter as token_counter_old
from litellm.litellm_core_utils.token_counter import token_counter as token_counter_new
from tests.large_text import text
from tests.test_litellm.litellm_core_utils.messages_with_counts import (
    MESSAGES_TEXT,
    MESSAGES_WITH_IMAGES,
    MESSAGES_WITH_TOOLS,
)


def token_counter_both_assert_same(**args):
    new = token_counter_new(**args)
    old = token_counter_old(**args)
    assert new == old, f"New token counter {new} does not match old token counter {old}"
    return new


## Choose which token_counter the test will use.

# token_counter = token_counter_new
# token_counter = token_counter_old
token_counter = token_counter_both_assert_same


def test_token_counter_basic():
    assert (
        token_counter(
            model="claude-2",
            messages=[
                {
                    "role": "user",
                    "content": "This is a long message that definitely exceeds the token limit.",
                }
            ],
        )
        == 19
    )


def test_token_counter_with_prefix():
    messages = [
        {"role": "user", "content": "Who won the world cup in 2022?"},
        {"role": "assistant", "content": "Argentina", "prefix": True}
    ]
    tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
    assert tokens == 22 , f"Expected 22 tokens, got {tokens}"


def test_token_counter_normal_plus_function_calling():
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
    assert tokens == 80


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
    MESSAGES_TEXT,
)
def test_token_counter_count_response_tokens(message_count_pair):
    counted_tokens = token_counter(
        model="gpt-35-turbo",
        messages=[message_count_pair["message"]],
        count_response_tokens=True,
    )
    # 3 tokens are not added because of count_response_tokens=True
    expected = message_count_pair["count"] - 3
    assert counted_tokens == expected


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
    actual_diff = counted_tokens - expected_tokens

    if "count-tolerate" in message_count_pair:
        if message_count_pair["count-tolerate"] == counted_tokens:
            pass  # expected
        else:
            tolerated_diff = message_count_pair["count-tolerate"] - expected_tokens
            assert (
                actual_diff <= tolerated_diff
            ), f"Expected {expected_tokens} tokens, got {counted_tokens}. Counted tokens is only allowed to be off by {tolerated_diff} in the over-counting direction."
            if actual_diff != tolerated_diff:
                raise NeedsToleranceUpdateError(
                    f"SOMETHING BROKEN GOT FIXED! THIS is good! Adjust 'count-tolerate' from {message_count_pair['count-tolerate']} to {counted_tokens}"
                )

    else:
        assert (
            expected_tokens == counted_tokens
        ), f"Expected {expected_tokens} tokens, got {counted_tokens}."


class NeedsToleranceUpdateError(Exception):
    """Custom exception to mark tests that have improved"""

    pass


def test_tokenizers():
    try:
        ### test the openai, claude, cohere and llama2 tokenizers.
        ### The tokenizer value should be different for all
        sample_text = "Hellö World, this is my input string! My name is ishaan CTO"

        # openai tokenizer
        openai_tokens = token_counter(model="gpt-3.5-turbo", text=sample_text)

        # claude tokenizer
        claude_tokens = token_counter(
            model="claude-3-5-haiku-20241022", text=sample_text
        )

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
        claude_tokens = encode(model="claude-3-5-haiku-20241022", text=sample_text)

        claude_text = decode(model="claude-3-5-haiku-20241022", tokens=claude_tokens)

        assert claude_text == sample_text

        # cohere encoding + decoding
        cohere_tokens = encode(model="command-nightly", text=sample_text)
        cohere_text = decode(model="command-nightly", tokens=cohere_tokens)

        assert cohere_text == sample_text

        # llama2 encoding + decoding
        llama2_tokens = encode(model="meta-llama/Llama-2-7b-chat", text=sample_text)
        llama2_text = decode(
            model="meta-llama/Llama-2-7b-chat", tokens=llama2_tokens.ids  # type: ignore
        )

        assert llama2_text == sample_text
    except Exception as e:
        pytest.fail(f"An exception occured: {e}\n{traceback.format_exc()}")


# test_encoding_and_decoding()


def test_gpt_vision_token_counting():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png",
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


@pytest.mark.skip(
    reason="Skipping this test temporarily because it relies on a function being called that I am removing."
)
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
    from litellm.litellm_core_utils.token_counter import get_image_dimensions

    width, height = get_image_dimensions(data=img_url)

    print(width, height)

    assert width is not None
    assert height is not None


def test_token_encode_disallowed_special():
    encode(model="gpt-3.5-turbo", text="Hello, world! <|endoftext|>")
    token_counter(model="gpt-3.5-turbo", text="Hello, world! <|endoftext|>")


def test_token_counter():
    try:
        messages = [{"role": "user", "content": "hi how are you what time is it"}]
        tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
        print("gpt-35-turbo")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="claude-2", messages=messages)
        print("claude-2")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="gemini/chat-bison", messages=messages)
        print("gemini/chat-bison")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="ollama/llama2", messages=messages)
        print("ollama/llama2")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="anthropic.claude-instant-v1", messages=messages)
        print("anthropic.claude-instant-v1")
        print(tokens)
        assert tokens > 0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


import unittest
from unittest.mock import MagicMock, patch

from litellm.utils import _select_tokenizer_helper, claude_json_str, encoding


class TestTokenizerSelection(unittest.TestCase):
    @patch("litellm.utils.Tokenizer.from_pretrained")
    def test_llama3_tokenizer_api_failure(self, mock_from_pretrained):
        # Setup mock to raise an error
        mock_from_pretrained.side_effect = Exception("Failed to load tokenizer")

        # Test with llama-3 model
        result = _select_tokenizer_helper("llama-3-7b")

        # Verify the attempt to load Llama-3 tokenizer
        mock_from_pretrained.assert_called_once_with("Xenova/llama-3-tokenizer")

        # Verify fallback to OpenAI tokenizer
        self.assertEqual(result["type"], "openai_tokenizer")
        self.assertEqual(result["tokenizer"], encoding)

    @patch("litellm.utils.Tokenizer.from_pretrained")
    def test_cohere_tokenizer_api_failure(self, mock_from_pretrained):
        # Setup mock to raise an error
        mock_from_pretrained.side_effect = Exception("Failed to load tokenizer")

        # Add Cohere model to the list for testing
        litellm.cohere_models = ["command-r-v1"]

        # Test with Cohere model
        result = _select_tokenizer_helper("command-r-v1")

        # Verify the attempt to load Cohere tokenizer
        mock_from_pretrained.assert_called_once_with(
            "Xenova/c4ai-command-r-v01-tokenizer"
        )

        # Verify fallback to OpenAI tokenizer
        self.assertEqual(result["type"], "openai_tokenizer")
        self.assertEqual(result["tokenizer"], encoding)

    @patch("litellm.utils.Tokenizer.from_str")
    def test_claude_tokenizer_api_failure(self, mock_from_str):
        # Setup mock to raise an error
        mock_from_str.side_effect = Exception("Failed to load tokenizer")

        # Add Claude model to the list for testing
        litellm.anthropic_models = ["claude-2"]

        # Test with Claude model
        result = _select_tokenizer_helper("claude-2")

        # Verify the attempt to load Claude tokenizer
        mock_from_str.assert_called_once_with(claude_json_str)

        # Verify fallback to OpenAI tokenizer
        self.assertEqual(result["type"], "openai_tokenizer")
        self.assertEqual(result["tokenizer"], encoding)

    @patch("litellm.utils.Tokenizer.from_pretrained")
    def test_llama2_tokenizer_api_failure(self, mock_from_pretrained):
        # Setup mock to raise an error
        mock_from_pretrained.side_effect = Exception("Failed to load tokenizer")

        # Test with Llama-2 model
        result = _select_tokenizer_helper("llama-2-7b")

        # Verify the attempt to load Llama-2 tokenizer
        mock_from_pretrained.assert_called_once_with(
            "hf-internal-testing/llama-tokenizer"
        )

        # Verify fallback to OpenAI tokenizer
        self.assertEqual(result["type"], "openai_tokenizer")
        self.assertEqual(result["tokenizer"], encoding)

    @patch("litellm.utils._return_huggingface_tokenizer")
    def test_disable_hf_tokenizer_download(self, mock_return_huggingface_tokenizer):
        # Use pytest.MonkeyPatch() directly instead of fixture
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(litellm, "disable_hf_tokenizer_download", True)

        result = _select_tokenizer_helper("grok-32r22r")
        mock_return_huggingface_tokenizer.assert_not_called()
        assert result["type"] == "openai_tokenizer"
        assert result["tokenizer"] == encoding


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "claude-3-opus-20240229",
    ],
)
@pytest.mark.parametrize(
    "messages",
    [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "These are some sample images from a movie. Based on these images, what do you think the tone of the movie is?",
                    },
                    {
                        "type": "text",
                        "image_url": {
                            "url": "https://gratisography.com/wp-content/uploads/2024/11/gratisography-augmented-reality-800x525.jpg",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "These are some sample images from a movie. Based on these images, what do you think the tone of the movie is?",
                    },
                    {
                        "type": "text",
                        "image_url": {
                            "url": "https://gratisography.com/wp-content/uploads/2024/11/gratisography-augmented-reality-800x525.jpg",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
    ],
)
def test_bad_input_token_counter(model, messages):
    """
    Safely handle bad input for token counter.
    """
    token_counter(
        model=model,
        messages=messages,
        default_token_count=1000,
    )


def test_token_counter_with_anthropic_tool_use():
    """
    Test that _count_anthropic_content() correctly handles tool_use blocks.
    
    Validates that:
    - 'name' field is counted (string)
    - 'input' field is counted (dict serialized to string)
    - Metadata fields ('type', 'id') are skipped
    """
    messages = [
        {
            "role": "user",
            "content": "What's the weather in San Francisco?"
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I'll check the weather for you."
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01234567890",  # Should be skipped
                    "name": "get_weather",  # Should be counted
                    "input": {  # Should be counted (serialized)
                        "location": "San Francisco, CA",
                        "unit": "fahrenheit"
                    }
                }
            ]
        }
    ]
    
    tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
    assert tokens > 0, f"Expected positive token count, got {tokens}"
    # Should count: user message + "I'll check" text + "get_weather" name + input dict
    assert tokens > 15, f"Expected reasonable token count for message with tool_use, got {tokens}"


def test_token_counter_with_anthropic_tool_result():
    """
    Test that _count_anthropic_content() correctly handles tool_result blocks.
    
    Validates that:
    - 'content' field (when string) is counted
    - Metadata fields ('type', 'tool_use_id') are skipped
    - Full conversation with tool_use → tool_result flow works
    """
    messages = [
        {
            "role": "user",
            "content": "What's the weather in San Francisco?"
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01234567890",
                    "name": "get_weather",
                    "input": {
                        "location": "San Francisco, CA"
                    }
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01234567890",  # Should be skipped
                    "content": "The weather in San Francisco is 65°F and sunny."  # Should be counted
                }
            ]
        }
    ]
    
    tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
    assert tokens > 0, f"Expected positive token count, got {tokens}"
    assert tokens > 25, f"Expected reasonable token count for conversation with tool_result, got {tokens}"


def test_token_counter_with_nested_tool_result():
    """
    Test that _count_anthropic_content() recursively handles nested content lists.
    
    Validates that:
    - tool_result with 'content' as a list (not string) is handled
    - Nested content blocks are recursively counted via _count_content_list()
    - TypedDict inference correctly identifies list fields
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01234567890",
                    "content": [  # Nested list - should recursively count
                        {
                            "type": "text",
                            "text": "The weather in San Francisco is 65°F and sunny."
                        },
                        {
                            "type": "text",
                            "text": "UV index is moderate."
                        }
                    ]
                }
            ]
        }
    ]
    
    tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
    assert tokens > 0, f"Expected positive token count, got {tokens}"
    # Should count both nested text blocks
    assert tokens > 15, f"Expected reasonable token count for nested tool_result, got {tokens}"


def test_token_counter_tool_use_and_result_combined():
    """
    Test dynamic field inference with multiple tool_use and tool_result blocks.
    
    Validates that:
    - Multiple tool_use blocks in same message are handled
    - Multiple tool_result blocks in same message are handled
    - skip_fields correctly filters metadata across all blocks
    - Full realistic conversation flow works end-to-end
    """
    messages = [
        {
            "role": "user",
            "content": "What's the weather in San Francisco and New York?"
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I'll check the weather in both cities for you."
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01A",
                    "name": "get_weather",
                    "input": {"location": "San Francisco, CA"}
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01B",
                    "name": "get_weather",
                    "input": {"location": "New York, NY"}
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A",
                    "content": "San Francisco: 65°F, sunny"
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01B",
                    "content": "New York: 45°F, cloudy"
                }
            ]
        },
        {
            "role": "assistant",
            "content": "The weather in San Francisco is 65°F and sunny, while New York is cooler at 45°F and cloudy."
        }
    ]
    
    tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
    assert tokens > 0, f"Expected positive token count, got {tokens}"
    # Should count all text, tool names, inputs, and results
    assert tokens > 60, f"Expected substantial token count for full tool conversation, got {tokens}"


def test_token_counter_with_image_url():
    """
    Test that _count_image_tokens() correctly handles image_url content blocks.
    
    Validates that:
    - image_url as dict with 'url' and 'detail' is handled
    - image_url as string is handled
    - 'detail' field validation works ('low', 'high', 'auto')
    - calculate_img_tokens is called with correct parameters
    """
    # Test with dict format (detail: low)
    messages_dict = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "low"  # Should use low token count (85 base tokens)
                    }
                }
            ]
        }
    ]
    
    tokens_dict = token_counter(
        model="gpt-3.5-turbo",
        messages=messages_dict,
        use_default_image_token_count=True  # Avoid actual HTTP request
    )
    assert tokens_dict > 0, f"Expected positive token count, got {tokens_dict}"
    assert tokens_dict > 85, f"Expected at least base image tokens, got {tokens_dict}"
    
    # Test with string format (defaults to auto/low)
    messages_str = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": "https://example.com/image.jpg"  # String format
                }
            ]
        }
    ]
    
    tokens_str = token_counter(
        model="gpt-3.5-turbo",
        messages=messages_str,
        use_default_image_token_count=True
    )
    assert tokens_str > 0, f"Expected positive token count for string image_url, got {tokens_str}"
    
    # Test invalid detail value raises error
    messages_invalid = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "invalid"  # Should raise ValueError
                    }
                }
            ]
        }
    ]
    
    try:
        token_counter(model="gpt-3.5-turbo", messages=messages_invalid)
        assert False, "Expected ValueError for invalid detail value"
    except ValueError as e:
        assert "Invalid detail value" in str(e), f"Expected detail validation error, got: {e}"

