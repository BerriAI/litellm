
import pytest

from litellm.llms.openai.responses.count_tokens.transformation import (
    OpenAICountTokensConfig,
)


def test_transform_basic_request():
    """Test basic request with model and input."""
    config = OpenAICountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="gpt-4o",
        input="Hello, how are you?",
    )

    assert result == {
        "model": "gpt-4o",
        "input": "Hello, how are you?",
    }


def test_transform_with_list_input():
    """Test request with list input format."""
    config = OpenAICountTokensConfig()

    input_items = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = config.transform_request_to_count_tokens(
        model="gpt-4o",
        input=input_items,
    )

    assert result["model"] == "gpt-4o"
    assert result["input"] == input_items


def test_transform_includes_instructions():
    """Test that instructions are included when provided."""
    config = OpenAICountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="gpt-4o",
        input="Hello",
        instructions="You are a helpful assistant.",
    )

    assert result["instructions"] == "You are a helpful assistant."
    assert result["model"] == "gpt-4o"
    assert result["input"] == "Hello"


def test_transform_includes_tools():
    """Test that tools are included when provided."""
    config = OpenAICountTokensConfig()

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]

    result = config.transform_request_to_count_tokens(
        model="gpt-4o",
        input="What's the weather?",
        tools=tools,
    )

    assert result["tools"] == tools


def test_transform_no_instructions_no_tools():
    """Test that None values are not included."""
    config = OpenAICountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="gpt-4o",
        input="Hello",
        instructions=None,
        tools=None,
    )

    assert "instructions" not in result
    assert "tools" not in result


def test_messages_to_responses_input_basic():
    """Test converting basic chat messages to Responses API input format."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(
        messages
    )

    assert len(input_items) == 3
    assert input_items[0] == {"role": "user", "content": "Hello"}
    assert input_items[1] == {"role": "assistant", "content": "Hi there!"}
    assert input_items[2] == {"role": "user", "content": "How are you?"}
    assert instructions is None


def test_messages_to_responses_input_with_system():
    """Test that system messages are extracted as instructions."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]

    input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(
        messages
    )

    assert len(input_items) == 1
    assert input_items[0] == {"role": "user", "content": "Hello"}
    assert instructions == "You are helpful."


def test_messages_to_responses_input_with_developer():
    """Test that developer messages are extracted as instructions."""
    messages = [
        {"role": "developer", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]

    input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(
        messages
    )

    assert len(input_items) == 1
    assert instructions == "Be concise."


def test_messages_to_responses_input_with_tool():
    """Test that tool messages are converted to function_call_output."""
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {"role": "tool", "content": "72°F", "tool_call_id": "call_123"},
    ]

    input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(
        messages
    )

    assert len(input_items) == 2
    assert input_items[1] == {
        "type": "function_call_output",
        "call_id": "call_123",
        "output": "72°F",
    }


def test_messages_to_responses_input_preserves_images():
    """An image block must survive the round trip, or OpenAI counts only the text.

    A 256x256 image is worth 255 tokens to OpenAI's counting API; dropping it
    turned a 268-token request into a 13-token one.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo=", "detail": "high"},
                },
            ],
        }
    ]

    input_items, instructions = OpenAICountTokensConfig.messages_to_responses_input(messages)

    assert instructions is None
    assert input_items == [
        {
            "role": "user",
            "content": (
                {"type": "input_text", "text": "What is in this image?"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBORw0KGgo=",
                    "detail": "high",
                },
            ),
        }
    ]


def test_messages_to_responses_input_image_without_detail_defaults_to_auto():
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}],
        }
    ]

    input_items, _ = OpenAICountTokensConfig.messages_to_responses_input(messages)

    assert input_items[0]["content"] == (
        {"type": "input_image", "image_url": "https://example.com/cat.png", "detail": "auto"},
    )


def test_messages_to_responses_input_bare_string_image_url_is_preserved():
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": "https://example.com/cat.png"}]}]

    input_items, _ = OpenAICountTokensConfig.messages_to_responses_input(messages)

    assert input_items[0]["content"] == (
        {"type": "input_image", "image_url": "https://example.com/cat.png", "detail": "auto"},
    )


def test_messages_to_responses_input_text_only_blocks_stay_a_joined_string():
    """Text-only content must keep collapsing to a string so existing counts do not shift."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        }
    ]

    input_items, _ = OpenAICountTokensConfig.messages_to_responses_input(messages)

    assert input_items == [{"role": "user", "content": "first\nsecond"}]


def test_messages_to_responses_input_drops_unmappable_blocks():
    """A block with no Responses API equivalent is skipped, never forwarded verbatim."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
            ],
        }
    ]

    input_items, _ = OpenAICountTokensConfig.messages_to_responses_input(messages)

    assert input_items[0]["content"] == (
        {"type": "input_text", "text": "hi"},
        {"type": "input_image", "image_url": "https://example.com/cat.png", "detail": "auto"},
    )


def test_validate_request_valid():
    """Test that valid requests pass validation."""
    config = OpenAICountTokensConfig()
    config.validate_request(model="gpt-4o", input="Hello")


def test_validate_request_missing_model():
    """Test that missing model raises ValueError."""
    config = OpenAICountTokensConfig()
    with pytest.raises(ValueError, match="model") as exc_info:
        config.validate_request(model="", input="Hello")
    e = exc_info.value
    assert "model" in str(e)


def test_validate_request_missing_input():
    """Test that missing input raises ValueError."""
    config = OpenAICountTokensConfig()
    with pytest.raises(ValueError, match="input") as exc_info:
        config.validate_request(model="gpt-4o", input="")
    e = exc_info.value
    assert "input" in str(e)


def test_get_endpoint_default():
    """Test default endpoint URL."""
    config = OpenAICountTokensConfig()
    assert (
        config.get_openai_count_tokens_endpoint()
        == "https://api.openai.com/v1/responses/input_tokens"
    )


def test_get_endpoint_custom_base():
    """Test custom API base URL."""
    config = OpenAICountTokensConfig()
    assert (
        config.get_openai_count_tokens_endpoint("https://custom.api.com/v1")
        == "https://custom.api.com/v1/responses/input_tokens"
    )


def test_get_required_headers():
    """Test required headers include Authorization."""
    config = OpenAICountTokensConfig()
    headers = config.get_required_headers("sk-test-key")

    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers["Content-Type"] == "application/json"
