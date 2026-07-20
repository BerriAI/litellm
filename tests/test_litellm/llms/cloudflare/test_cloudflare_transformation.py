"""
Regression tests for Cloudflare content-part message normalization.
https://github.com/BerriAI/litellm/issues/33984
"""
from litellm.llms.cloudflare.chat.transformation import CloudflareChatConfig


def test_string_content_passthrough():
    """Plain string content should pass through unchanged."""
    config = CloudflareChatConfig()
    messages = [{"role": "user", "content": "Hello"}]
    result = config._transform_messages(messages, model="cloudflare/@cf/meta/llama-3.2-1b-instruct")
    assert result[0]["content"] == "Hello"


def test_content_part_array_flattened_to_string():
    """OpenAI content-part array should be flattened to a plain string."""
    config = CloudflareChatConfig()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "More context"},
            ],
        }
    ]
    result = config._transform_messages(messages, model="cloudflare/@cf/meta/llama-3.2-1b-instruct")
    assert result[0]["content"] == "Hello\n\nMore context"


def test_non_text_parts_ignored():
    """Only text-type parts should be included in the joined string."""
    config = CloudflareChatConfig()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                {"type": "text", "text": "Describe the image above"},
            ],
        }
    ]
    result = config._transform_messages(messages, model="cloudflare/@cf/meta/llama-3.2-1b-instruct")
    assert result[0]["content"] == "Hello\n\nDescribe the image above"


def test_multiple_messages_mixed():
    """Multiple messages with mixed content formats all transform correctly."""
    config = CloudflareChatConfig()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "Environment details here"},
            ],
        },
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]
    result = config._transform_messages(messages, model="cloudflare/@cf/meta/llama-3.2-1b-instruct")
    assert result[0]["content"] == "You are a helpful assistant."
    assert result[1]["content"] == "Hello\n\nEnvironment details here"
    assert result[2]["content"] == "Hi! How can I help?"
