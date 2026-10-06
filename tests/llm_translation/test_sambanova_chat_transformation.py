"""
Unit tests for SambaNova chat message transformation
"""

from litellm.llms.sambanova.chat import SambanovaConfig


class TestSambanovaContentListHandling:
    """
    Test that SambaNova properly transforms content lists to strings
    """

    def test_content_list_to_string_transformation(self):
        """
        Test content list with text objects is converted to string.

        SambaNova API doesn't support content as a list - only string content.
        """
        config = SambanovaConfig()

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, how are you?"}],
            }
        ]

        transformed_messages = config._transform_messages(
            messages=messages, model="sambanova/gpt-oss-120b", is_async=False
        )

        assert len(transformed_messages) == 1
        assert transformed_messages[0]["role"] == "user"
        assert isinstance(transformed_messages[0]["content"], str)
        assert transformed_messages[0]["content"] == "Hello, how are you?"

    def test_content_list_multiple_text_blocks(self):
        """
        Test content list with multiple text blocks is converted to concatenated string.
        """
        config = SambanovaConfig()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, "},
                    {"type": "text", "text": "how are you?"},
                ],
            }
        ]

        transformed_messages = config._transform_messages(
            messages=messages, model="sambanova/gpt-oss-120b", is_async=False
        )

        assert transformed_messages[0]["content"] == "Hello, how are you?"

    def test_string_content_unchanged(self):
        """
        Test that string content is passed through unchanged.
        """
        config = SambanovaConfig()

        messages = [{"role": "user", "content": "Hello, how are you?"}]

        transformed_messages = config._transform_messages(
            messages=messages, model="sambanova/gpt-oss-120b", is_async=False
        )

        assert transformed_messages[0]["content"] == "Hello, how are you?"

    def test_multiple_messages_transformation(self):
        """
        Test transformation of multiple messages with mixed content types.
        """
        config = SambanovaConfig()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the weather?"}],
            },
            {"role": "assistant", "content": "I need your location."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I'm in "},
                    {"type": "text", "text": "San Francisco"},
                ],
            },
        ]

        transformed_messages = config._transform_messages(
            messages=messages, model="sambanova/gpt-oss-120b", is_async=False
        )

        assert len(transformed_messages) == 4
        assert transformed_messages[0]["content"] == "You are a helpful assistant."
        assert transformed_messages[1]["content"] == "What is the weather?"
        assert transformed_messages[2]["content"] == "I need your location."
        assert transformed_messages[3]["content"] == "I'm in San Francisco"

    def test_content_list_with_image_url_is_preserved(self):
        """
        A content list carrying a non-text part (image_url) must stay a list so
        the image reaches the SambaNova API, instead of being flattened to a
        text-only string that silently drops the image.
        """
        config = SambanovaConfig()

        image_part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
        }
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colors are in this image?"},
                    image_part,
                ],
            }
        ]

        transformed_messages = config._transform_messages(
            messages=messages, model="sambanova/gpt-oss-120b", is_async=False
        )

        content = transformed_messages[0]["content"]
        assert isinstance(content, list)
        assert image_part in content
        assert any(part.get("type") == "image_url" for part in content)
