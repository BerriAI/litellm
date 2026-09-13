"""
Unit tests for SambaNova chat message transformation
"""

import pytest

from litellm.llms.sambanova.chat import SambanovaConfig


class TestSambanovaNonTextContentParts:
    """
    Content lists that carry a non-text part must keep their list form.
    """

    def test_content_list_with_image_is_preserved(self):
        """
        A content list carrying an `image_url` part must NOT be flattened.

        SambaNova's API accepts the OpenAI content-list form with `image_url`, and its
        vision models read it. Flattening dropped the image while still returning HTTP
        200, so the model answered as if nothing had been attached.
        """
        config = SambanovaConfig()

        image_part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
        }
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What colour is this?"}, image_part],
            }
        ]

        transformed_messages = config._transform_messages(
            messages=messages, model="sambanova/gemma-4-31B-it", is_async=False
        )

        content = transformed_messages[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "What colour is this?"}
        assert content[1] == image_part

    def test_string_content_is_left_alone(self):
        """A message whose content is already a string passes through untouched."""
        config = SambanovaConfig()

        messages = [{"role": "user", "content": "Hello"}]

        transformed_messages = config._transform_messages(
            messages=messages, model="sambanova/gemma-4-31B-it", is_async=False
        )

        assert transformed_messages[0]["content"] == "Hello"

    def test_text_only_messages_are_still_flattened_alongside_image_messages(self):
        """
        Mixed conversation: text-only lists are flattened as before, and only the
        message that carries the image keeps its list form.
        """
        config = SambanovaConfig()

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ],
            },
        ]

        transformed_messages = config._transform_messages(
            messages=messages, model="sambanova/gemma-4-31B-it", is_async=False
        )

        assert transformed_messages[0]["content"] == "Hello"
        assert isinstance(transformed_messages[1]["content"], list)

    @pytest.mark.asyncio
    async def test_async_transform_preserves_image_content(self):
        """The async path must behave like the sync one."""
        config = SambanovaConfig()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ],
            }
        ]

        transformed_messages = await config._transform_messages(
            messages=messages, model="sambanova/gemma-4-31B-it", is_async=True
        )

        assert isinstance(transformed_messages[0]["content"], list)
