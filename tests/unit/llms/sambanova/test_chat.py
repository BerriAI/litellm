"""
Unit tests for SambaNova chat message transformation
"""

import asyncio
import importlib

import pytest

import litellm
from litellm.llms.sambanova.chat import SambanovaConfig
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


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

        transformed_messages = config.transform_messages(
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

        transformed_messages = config.transform_messages(
            messages=messages, model="sambanova/gpt-oss-120b", is_async=False
        )

        assert transformed_messages[0]["content"] == "Hello, how are you?"

    def test_string_content_unchanged(self):
        """
        Test that string content is passed through unchanged.
        """
        config = SambanovaConfig()

        messages = [{"role": "user", "content": "Hello, how are you?"}]

        transformed_messages = config.transform_messages(
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

        transformed_messages = config.transform_messages(
            messages=messages, model="sambanova/gpt-oss-120b", is_async=False
        )

        assert len(transformed_messages) == 4
        assert transformed_messages[0]["content"] == "You are a helpful assistant."
        assert transformed_messages[1]["content"] == "What is the weather?"
        assert transformed_messages[2]["content"] == "I need your location."
        assert transformed_messages[3]["content"] == "I'm in San Francisco"


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


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown(event_loop):
    import litellm

    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}
