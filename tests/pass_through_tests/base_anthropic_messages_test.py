from abc import ABC, abstractmethod

import anthropic
import pytest
from anthropic._exceptions import OverloadedError


class BaseAnthropicMessagesTest(ABC):
    """
    Abstract base test class that enforces a common test across all test classes.
    """

    @abstractmethod
    def get_client(self):
        return anthropic.Anthropic()

    @pytest.mark.flaky(retries=3, delay=2)
    def test_anthropic_basic_completion(self):
        print("making basic completion request to anthropic passthrough")
        client = self.get_client()
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Say 'hello test' and nothing else"}],
                extra_body={
                    "litellm_metadata": {
                        "tags": ["test-tag-1", "test-tag-2"],
                    }
                },
            )
            print(response)
            assert response is not None
            assert hasattr(response, 'content')
            assert len(response.content) > 0
            # Check if the first content block is a text block
            first_block = response.content[0]
            from anthropic.types import TextBlock
            if isinstance(first_block, TextBlock):
                assert first_block.text is not None
                assert "hello test" in first_block.text.lower()
        except OverloadedError as e:
            # Anthropic API is overloaded - this is expected and acceptable for CI
            print(f"Anthropic API overloaded (expected): {e}")
            pytest.skip("Anthropic API is temporarily overloaded - skipping test")

    @pytest.mark.flaky(retries=3, delay=2)
    def test_anthropic_streaming(self):
        print("making streaming request to anthropic passthrough")
        try:
            collected_output = []
            client = self.get_client()
            with client.messages.stream(
                max_tokens=10,
                messages=[
                    {"role": "user", "content": "Say 'hello stream test' and nothing else"}
                ],
                model="claude-3-5-sonnet-20241022",
                extra_body={
                    "litellm_metadata": {
                        "tags": ["test-tag-stream-1", "test-tag-stream-2"],
                    }
                },
            ) as stream:
                for text in stream.text_stream:
                    collected_output.append(text)

            full_response = "".join(collected_output)
            print(full_response)
        except OverloadedError as e:
            # Anthropic API is overloaded - this is expected and acceptable for CI
            print(f"Anthropic API overloaded (expected): {e}")
            pytest.skip("Anthropic API is temporarily overloaded - skipping test")

    def test_anthropic_messages_with_thinking(self):
        print("making request to anthropic passthrough with thinking")
        client = self.get_client()
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=20000,
            thinking={"type": "enabled", "budget_tokens": 16000},
            messages=[
                {"role": "user", "content": "Just pinging with thinking enabled"}
            ],
        )

        print(response)

        # Verify the first content block is a thinking block
        first_block = response.content[0]
        from anthropic.types import ThinkingBlock
        if isinstance(first_block, ThinkingBlock):
            response_thinking = first_block.thinking
            assert response_thinking is not None
            assert len(response_thinking) > 0

    def test_anthropic_streaming_with_thinking(self):
        print("making streaming request to anthropic passthrough with thinking enabled")
        collected_thinking = []
        collected_response = []
        client = self.get_client()
        with client.messages.stream(
            model="claude-3-7-sonnet-20250219",
            max_tokens=20000,
            thinking={"type": "enabled", "budget_tokens": 16000},
            messages=[
                {"role": "user", "content": "Just pinging with thinking enabled"}
            ],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        collected_thinking.append(event.delta.thinking)
                    elif event.delta.type == "text_delta":
                        collected_response.append(event.delta.text)

        full_thinking = "".join(collected_thinking)
        full_response = "".join(collected_response)

        print(
            f"Thinking Response: {full_thinking[:100]}..."
        )  # Print first 100 chars of thinking
        print(f"Response: {full_response}")

        # Verify we received thinking content
        assert len(collected_thinking) > 0
        assert len(full_thinking) > 0

        # Verify we also received a response
        assert len(collected_response) > 0
        assert len(full_response) > 0

    def test_bad_request_error_handling_streaming(self):
        print("making request to anthropic passthrough with bad request")
        try:
            client = self.get_client()
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                stream=True,
                messages=[{"role": "user", "content": "hi"}],
                temperature=3.0,  # Invalid temperature (> 2.0)
            )
            print(response)
            assert pytest.fail("Expected BadRequestError")
        except anthropic.BadRequestError as e:
            print("Got BadRequestError from anthropic, e=", e)
            print(e.__cause__)
            print(e.status_code)
            print(e.response)
        except Exception as e:
            pytest.fail(f"Got unexpected exception: {e}")

    def test_bad_request_error_handling_non_streaming(self):
        print("making request to anthropic passthrough with bad request")
        try:
            client = self.get_client()
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
                temperature=3.0,  # Invalid temperature (> 2.0)
            )
            print(response)
            assert pytest.fail("Expected BadRequestError")
        except anthropic.BadRequestError as e:
            print("Got BadRequestError from anthropic, e=", e)
            print(e.__cause__)
            print(e.status_code)
            print(e.response)
        except Exception as e:
            pytest.fail(f"Got unexpected exception: {e}")
