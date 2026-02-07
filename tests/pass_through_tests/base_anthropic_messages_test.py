from abc import ABC, abstractmethod

import anthropic
import pytest


class BaseAnthropicMessagesTest(ABC):
    """
    Abstract base test class that enforces a common test across all test classes.
    """

    @abstractmethod
    def get_client(self):
        return anthropic.Anthropic()

    def test_anthropic_basic_completion(self):
        print("making basic completion request to anthropic passthrough")
        client = self.get_client()
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say 'hello test' and nothing else"}],
            extra_body={
                "litellm_metadata": {
                    "tags": ["test-tag-1", "test-tag-2"],
                }
            },
        )
        print(response)

    def test_anthropic_streaming(self):
        print("making streaming request to anthropic passthrough")
        collected_output = []
        client = self.get_client()
        with client.messages.stream(
            max_tokens=10,
            messages=[
                {"role": "user", "content": "Say 'hello stream test' and nothing else"}
            ],
            model="claude-sonnet-4-5-20250929",
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
        response_thinking = response.content[0].thinking
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
                model="claude-sonnet-4-5-20250929",
                max_tokens=10,
                stream=True,
                messages=["hi"],
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
                model="claude-sonnet-4-5-20250929",
                max_tokens=10,
                messages=["hi"],
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
