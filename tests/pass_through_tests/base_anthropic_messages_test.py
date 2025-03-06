from abc import ABC, abstractmethod

import anthropic


class BaseAnthropicMessagesTest(ABC):
    """
    Abstract base test class that enforces a common test across all test classes.
    """

    @property
    def client(self):
        return anthropic.Anthropic()

    def test_anthropic_basic_completion(self):
        print("making basic completion request to anthropic passthrough")
        response = self.client.messages.create(
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

    def test_anthropic_streaming(self):
        print("making streaming request to anthropic passthrough")
        collected_output = []

        with self.client.messages.stream(
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

    def test_anthropic_messages_with_thinking(self):
        print("making request to anthropic passthrough with thinking")
        response = self.client.messages.create(
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
