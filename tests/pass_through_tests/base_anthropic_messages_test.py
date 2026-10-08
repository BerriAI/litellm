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




    def test_anthropic_streaming_with_thinking(self):
        print("making streaming request to anthropic passthrough with thinking enabled")
        collected_thinking = []
        collected_response = []
        client = self.get_client()
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
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


