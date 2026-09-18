from base_anthropic_messages_test import BaseAnthropicMessagesTest
import anthropic


class TestAnthropicPassthroughBasic(BaseAnthropicMessagesTest):

    def get_client(self):
        return anthropic.Anthropic(
            base_url="http://0.0.0.0:4000/anthropic",
            api_key="sk-1234",
        )


class TestAnthropicMessagesEndpoint(BaseAnthropicMessagesTest):
    def get_client(self):
        return anthropic.Anthropic(
            base_url="http://0.0.0.0:4000",
            api_key="sk-1234",
        )

    def test_anthropic_messages_to_wildcard_model(self):
        client = self.get_client()
        response = client.messages.create(
            model="anthropic/claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hello, world!"}],
            max_tokens=100,
        )
        print(response)
