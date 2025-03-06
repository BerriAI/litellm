from base_anthropic_messages_test import BaseAnthropicMessagesTest
import anthropic


class TestAnthropicPassthroughBasic(BaseAnthropicMessagesTest):

    @property
    def client(self):
        return anthropic.Anthropic(
            base_url="http://0.0.0.0:4000/anthropic",
            api_key="sk-1234",
        )


class TestAnthropicMessagesEndpoint(BaseAnthropicMessagesTest):
    @property
    def client(self):
        return anthropic.Anthropic(
            base_url="http://0.0.0.0:4000",
            api_key="sk-1234",
        )

    def test_anthropic_messages_to_wildcard_model(self):
        response = self.client.messages.create(
            model="anthropic/claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hello, world!"}],
            max_tokens=100,
        )
        print(response)
