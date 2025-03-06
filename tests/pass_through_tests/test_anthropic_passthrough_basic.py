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
