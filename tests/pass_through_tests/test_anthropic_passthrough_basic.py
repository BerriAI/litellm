import os
from base_anthropic_messages_test import BaseAnthropicMessagesTest
import anthropic


class TestAnthropicPassthroughBasic(BaseAnthropicMessagesTest):

    def get_client(self):
        return anthropic.Anthropic(
            base_url="http://0.0.0.0:4000/anthropic",
            api_key=os.environ["LITELLM_MASTER_KEY"],
        )


class TestAnthropicMessagesEndpoint(BaseAnthropicMessagesTest):
    def get_client(self):
        return anthropic.Anthropic(
            base_url="http://0.0.0.0:4000",
            api_key=os.environ["LITELLM_MASTER_KEY"],
        )

