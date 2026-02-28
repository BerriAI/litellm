"""
Translate from OpenAI's `/v1/chat/completions` to Friendliai's `/v1/chat/completions`
"""

from ...openai_like.chat.handler import OpenAILikeChatConfig


class FriendliaiChatConfig(OpenAILikeChatConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "max_retries",
            "extra_headers",
        ]
