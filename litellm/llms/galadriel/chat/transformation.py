"""
Translate from OpenAI's `/v1/chat/completions` to Galadriel's `/v1/chat/completions`
"""

from ...openai_like.chat.handler import OpenAILikeChatConfig


class GaladrielChatConfig(OpenAILikeChatConfig):
    pass
