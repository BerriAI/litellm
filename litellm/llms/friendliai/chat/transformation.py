"""
Translate from OpenAI's `/v1/chat/completions` to Friendliai's `/v1/chat/completions`
"""

from ...openai_like.chat.handler import OpenAILikeChatConfig


class FriendliaiChatConfig(OpenAILikeChatConfig):
    pass
