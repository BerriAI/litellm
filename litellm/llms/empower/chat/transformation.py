"""
Translate from OpenAI's `/v1/chat/completions` to Empower's `/v1/chat/completions`
"""

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class EmpowerChatConfig(OpenAILikeChatConfig):
    pass
