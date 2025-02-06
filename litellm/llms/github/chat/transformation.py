"""
Translate from OpenAI's `/v1/chat/completions` to Github's `/v1/chat/completions`
"""

from ...openai_like.chat.handler import OpenAILikeChatConfig


class GithubChatConfig(OpenAILikeChatConfig):
    pass
