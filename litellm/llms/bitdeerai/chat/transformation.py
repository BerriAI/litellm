"""
Translates from OpenAI's `/v1/chat/completions` to BitdeerAI's `/v1/chat/completions`
"""
from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class BitdeerAIChatConfig(OpenAIGPTConfig):
    pass