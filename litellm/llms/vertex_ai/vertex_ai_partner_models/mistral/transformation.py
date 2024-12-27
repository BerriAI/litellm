"""
Transformation from OpenAI's `/v1/chat/completions` to Vertex AI Mistral's `/chat/completions` format
"""

from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig


class VertexAIMistralConfig(OpenAILikeChatConfig):
    pass
