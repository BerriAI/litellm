"""
Support for OpenAI's `/v1/chat/completions` endpoint. 

Calls done in OpenAI/openai.py as TogetherAI is openai-compatible.

Docs: https://docs.together.ai/reference/completions-1
"""

from ..OpenAI.openai import OpenAIConfig


class TogetherAIConfig(OpenAIConfig):
    pass
