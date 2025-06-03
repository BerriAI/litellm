"""
Handler for CentML's chat completion endpoint.

Since CentML is OpenAI-compatible, this is a minimal wrapper that uses the OpenAI implementation.
"""

from litellm.llms.openai.chat.handler import OpenAIChatCompletion


class CentmlChatCompletion(OpenAIChatCompletion):
    """
    CentML's chat completion endpoint is OpenAI-compatible, so we can use the OpenAI implementation.
    The only difference is in the configuration and parameter handling, which is done in transformation.py.
    """
    pass 