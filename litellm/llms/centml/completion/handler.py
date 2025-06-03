"""
Handler for CentML's text completion endpoint.

Since CentML is OpenAI-compatible, this is a minimal wrapper that uses the OpenAI implementation.
"""

from litellm.llms.openai.completion.handler import OpenAITextCompletion


class CentmlTextCompletion(OpenAITextCompletion):
    """
    CentML's text completion endpoint is OpenAI-compatible, so we can use the OpenAI implementation.
    The only difference is in the configuration and parameter handling, which is done in transformation.py.
    """
    pass 