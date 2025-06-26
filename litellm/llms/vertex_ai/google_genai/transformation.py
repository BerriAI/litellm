"""
Transformation for Calling Google models in their native format.
"""
from typing import Literal

from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig


class VertexAIGoogleGenAIConfig(GoogleGenAIConfig):
    """
    Configuration for calling Google models in their native format.
    """
    @property
    def custom_llm_provider(self) -> Literal["gemini", "vertex_ai"]:
        return "vertex_ai"
    