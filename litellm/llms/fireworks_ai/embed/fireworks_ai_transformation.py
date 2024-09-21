"""
This is OpenAI compatible - no transformation is applied

"""

import types
from typing import Literal, Optional, Union


class FireworksAIEmbeddingConfig:
    def get_supported_openai_params(self, model: str):
        """
        dimensions Only supported in nomic-ai/nomic-embed-text-v1.5 and later models.

        https://docs.fireworks.ai/api-reference/creates-an-embedding-vector-representing-the-input-text
        """
        if "nomic-ai" in model:
            return ["dimensions"]
        return []

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict, model: str
    ):
        """
        No transformation is applied - fireworks ai is openai compatible
        """
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params
