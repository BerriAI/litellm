"""
This is OpenAI compatible - no transformation is applied

"""

import types
from typing import Literal, Optional, Union

import litellm


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

    def is_fireworks_embedding_model(self, model: str):
        """
        helper to check if a model is a fireworks embedding model

        Fireworks embeddings does not support passing /accounts/fireworks in the model name so we need to know if it's a known embedding model
        """
        if (
            model in litellm.fireworks_ai_embedding_models
            or f"fireworks_ai/{model}" in litellm.fireworks_ai_embedding_models
        ):
            return True

        return False
