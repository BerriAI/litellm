"""
Transformation logic from OpenAI /v1/embeddings format to LM Studio's  `/v1/embeddings` format. 

Why separate file? Make it easy to see how transformation works

Docs - https://lmstudio.ai/docs/basics/server
"""

import types
from typing import List


class LmStudioEmbeddingConfig:
    """
    Reference: https://lmstudio.ai/docs/basics/server
    """

    def __init__(
        self,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self) -> List[str]:
        return []

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        return optional_params
