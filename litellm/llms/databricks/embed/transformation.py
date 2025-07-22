"""
Translates from OpenAI's `/v1/embeddings` to Databricks' `/embeddings`
"""

import types
from typing import Optional


class DatabricksEmbeddingConfig:
    """
    Reference: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-models/api-reference#--embedding-task
    """

    instruction: Optional[
        str
    ] = None  # An optional instruction to pass to the embedding model. BGE Authors recommend 'Represent this sentence for searching relevant passages:' for retrieval queries

    def __init__(self, instruction: Optional[str] = None) -> None:
        locals_ = locals().copy()
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

    def get_supported_openai_params(
        self,
    ):  # no optional openai embedding params supported
        return []

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        return optional_params
