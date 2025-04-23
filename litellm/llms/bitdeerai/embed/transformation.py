"""
Translates from OpenAI's `/v1/embeddings` to BitdeerAI's `/v1/embeddings`

from typing import Optional

class BitdeerAIEmbeddingConfig:
    @classmethod
    def get_supported_openai_params(cls) -> list:
        return []

    @classmethod
    def map_openai_params(
            cls,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: Optional[dict] = None,
    ) -> dict:
        return optional_params
"""