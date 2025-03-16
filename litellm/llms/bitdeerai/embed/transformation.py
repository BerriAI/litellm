"""
Translates from OpenAI's `/v1/embeddings` to BitdeerAI's `/v1/embeddings`
"""


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
            drop_params: dict = None,
    ) -> dict:
        return optional_params