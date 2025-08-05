"""
This is OpenAI compatible - no transformation is applied

"""


class SambaNovaEmbeddingConfig:
    def get_supported_openai_params(self, model: str):
        """
        Non additional params supported, placeholder method for future supported params
        https://docs.sambanova.ai/cloud/api-reference/endpoints/embeddings-api
        """
        return []

    def map_openai_params(
        self, non_default_params: dict,
        optional_params: dict, 
        model: str, 
        drop_params: bool,
    ):
        """
        No transformation is applied - SambaNova is openai compatible
        """
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params
