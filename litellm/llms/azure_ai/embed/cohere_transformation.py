"""
Transformation logic from OpenAI /v1/embeddings format to Azure AI Cohere's /v1/embed. 

Why separate file? Make it easy to see how transformation works

Convers
- Cohere request format

Docs - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
"""

from typing import Optional

from litellm.types.utils import EmbeddingResponse, Usage


class AzureAICohereConfig:
    def __init__(self) -> None:
        pass

    def _map_azure_model_group(self, model: str) -> str:
        if "model=offer-cohere-embed-multili-paygo":
            return "Cohere-embed-v3-multilingual"
        elif "model=offer-cohere-embed-english-paygo":
            return "Cohere-embed-v3-english"

        return model

    def _transform_response(self, response: EmbeddingResponse) -> EmbeddingResponse:
        additional_headers: Optional[dict] = response._hidden_params.get(
            "additional_headers"
        )
        if additional_headers:
            # CALCULATE USAGE
            input_tokens: Optional[str] = additional_headers.get(
                "llm_provider-num_tokens"
            )
            if input_tokens:
                if response.usage:
                    response.usage.prompt_tokens = int(input_tokens)
                else:
                    response.usage = Usage(prompt_tokens=int(input_tokens))

            # SET MODEL
            base_model: Optional[str] = additional_headers.get(
                "llm_provider-azureml-model-group"
            )
            if base_model:
                response.model = self._map_azure_model_group(base_model)

        return response
