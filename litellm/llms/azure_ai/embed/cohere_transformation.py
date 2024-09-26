"""
Transformation logic from OpenAI /v1/embeddings format to Azure AI Cohere's /v1/embed. 

Why separate file? Make it easy to see how transformation works

Convers
- Cohere request format

Docs - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
"""

from typing import List, Optional, Tuple, Union

from litellm.types.llms.azure_ai import ImageEmbeddingInput, ImageEmbeddingRequest
from litellm.types.llms.openai import EmbeddingCreateParams
from litellm.types.utils import Embedding, EmbeddingResponse, Usage
from litellm.utils import is_base64_encoded


class AzureAICohereConfig:
    def __init__(self) -> None:
        pass

    def _map_azure_model_group(self, model: str) -> str:
        if "model=offer-cohere-embed-multili-paygo":
            return "Cohere-embed-v3-multilingual"
        elif "model=offer-cohere-embed-english-paygo":
            return "Cohere-embed-v3-english"

        return model

    def _transform_request_image_embeddings(
        self, input: List[str], optional_params: dict
    ) -> ImageEmbeddingRequest:
        """
        Assume all str in list is base64 encoded string
        """
        image_input: List[ImageEmbeddingInput] = []
        for i in input:
            embedding_input = ImageEmbeddingInput(image=i)
            image_input.append(embedding_input)
        return ImageEmbeddingRequest(input=image_input, **optional_params)

    def _transform_request(
        self, input: List[str], optional_params: dict, model: str
    ) -> Tuple[ImageEmbeddingRequest, EmbeddingCreateParams, List[int]]:
        """
        Return the list of input to `/image/embeddings`, `/v1/embeddings`, list of image_embedding_idx for recombination
        """
        image_embeddings: List[str] = []
        image_embedding_idx: List[int] = []
        for idx, i in enumerate(input):
            """
            - is base64 -> route to image embeddings
            - is ImageEmbeddingInput -> route to image embeddings
            - else -> route to `/v1/embeddings`
            """
            if is_base64_encoded(i):
                image_embeddings.append(i)
                image_embedding_idx.append(idx)

        ## REMOVE IMAGE EMBEDDINGS FROM input list
        filtered_input = [
            item for idx, item in enumerate(input) if idx not in image_embedding_idx
        ]

        v1_embeddings_request = EmbeddingCreateParams(
            input=filtered_input, model=model, **optional_params
        )
        image_embeddings_request = self._transform_request_image_embeddings(
            input=image_embeddings, optional_params=optional_params
        )

        return image_embeddings_request, v1_embeddings_request, image_embedding_idx

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
