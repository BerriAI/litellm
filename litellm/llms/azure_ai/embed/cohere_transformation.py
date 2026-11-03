"""
Transformation logic from OpenAI /v1/embeddings format to Azure AI Cohere's /v1/embed.

Why separate file? Make it easy to see how transformation works

Convers
- Cohere request format

Docs - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
"""

from collections.abc import Mapping, Sequence
from typing import Final
from urllib.parse import urlsplit, urlunsplit

from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo, get_azure_ai_auth_headers
from litellm.llms.cohere.embed.transformation import CohereEmbeddingConfig
from litellm.types.llms.azure_ai import ImageEmbeddingInput, ImageEmbeddingRequest
from litellm.types.llms.openai import AllMessageValues, EmbeddingCreateParams
from litellm.types.utils import EmbeddingResponse, Usage
from litellm.utils import is_base64_encoded


class AzureAICohereNativeEmbeddingConfig(CohereEmbeddingConfig):
    @staticmethod
    def supports_model(model: str, api_base: str | None = None) -> bool:
        path: Final = urlsplit(api_base or "").path.rstrip("/")
        name: Final = model.lower().removeprefix("azure_ai/")
        return path.endswith(("/providers/cohere", "/providers/cohere/v2", "/providers/cohere/v2/embed")) or (
            name.startswith("cohere-embed-") and not name.startswith(("cohere-embed-v3", "cohere-embed-v4"))
        )

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        resolved_base: Final = AzureFoundryModelInfo.get_api_base(api_base)
        if resolved_base is None:
            raise ValueError("Azure AI API base is required for native Cohere embeddings")
        parts: Final = urlsplit(resolved_base)
        path: Final = parts.path.rstrip("/")
        endpoint: Final = (
            path
            if path.endswith("/providers/cohere/v2/embed")
            else f"{path}/embed"
            if path.endswith("/providers/cohere/v2")
            else f"{path}/v2/embed"
            if path.endswith("/providers/cohere")
            else f"{path.removesuffix('/models')}/providers/cohere/v2/embed"
        )
        return urlunsplit((parts.scheme, parts.netloc, endpoint, parts.query, parts.fragment))

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseEmbeddingConfig requires a mutable header dictionary
        return {
            "Content-Type": "application/json",
            **get_azure_ai_auth_headers(
                api_key=AzureFoundryModelInfo.get_api_key(api_key), litellm_params=litellm_params
            ),
            **headers,
        }


class AzureAICohereConfig:
    def __init__(self) -> None:
        pass

    def _map_azure_model_group(self, model: str) -> str:
        if model == "offer-cohere-embed-multili-paygo":
            return "Cohere-embed-v3-multilingual"
        elif model == "offer-cohere-embed-english-paygo":
            return "Cohere-embed-v3-english"

        return model

    def _transform_request_image_embeddings(self, input: list[str], optional_params: dict) -> ImageEmbeddingRequest:
        """
        Assume all str in list is base64 encoded string
        """
        image_input: Final[list[ImageEmbeddingInput]] = []
        for i in input:
            embedding_input = ImageEmbeddingInput(image=i)
            image_input.append(embedding_input)
        return ImageEmbeddingRequest(input=image_input, **optional_params)

    def _transform_request(
        self, input: list[str], optional_params: dict, model: str
    ) -> tuple[ImageEmbeddingRequest, EmbeddingCreateParams, list[int]]:
        """
        Return the list of input to `/image/embeddings`, `/v1/embeddings`, list of image_embedding_idx for recombination
        """
        image_embeddings: Final[list[str]] = []
        image_embedding_idx: Final[list[int]] = []
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
        filtered_input: Final = [item for idx, item in enumerate(input) if idx not in image_embedding_idx]

        v1_embeddings_request: Final = EmbeddingCreateParams(input=filtered_input, model=model, **optional_params)
        image_embeddings_request: Final = self._transform_request_image_embeddings(
            input=image_embeddings, optional_params=optional_params
        )

        return image_embeddings_request, v1_embeddings_request, image_embedding_idx

    def transform_request(
        self,
        input: list[str],  # mutable-ok: mirrors override contract
        optional_params: dict[str, object],  # mutable-ok: mirrors override contract
        model: str,
    ) -> tuple[ImageEmbeddingRequest, EmbeddingCreateParams, list[int]]:  # mutable-ok: mirrors override contract
        return self._transform_request(input, optional_params, model)

    def _transform_response(self, response: EmbeddingResponse) -> EmbeddingResponse:
        additional_headers: Final[dict | None] = response.hidden_params.get("additional_headers")
        if additional_headers:
            # CALCULATE USAGE
            input_tokens: Final[str | None] = additional_headers.get("llm_provider-num_tokens")
            if input_tokens:
                if response.usage:
                    response.usage.prompt_tokens = int(input_tokens)
                else:
                    response.usage = Usage(prompt_tokens=int(input_tokens))

            # SET MODEL
            base_model: Final[str | None] = additional_headers.get("llm_provider-azureml-model-group")
            if base_model:
                response.model = self._map_azure_model_group(base_model)

        return response

    def transform_response(
        self,
        response: EmbeddingResponse,
    ) -> EmbeddingResponse:
        return self._transform_response(response)
