"""
Legacy /v1/embedding transformation logic for Bedrock Cohere.
"""

from collections.abc import Sized
from typing import Final, Protocol

import httpx

from litellm import COHERE_DEFAULT_EMBEDDING_INPUT_TYPE
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import header_value
from litellm.types.llms.bedrock import (
    CohereEmbeddingRequest,
    CohereEmbeddingRequestWithModel,
)
from litellm.types.utils import EmbeddingResponse, PromptTokensDetailsWrapper, Usage
from litellm.utils import is_base64_encoded


class _SupportsEncode(Protocol):
    """Tokenizer handle: the embedding usage path only encodes text to measure its token length."""

    def encode(self, text: str, /) -> Sized: ...


def _bedrock_billed_input_tokens(response: httpx.Response) -> int | None:
    count: Final = header_value(response.headers, "x-amzn-bedrock-input-token-count")
    return int(count) if count is not None and count.isdigit() else None


class CohereEmbeddingConfig:
    """
    Reference: https://docs.cohere.com/v2/reference/embed
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self) -> list[str]:
        return ["encoding_format"]

    def map_openai_params(self, non_default_params: dict, optional_params: dict) -> dict:
        for k, v in non_default_params.items():
            if k == "encoding_format":
                optional_params["embedding_types"] = v
        return optional_params

    def _is_v3_model(self, model: str) -> bool:
        return "3" in model

    def _transform_request(
        self, model: str, input: list[str], inference_params: dict
    ) -> CohereEmbeddingRequestWithModel:
        is_encoded = False
        for input_str in input:
            is_encoded = is_base64_encoded(input_str)

        if is_encoded:  # check if string is b64 encoded image or not
            transformed_request = CohereEmbeddingRequestWithModel(
                model=model,
                images=input,
                input_type="image",
            )
        else:
            transformed_request = CohereEmbeddingRequestWithModel(
                model=model,
                texts=input,
                input_type=COHERE_DEFAULT_EMBEDDING_INPUT_TYPE,
            )

        for k, v in inference_params.items():
            transformed_request[k] = v

        return transformed_request

    def transform_request(
        self,
        model: str,
        input: list[str],  # mutable-ok: mirrors override contract
        inference_params: dict[str, object],  # mutable-ok: mirrors override contract
    ) -> CohereEmbeddingRequestWithModel:
        return self._transform_request(model, input, inference_params)

    def _calculate_usage(
        self, input: list[str], encoding: _SupportsEncode, meta: dict, billed_input_tokens: int | None = None
    ) -> Usage:
        text_tokens: Final[int | None] = meta.get("billed_units", {}).get("input_tokens")

        image_tokens: Final[int | None] = meta.get("billed_units", {}).get("images")

        if image_tokens is None and text_tokens is None:
            counted_tokens: Final = (
                billed_input_tokens
                if billed_input_tokens is not None
                else sum(len(encoding.encode(text)) for text in input)
            )
            return Usage(prompt_tokens=counted_tokens, completion_tokens=0, total_tokens=counted_tokens)

        input_tokens: Final = (image_tokens or 0) + (text_tokens or 0)
        return Usage(
            prompt_tokens=input_tokens,
            completion_tokens=0,
            total_tokens=input_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                image_tokens=image_tokens,
                text_tokens=text_tokens,
            ),
        )

    def _transform_response(
        self,
        response: httpx.Response,
        api_key: str | None,
        logging_obj: LiteLLMLoggingObj,
        data: dict | CohereEmbeddingRequest,
        model_response: EmbeddingResponse,
        model: str,
        encoding: _SupportsEncode,
        input: list,
    ) -> EmbeddingResponse:
        response_json: Final = response.json()
        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=response_json,
        )
        return self._populate_embedding_response(
            response_json=response_json,
            model_response=model_response,
            model=model,
            encoding=encoding,
            input=input,
            billed_input_tokens=_bedrock_billed_input_tokens(response) if "images" in data else None,
        )

    def transform_response(
        self,
        response: httpx.Response,
        api_key: str | None,
        logging_obj: LiteLLMLoggingObj,
        data: dict[str, object] | CohereEmbeddingRequest,  # mutable-ok: mirrors override contract
        model_response: EmbeddingResponse,
        model: str,
        encoding: _SupportsEncode,
        input: list[str],  # mutable-ok: mirrors override contract
    ) -> EmbeddingResponse:
        return self._transform_response(response, api_key, logging_obj, data, model_response, model, encoding, input)

    def _populate_embedding_response(
        self,
        response_json: dict,
        model_response: EmbeddingResponse,
        model: str,
        encoding: _SupportsEncode,
        input: list,
        billed_input_tokens: int | None = None,
    ) -> EmbeddingResponse:
        """
        Parse a Cohere embed response body into an OpenAI-style EmbeddingResponse.

        Split out from `_transform_response` so callers that already log
        `post_call` themselves (e.g. SageMaker's embedding handler) can reuse
        the parsing without triggering a second `post_call`.

        Response shape:
            {
                'object': "list",
                'data': [...],
                'model',
                'usage',
            }
        """
        embeddings: Final = response_json["embeddings"]
        output_data: Final = []
        is_embeddings_by_type = response_json.get("response_type") == "embeddings_by_type"

        if isinstance(embeddings, dict):
            is_embeddings_by_type = True

        if is_embeddings_by_type:
            for embedding_type in embeddings:
                for idx, embedding in enumerate(embeddings[embedding_type]):
                    output_data.append(
                        {
                            "object": "embedding",
                            "index": idx,
                            "embedding": embedding,
                            "type": embedding_type,
                        }
                    )
        else:
            for idx, embedding in enumerate(embeddings):
                output_data.append({"object": "embedding", "index": idx, "embedding": embedding})
        model_response.object = "list"
        model_response.data = output_data
        model_response.model = model

        setattr(
            model_response,
            "usage",
            self._calculate_usage(input, encoding, response_json.get("meta", {}), billed_input_tokens),
        )

        return model_response

    def populate_embedding_response(
        self,
        response_json: dict[str, object],  # mutable-ok: mirrors override contract
        model_response: EmbeddingResponse,
        model: str,
        encoding: _SupportsEncode,
        input: list[str],  # mutable-ok: mirrors override contract
    ) -> EmbeddingResponse:
        return self._populate_embedding_response(response_json, model_response, model, encoding, input)
