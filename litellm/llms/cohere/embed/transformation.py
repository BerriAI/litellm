"""
Transformation logic from OpenAI /v1/embeddings format to Cohere's /v1/embed format.

Why separate file? Make it easy to see how transformation works

Convers
- v3 embedding models
- v2 embedding models

Docs - https://docs.cohere.com/v2/reference/embed
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from enum import Enum
from itertools import chain, starmap
from typing import Final, Protocol, TypeAlias, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

import litellm
from litellm import COHERE_DEFAULT_EMBEDDING_INPUT_TYPE
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm import BaseEmbeddingConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.bedrock import (
    CohereEmbeddingRequest,
    CohereEmbeddingRequestWithModel,
)
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, PromptTokensDetailsWrapper, Usage
from litellm.utils import is_base64_encoded

from ..common_utils import CohereError

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_DECODED_JSON: Final = TypeAdapter(object)


class _SupportsEncode(Protocol):
    """Tokenizer handle: the embedding usage path only encodes text to measure its token length."""

    def encode_ordinary(self, text: str, /) -> Sized: ...


class CohereImageTokenBilling(Enum):
    SEPARATE_FROM_INPUT_TOKENS = "separate_from_input_tokens"
    FOLDED_INTO_INPUT_TOKENS = "folded_into_input_tokens"


class CohereEmbeddingRowShape(Enum):
    OPENAI = "openai"
    TYPED = "typed"


class _CohereBilledUnits(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    input_tokens: int | None = None
    image_tokens: int | None = None
    images: int | None = None


class _CohereEmbedMeta(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    billed_units: _CohereBilledUnits = Field(default_factory=_CohereBilledUnits)


_CohereEmbeddings: TypeAlias = Mapping[str, Sequence[object]] | Sequence[object]


class _CohereEmbedBody(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    embeddings: _CohereEmbeddings
    meta: _CohereEmbedMeta = Field(default_factory=_CohereEmbedMeta)


def _embedding_rows(vectors: Sequence[object]) -> Iterator[Mapping[str, object]]:
    return ({"object": "embedding", "index": index, "embedding": vector} for index, vector in enumerate(vectors))


def _typed_embedding_rows(embedding_type: str, vectors: Sequence[object]) -> Iterator[Mapping[str, object]]:
    return ({**row, "type": embedding_type} for row in _embedding_rows(vectors))


def _response_rows(embeddings: _CohereEmbeddings, row_shape: CohereEmbeddingRowShape) -> Iterable[Mapping[str, object]]:
    if not isinstance(embeddings, Mapping):
        return _embedding_rows(embeddings)
    match row_shape:
        case CohereEmbeddingRowShape.OPENAI:
            return chain.from_iterable(map(_embedding_rows, embeddings.values()))
        case CohereEmbeddingRowShape.TYPED:
            return chain.from_iterable(starmap(_typed_embedding_rows, embeddings.items()))


def _prompt_usage(prompt_tokens: int, prompt_tokens_details: PromptTokensDetailsWrapper | None) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        total_tokens=prompt_tokens,
        prompt_tokens_details=prompt_tokens_details,
    )


def _billed_usage(billed_units: _CohereBilledUnits, image_billing: CohereImageTokenBilling) -> Usage:
    match image_billing:
        case CohereImageTokenBilling.SEPARATE_FROM_INPUT_TOKENS:
            image_units: Final = (billed_units.image_tokens or 0) + (billed_units.images or 0)
            return _prompt_usage(
                (billed_units.input_tokens or 0) + image_units,
                PromptTokensDetailsWrapper(
                    text_tokens=billed_units.input_tokens,
                    image_tokens=image_units or None,
                    image_count=billed_units.images,
                ),
            )
        case CohereImageTokenBilling.FOLDED_INTO_INPUT_TOKENS:
            return _prompt_usage(billed_units.input_tokens or 0, None)


def _cohere_embed_usage(
    billed_units: _CohereBilledUnits,
    input_texts: Iterable[str],
    encoding: _SupportsEncode,
    image_billing: CohereImageTokenBilling,
) -> Usage:
    if billed_units.input_tokens is None and billed_units.image_tokens is None and billed_units.images is None:
        return _prompt_usage(sum(len(encoding.encode_ordinary(text)) for text in input_texts), None)
    return _billed_usage(billed_units, image_billing)


def cohere_embedding_response(
    body: Mapping[str, object],
    model: str | None,
    encoding: _SupportsEncode,
    input_texts: Iterable[str],
    image_billing: CohereImageTokenBilling,
    row_shape: CohereEmbeddingRowShape,
) -> EmbeddingResponse:
    cohere_body: Final = _CohereEmbedBody.model_validate(body)
    return EmbeddingResponse(
        model=model,
        data=list(_response_rows(cohere_body.embeddings, row_shape)),
        usage=_cohere_embed_usage(cohere_body.meta.billed_units, input_texts, encoding, image_billing),
    )


class CohereEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://docs.cohere.com/v2/reference/embed
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["encoding_format", "dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "encoding_format":
                if isinstance(v, list):
                    optional_params["embedding_types"] = v
                else:
                    optional_params["embedding_types"] = [v]
            elif k == "dimensions":
                optional_params["output_dimension"] = v
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        default_headers: Final = {
            "Content-Type": "application/json",
        }
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        headers = {**default_headers, **headers}
        return headers

    def _is_v3_model(self, model: str) -> bool:
        return "3" in model

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return api_base or "https://api.cohere.ai/v2/embed"

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

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        if isinstance(input, list) and (isinstance(input[0], list) or isinstance(input[0], int)):
            raise ValueError("Input must be a list of strings")
        return cast(
            dict,
            self._transform_request(
                model=model,
                input=cast(list[str], input) if isinstance(input, list) else [input],
                inference_params=optional_params,
            ),
        )

    def transform_native_embed_response(
        self,
        response: httpx.Response,
        api_key: str | None,
        logging_obj: LiteLLMLoggingObj,
        data: Mapping[str, object] | CohereEmbeddingRequest,
        model: str,
        encoding: _SupportsEncode,
        input: Sequence[str],
        row_shape: CohereEmbeddingRowShape,
    ) -> EmbeddingResponse:
        response_json: Final = _DECODED_JSON.validate_python(response.json())
        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=response_json,
        )
        return cohere_embedding_response(
            _JSON_OBJECT.validate_python(response_json),
            model,
            encoding,
            input,
            CohereImageTokenBilling.SEPARATE_FROM_INPUT_TOKENS,
            row_shape,
        )

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        return self.transform_native_embed_response(
            response=raw_response,
            api_key=api_key,
            logging_obj=logging_obj,
            data=request_data,
            model=model,
            encoding=litellm.encoding,
            input=logging_obj.model_call_details["input"],
            row_shape=CohereEmbeddingRowShape.OPENAI,
        )

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return CohereError(
            status_code=status_code,
            message=error_message,
        )
