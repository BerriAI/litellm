"""
GigaChat Embedding Transformation

Transforms OpenAI /v1/embeddings format to GigaChat format.
API Documentation: https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/post-embeddings
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Final, Protocol, runtime_checkable

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm import LlmProviders
from litellm.litellm_core_utils.core_helpers import iter_items, json_contains, subscript
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.gigachat.utils import get_api_base
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse

from ..authenticator import get_access_token

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_DECODED_JSON: Final = TypeAdapter(object)


@runtime_checkable
class _SupportsDelItem(Protocol):
    def __delitem__(self, key: str, /) -> None: ...


@runtime_checkable
class _SupportsSetItem(Protocol):
    def __setitem__(self, key: str, value: object, /) -> None: ...


def _add_prompt_tokens(total_tokens: float, embedding: object) -> float:
    if not (json_contains(embedding, "usage") and json_contains(subscript(embedding, "usage"), "prompt_tokens")):
        return total_tokens
    prompt_tokens: Final = subscript(subscript(embedding, "usage"), "prompt_tokens")
    if isinstance(prompt_tokens, int | float):
        return total_tokens + prompt_tokens
    raise TypeError(
        f"unsupported operand type(s) for +=: '{type(total_tokens).__name__}' and '{type(prompt_tokens).__name__}'"
    )


class GigaChatEmbeddingError(BaseLLMException):
    """GigaChat Embedding API error."""


class GigaChatEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration class for GigaChat Embeddings API.

    GigaChat embeddings endpoint: POST /api/v1/embeddings
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def get_config(cls):
        class_attributes: Final[Mapping[str, object]] = types.MappingProxyType(cls.__dict__)
        return {
            k: v
            for k, v in class_attributes.items()
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

    def get_supported_openai_params(self, model: str) -> list[str]:
        """GigaChat embeddings don't support additional parameters."""
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """Map OpenAI params to GigaChat format (no special mapping needed)."""
        return optional_params

    def _get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str, str | None, str | None]:
        """
        Returns provider info for GigaChat.

        Returns:
            tuple of (custom_llm_provider, api_base, dynamic_api_key)
        """
        api_base = get_api_base(api_base)
        return LlmProviders.GIGACHAT.value, api_base, api_key

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """Get the complete URL for embeddings endpoint."""
        return join_api_base(get_api_base(api_base), "embeddings")

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI embedding request to GigaChat format.

        GigaChat format:
        {
            "model": "Embeddings",
            "input": ["text1", "text2", ...]
        }
        """
        normalized_input: Final = [input] if isinstance(input, str) else input
        return {
            "model": model.removeprefix("gigachat/"),
            "input": normalized_input,
        }

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
        """
        Transform GigaChat embedding response to OpenAI format.

        GigaChat returns:
        {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [...], "index": 0, "usage": {...}}],
            "model": "Embeddings"
        }
        """
        response_json: Final = _DECODED_JSON.validate_python(raw_response.json())

        # Log response
        logging_obj.post_call(
            input=request_data.get("input"),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_json,
        )

        # Calculate total tokens from individual embeddings
        total_tokens: float = 0
        if json_contains(response_json, "data"):
            for emb in iter_items(subscript(response_json, "data")):
                total_tokens = _add_prompt_tokens(total_tokens, emb)
                # Remove usage from individual embeddings (not part of OpenAI format)
                if json_contains(emb, "usage"):
                    if not isinstance(emb, _SupportsDelItem):
                        raise TypeError(f"'{type(emb).__name__}' object does not support item deletion")
                    del emb["usage"]

        # Set overall usage
        if not isinstance(response_json, _SupportsSetItem):
            raise TypeError(f"'{type(response_json).__name__}' object does not support item assignment")
        response_json["usage"] = {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        }

        return EmbeddingResponse.model_validate(_JSON_OBJECT.validate_python(response_json))

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
        """
        Set up headers with OAuth token for GigaChat.
        """
        # Get access token via OAuth
        access_token: Final = get_access_token(credentials=api_key, litellm_params=litellm_params)

        default_headers: Final = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        return {**default_headers, **headers}

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        """Return GigaChat-specific error class."""
        return GigaChatEmbeddingError(
            status_code=status_code,
            message=error_message,
        )
