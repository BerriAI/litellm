from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

import httpx

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.gigachat.authenticator import get_access_token
from litellm.llms.gigachat.chat.streaming import GigaChatModelResponseIterator
from litellm.llms.gigachat.utils import GIGACHAT_BASE_URL
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import EmbeddingResponse

if TYPE_CHECKING:
    from httpx import URL, Response

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import CostResponseTypes


class GigaChatPassthroughConfig(BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: Mapping[str, object]) -> bool:
        return request_data.get("stream", False)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
        request_query_params: Mapping[str, object] | None,
        litellm_params: Mapping[str, object],
    ) -> tuple[URL, str]:
        """Get complete API URL for chat completions."""
        base_target_url: Final = self.get_api_base(api_base)

        if base_target_url is None:
            raise Exception("GigaChat api base not found")

        complete_url: Final = f"{base_target_url}/{endpoint.lstrip('/')}"

        return (
            httpx.URL(complete_url),
            base_target_url,
        )

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: mutates in place to set OAuth headers
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: base class contract returns dict for httpx
        """
        Set up headers with OAuth token.
        """
        access_token: Final = get_access_token(credentials=api_key, litellm_params=litellm_params)

        headers["Authorization"] = f"Bearer {access_token}"  # rebind-ok: mutating for OAuth setup
        headers["Content-Type"] = "application/json"  # rebind-ok: mutating for OAuth setup
        headers["Accept"] = "application/json"  # rebind-ok: mutating for OAuth setup

        return headers

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: Response,
        request_data: Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
        endpoint: str,
    ) -> CostResponseTypes | None:
        from litellm import encoding
        from litellm.types.utils import LlmProviders, ModelResponse
        from litellm.utils import ProviderConfigManager

        if "completions" in endpoint:
            provider_chat_config: Final = ProviderConfigManager.get_provider_chat_config(
                provider=LlmProviders(custom_llm_provider),
                model=model,
            )

            if provider_chat_config is None:
                raise ValueError(f"No provider config found for model: {model}")

            raw_messages: Final = request_data.get("messages")
            litellm_model_response: Final = provider_chat_config.transform_response(
                model=model,
                messages=list(raw_messages)
                if isinstance(raw_messages, list)
                else [],  # mutable-ok: transform_response wants a list
                raw_response=httpx_response,
                model_response=ModelResponse(),
                logging_obj=logging_obj,
                optional_params={},  # mutable-ok: empty dict kwarg for transform_response
                litellm_params={},  # mutable-ok: empty dict kwarg for transform_response
                api_key="",
                request_data=dict(request_data),  # mutable-ok: transform_response wants a dict
                encoding=encoding,
            )

            return litellm_model_response

        if "embeddings" in endpoint:
            provider_embedding_config: Final = ProviderConfigManager.get_provider_embedding_config(
                provider=LlmProviders(custom_llm_provider),
                model=model,
            )

            if provider_embedding_config is None:
                raise ValueError(f"No provider config found for model: {model}")

            litellm_embedding_response: Final[EmbeddingResponse] = (
                provider_embedding_config.transform_embedding_response(
                    model=model,
                    raw_response=httpx_response,
                    model_response=EmbeddingResponse(),
                    logging_obj=logging_obj,
                    optional_params={},  # mutable-ok: empty dict kwarg for transform_embedding_response
                    api_key="",
                    request_data=dict(request_data),  # mutable-ok: transform_embedding_response wants a dict
                    litellm_params={},  # mutable-ok: empty dict kwarg for transform_embedding_response
                )
            )

            return litellm_embedding_response

        return None

    def handle_logging_collected_chunks(
        self,
        all_chunks: Sequence[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        custom_llm_provider: str,
        endpoint: str,
    ) -> CostResponseTypes | None:
        """
        1. Convert all_chunks to a ModelResponseStream
        2. combine model_response_stream to model_response
        3. Return the model_response
        """

        from litellm.litellm_core_utils.streaming_handler import (
            convert_generic_chunk_to_model_response_stream,
            generic_chunk_has_all_required_fields,
        )
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import ModelResponseStream

        all_translated_chunks: Final[list[object]] = []  # mutable-ok: accumulator

        for chunk in all_chunks:
            chunk = chunk.strip()
            if not chunk or chunk == "[DONE]":
                continue
            chunk = chunk.removeprefix("data: ")
            try:
                message = json.loads(chunk)
            except json.JSONDecodeError:
                continue

            gigachat_iterator = GigaChatModelResponseIterator(
                streaming_response=None,
                sync_stream=False,
            )
            translated_chunk = gigachat_iterator.chunk_parser(chunk=message)

            if isinstance(translated_chunk, dict) and generic_chunk_has_all_required_fields(  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard for patched chunk_parser
                dict(translated_chunk)
            ):
                chunk_obj = convert_generic_chunk_to_model_response_stream(
                    translated_chunk  # pyright: ignore[reportArgumentType]  # validated TypedDict
                )
            elif isinstance(translated_chunk, ModelResponseStream):
                chunk_obj = translated_chunk
            else:
                continue

            all_translated_chunks.append(chunk_obj)

        if len(all_translated_chunks) > 0:
            return stream_chunk_builder(
                chunks=all_translated_chunks,
                logging_obj=litellm_logging_obj,
            )
        return None

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("GIGACHAT_API_BASE") or GIGACHAT_BASE_URL

    @staticmethod
    def get_api_key(
        api_key: str | None = None,
    ) -> str | None:
        return api_key or get_secret_str("GIGACHAT_API_KEY")

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        return list(super().get_models(api_key, api_base))
