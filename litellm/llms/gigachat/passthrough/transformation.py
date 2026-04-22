import json
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

import httpx

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.gigachat.authenticator import get_access_token
from litellm.llms.gigachat.chat.streaming import GigaChatModelResponseIterator
from litellm.llms.gigachat.utils import get_api_base
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

if TYPE_CHECKING:
    from httpx import URL, Response

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import CostResponseTypes


class GigaChatPassthroughConfig(BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        return request_data.get("stream", False)

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        endpoint: str,
        request_query_params: Optional[dict],
        litellm_params: dict,
    ) -> Tuple["URL", str]:
        """Get complete API URL for chat completions."""
        base_target_url = self.get_api_base(api_base)

        if base_target_url is None:
            raise Exception("GigaChat api base not found")

        complete_url = f"{base_target_url}/{endpoint.lstrip('/')}"

        return (
            httpx.URL(complete_url),
            base_target_url,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Set up headers with OAuth token.
        """
        # Get access token
        access_token = get_access_token(
            credentials=api_key, litellm_params=litellm_params
        )

        headers["Authorization"] = f"Bearer {access_token}"
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"

        return headers

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: "Response",
        request_data: dict,
        logging_obj: "LiteLLMLoggingObj",
        endpoint: str,
    ) -> Optional["CostResponseTypes"]:
        from litellm import encoding
        from litellm.types.utils import LlmProviders, ModelResponse
        from litellm.utils import ProviderConfigManager

        # cost tracking only for completions 
        if "completions" not in endpoint:
            return None

        provider_chat_config = ProviderConfigManager.get_provider_chat_config(
            provider=LlmProviders(custom_llm_provider),
            model=model,
        )

        if provider_chat_config is None:
            raise ValueError(f"No provider config found for model: {model}")

        litellm_model_response: ModelResponse = provider_chat_config.transform_response(
            model=model,
            messages=request_data.get("messages", []),
            raw_response=httpx_response,
            model_response=ModelResponse(),
            logging_obj=logging_obj,
            optional_params={},
            litellm_params={},
            api_key="",
            request_data=request_data,
            encoding=encoding,
        )

        return litellm_model_response

    def handle_logging_collected_chunks(
        self,
        all_chunks: List[str],
        litellm_logging_obj: "LiteLLMLoggingObj",
        model: str,
        custom_llm_provider: str,
        endpoint: str,
    ) -> Optional["CostResponseTypes"]:
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
        from litellm.types.utils import GenericStreamingChunk, ModelResponseStream

        all_translated_chunks = []

        for chunk in all_chunks:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="ignore")

            if isinstance(chunk, str):
                chunk = chunk.strip()
                if not chunk or chunk == "[DONE]":
                    continue
                if chunk.startswith("data: "):
                    chunk = chunk[6:]
                try:
                    message = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
            elif isinstance(chunk, dict):
                message = chunk
            else:
                continue

            gigachat_iterator = GigaChatModelResponseIterator(
                streaming_response=None,
                sync_stream=False,
            )
            translated_chunk = gigachat_iterator.chunk_parser(chunk=message)

            if isinstance(
                translated_chunk, dict
            ) and generic_chunk_has_all_required_fields(cast(dict, translated_chunk)):
                chunk_obj = convert_generic_chunk_to_model_response_stream(
                    cast(GenericStreamingChunk, translated_chunk)
                )
            elif isinstance(translated_chunk, ModelResponseStream):
                chunk_obj = translated_chunk
            else:
                continue

            all_translated_chunks.append(chunk_obj)

        if len(all_translated_chunks) > 0:
            model_response = stream_chunk_builder(
                chunks=all_translated_chunks,
                logging_obj=litellm_logging_obj,
            )
            return model_response
        return None

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return get_api_base(api_base)

    @staticmethod
    def get_api_key(
        api_key: Optional[str] = None,
    ) -> Optional[str]:
        return api_key or get_secret_str("GIGACHAT_API_KEY")

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        return model

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        return super().get_models(api_key, api_base)
