from datetime import datetime
from typing import List, Optional, Union

import httpx

import litellm
from litellm import stream_chunk_builder
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig
from litellm.llms.cohere.common_utils import (
    ModelResponseIterator as CohereModelResponseIterator,
)
from litellm.llms.cohere.embed.v1_transformation import CohereEmbeddingConfig
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import (
    LlmProviders,
    ModelResponse,
    TextCompletionResponse,
)

from .base_passthrough_logging_handler import BasePassthroughLoggingHandler


class CoherePassthroughLoggingHandler(BasePassthroughLoggingHandler):
    @property
    def llm_provider_name(self) -> LlmProviders:
        return LlmProviders.COHERE

    def get_provider_config(self, model: str) -> BaseConfig:
        return CohereV2ChatConfig()

    def _build_complete_streaming_response(
        self,
        all_chunks: List[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        cohere_model_response_iterator = CohereModelResponseIterator(
            streaming_response=None,
            sync_stream=False,
        )
        litellm_custom_stream_wrapper = CustomStreamWrapper(
            completion_stream=cohere_model_response_iterator,
            model=model,
            logging_obj=litellm_logging_obj,
            custom_llm_provider="cohere",
        )
        all_openai_chunks = []
        for _chunk_str in all_chunks:
            try:
                generic_chunk = (
                    cohere_model_response_iterator.convert_str_chunk_to_generic_chunk(
                        chunk=_chunk_str
                    )
                )
                litellm_chunk = litellm_custom_stream_wrapper.chunk_creator(
                    chunk=generic_chunk
                )
                if litellm_chunk is not None:
                    all_openai_chunks.append(litellm_chunk)
            except (StopIteration, StopAsyncIteration):
                break
        complete_streaming_response = stream_chunk_builder(chunks=all_openai_chunks)
        return complete_streaming_response

    def cohere_passthrough_handler(  # noqa: PLR0915
        self,
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle Cohere passthrough logging with route detection and cost tracking.
        """
        # Check if this is an embed endpoint
        if "/v1/embed" in url_route:
            model = request_body.get("model", response_body.get("model", ""))
            try:
                cohere_embed_config = CohereEmbeddingConfig()
                litellm_model_response = litellm.EmbeddingResponse()
                handler_instance = CoherePassthroughLoggingHandler()

                input_texts = request_body.get("texts", [])
                if not input_texts:
                    input_texts = request_body.get("input", [])

                # Transform the response
                litellm_model_response = cohere_embed_config._transform_response(
                    response=httpx_response,
                    api_key="",
                    logging_obj=logging_obj,
                    data=request_body,
                    model_response=litellm_model_response,
                    model=model,
                    encoding=litellm.encoding,
                    input=input_texts,
                )

                # Calculate cost using LiteLLM's cost calculator
                response_cost = litellm.completion_cost(
                    completion_response=litellm_model_response,
                    model=model,
                    custom_llm_provider="cohere",
                    call_type="aembedding",
                )

                # Set the calculated cost in _hidden_params to prevent recalculation
                if not hasattr(litellm_model_response, "_hidden_params"):
                    litellm_model_response._hidden_params = {}
                litellm_model_response._hidden_params["response_cost"] = response_cost

                kwargs["response_cost"] = response_cost
                kwargs["model"] = model
                kwargs["custom_llm_provider"] = "cohere"

                # Extract user information for tracking
                passthrough_logging_payload: Optional[
                    PassthroughStandardLoggingPayload
                ] = kwargs.get("passthrough_logging_payload")
                if passthrough_logging_payload:
                    user = handler_instance._get_user_from_metadata(
                        passthrough_logging_payload=passthrough_logging_payload,
                    )
                    if user:
                        kwargs.setdefault("litellm_params", {})
                        kwargs["litellm_params"].update(
                            {"proxy_server_request": {"body": {"user": user}}}
                        )

                # Create standard logging object
                if litellm_model_response is not None:
                    get_standard_logging_object_payload(
                        kwargs=kwargs,
                        init_response_obj=litellm_model_response,
                        start_time=start_time,
                        end_time=end_time,
                        logging_obj=logging_obj,
                        status="success",
                    )

                # Update logging object with cost information
                logging_obj.model_call_details["model"] = model
                logging_obj.model_call_details["custom_llm_provider"] = "cohere"
                logging_obj.model_call_details["response_cost"] = response_cost

                return {
                    "result": litellm_model_response,
                    "kwargs": kwargs,
                }
            except Exception:
                # For other routes (e.g., /v2/chat), fall back to chat handler
                return super().passthrough_chat_handler(
                    httpx_response=httpx_response,
                    response_body=response_body,
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
        
        # For non-embed routes (e.g., /v2/chat), fall back to chat handler
        return super().passthrough_chat_handler(
            httpx_response=httpx_response,
            response_body=response_body,
            logging_obj=logging_obj,
            url_route=url_route,
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            request_body=request_body,
            **kwargs,
        )
