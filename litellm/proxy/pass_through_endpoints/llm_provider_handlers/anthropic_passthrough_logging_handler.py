import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.llms.anthropic.chat.handler import (
    ModelResponseIterator as AnthropicModelResponseIterator,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict

if TYPE_CHECKING:
    from ..success_handler import PassThroughEndpointLogging
    from ..types import EndpointType
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


class AnthropicPassthroughLoggingHandler:

    @staticmethod
    def anthropic_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Transforms Anthropic response to OpenAI response, generates a standard logging object so downstream logging can be handled
        """
        model = response_body.get("model", "")
        litellm_model_response: litellm.ModelResponse = (
            AnthropicConfig._process_response(
                response=httpx_response,
                model_response=litellm.ModelResponse(),
                model=model,
                stream=False,
                messages=[],
                logging_obj=logging_obj,
                optional_params={},
                api_key="",
                data={},
                print_verbose=litellm.print_verbose,
                encoding=None,
                json_mode=False,
            )
        )

        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=litellm_model_response,
            model=model,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
        )

        return {
            "result": litellm_model_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _create_anthropic_response_logging_payload(
        litellm_model_response: Union[
            litellm.ModelResponse, litellm.TextCompletionResponse
        ],
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
    ):
        """
        Create the standard logging object for Anthropic passthrough

        handles streaming and non-streaming responses
        """
        response_cost = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
        )
        kwargs["response_cost"] = response_cost
        kwargs["model"] = model

        # Make standard logging object for Anthropic
        standard_logging_object = get_standard_logging_object_payload(
            kwargs=kwargs,
            init_response_obj=litellm_model_response,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
            status="success",
        )

        # pretty print standard logging object
        verbose_proxy_logger.debug(
            "standard_logging_object= %s", json.dumps(standard_logging_object, indent=4)
        )
        kwargs["standard_logging_object"] = standard_logging_object

        # set litellm_call_id to logging response object
        litellm_model_response.id = logging_obj.litellm_call_id
        litellm_model_response.model = model
        logging_obj.model_call_details["model"] = model
        return kwargs

    @staticmethod
    def _handle_logging_anthropic_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Takes raw chunks from Anthropic passthrough endpoint and logs them in litellm callbacks

        - Builds complete response from chunks
        - Creates standard logging object
        - Logs in litellm callbacks
        """
        model = request_body.get("model", "")
        complete_streaming_response = (
            AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
                all_chunks=all_chunks,
                litellm_logging_obj=litellm_logging_obj,
                model=model,
            )
        )
        if complete_streaming_response is None:
            verbose_proxy_logger.error(
                "Unable to build complete streaming response for Anthropic passthrough endpoint, not logging..."
            )
            return {
                "result": None,
                "kwargs": {},
            }
        kwargs = AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
            litellm_model_response=complete_streaming_response,
            model=model,
            kwargs={},
            start_time=start_time,
            end_time=end_time,
            logging_obj=litellm_logging_obj,
        )

        return {
            "result": complete_streaming_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _build_complete_streaming_response(
        all_chunks: List[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
    ) -> Optional[Union[litellm.ModelResponse, litellm.TextCompletionResponse]]:
        """
        Builds complete response from raw Anthropic chunks

        - Converts str chunks to generic chunks
        - Converts generic chunks to litellm chunks (OpenAI format)
        - Builds complete response from litellm chunks
        """
        anthropic_model_response_iterator = AnthropicModelResponseIterator(
            streaming_response=None,
            sync_stream=False,
        )
        litellm_custom_stream_wrapper = litellm.CustomStreamWrapper(
            completion_stream=anthropic_model_response_iterator,
            model=model,
            logging_obj=litellm_logging_obj,
            custom_llm_provider="anthropic",
        )
        all_openai_chunks = []
        for _chunk_str in all_chunks:
            try:
                generic_chunk = anthropic_model_response_iterator.convert_str_chunk_to_generic_chunk(
                    chunk=_chunk_str
                )
                litellm_chunk = litellm_custom_stream_wrapper.chunk_creator(
                    chunk=generic_chunk
                )
                if litellm_chunk is not None:
                    all_openai_chunks.append(litellm_chunk)
            except (StopIteration, StopAsyncIteration):
                break
        complete_streaming_response = litellm.stream_chunk_builder(
            chunks=all_openai_chunks
        )
        return complete_streaming_response
