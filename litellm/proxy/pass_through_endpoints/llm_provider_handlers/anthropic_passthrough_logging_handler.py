import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic import get_anthropic_config
from litellm.llms.anthropic.chat.handler import (
    ModelResponseIterator as AnthropicModelResponseIterator,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import ModelResponse, TextCompletionResponse

if TYPE_CHECKING:
    from ..success_handler import PassThroughEndpointLogging
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
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
        anthropic_config = get_anthropic_config(url_route)
        litellm_model_response: ModelResponse = anthropic_config().transform_response(
            raw_response=httpx_response,
            model_response=litellm.ModelResponse(),
            model=model,
            messages=[],
            logging_obj=logging_obj,
            optional_params={},
            api_key="",
            request_data={},
            encoding=litellm.encoding,
            json_mode=False,
            litellm_params={},
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
    def _get_user_from_metadata(
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
    ) -> Optional[str]:
        request_body = passthrough_logging_payload.get("request_body")
        if request_body:
            return get_end_user_id_from_request_body(request_body)
        return None

    @staticmethod
    def _create_anthropic_response_logging_payload(
        litellm_model_response: Union[ModelResponse, TextCompletionResponse],
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
        try:
            response_cost = litellm.completion_cost(
                completion_response=litellm_model_response,
                model=model,
            )

            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = (  # type: ignore
                kwargs.get("passthrough_logging_payload")
            )
            if passthrough_logging_payload:
                user = AnthropicPassthroughLoggingHandler._get_user_from_metadata(
                    passthrough_logging_payload=passthrough_logging_payload,
                )
                if user:
                    kwargs.setdefault("litellm_params", {})
                    kwargs["litellm_params"].update(
                        {"proxy_server_request": {"body": {"user": user}}}
                    )

            # pretty print standard logging object
            verbose_proxy_logger.debug(
                "kwargs= %s",
                json.dumps(kwargs, indent=4, default=str),
            )

            # set litellm_call_id to logging response object
            litellm_model_response.id = logging_obj.litellm_call_id
            litellm_model_response.model = model
            logging_obj.model_call_details["model"] = model
            if not logging_obj.model_call_details.get("custom_llm_provider"):
                logging_obj.model_call_details["custom_llm_provider"] = (
                    litellm.LlmProviders.ANTHROPIC.value
                )
            return kwargs
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error creating Anthropic response logging payload: %s", e
            )
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
        # Dheck if it's available in the logging object
        if (
            not model
            and hasattr(litellm_logging_obj, "model_call_details")
            and litellm_logging_obj.model_call_details.get("model")
        ):
            model = cast(str, litellm_logging_obj.model_call_details.get("model"))
            custom_llm_provider = litellm_logging_obj.model_call_details.get(
                "custom_llm_provider"
            )

            if custom_llm_provider and not model.startswith(custom_llm_provider):
                model = f"{custom_llm_provider}/{model}"
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
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
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
        all_openai_chunks = []
        for _chunk_str in all_chunks:
            try:
                transformed_openai_chunk = anthropic_model_response_iterator.convert_str_chunk_to_generic_chunk(
                    chunk=_chunk_str
                )
                if transformed_openai_chunk is not None:
                    all_openai_chunks.append(transformed_openai_chunk)

            except (StopIteration, StopAsyncIteration):
                break
        complete_streaming_response = litellm.stream_chunk_builder(
            chunks=all_openai_chunks,
            logging_obj=litellm_logging_obj,
        )
        return complete_streaming_response
