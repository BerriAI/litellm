import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Union

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.gemini.videos.transformation import GeminiVideoConfig
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import (
    ModelResponse,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    from ..success_handler import PassThroughEndpointLogging
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


class GeminiPassthroughLoggingHandler:
    @staticmethod
    def gemini_passthrough_handler(
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
        if "predictLongRunning" in url_route:
            model = GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)

            gemini_video_config = GeminiVideoConfig()
            litellm_video_response = gemini_video_config.transform_video_create_response(
                model=model,
                raw_response=httpx_response,
                logging_obj=logging_obj,
                custom_llm_provider="gemini",
                request_data=request_body,
            )
            logging_obj.model = model
            logging_obj.model_call_details["model"] = model
            logging_obj.model_call_details["custom_llm_provider"] = "gemini"
            logging_obj.custom_llm_provider = "gemini"

            response_cost = litellm.completion_cost(
                completion_response=litellm_video_response,
                model=model,
                custom_llm_provider="gemini",
                call_type="create_video",
            )

            # Set response_cost in _hidden_params to prevent recalculation
            if not hasattr(litellm_video_response, "_hidden_params"):
                litellm_video_response._hidden_params = {}
            litellm_video_response._hidden_params["response_cost"] = response_cost

            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            kwargs["custom_llm_provider"] = "gemini"
            logging_obj.model_call_details["response_cost"] = response_cost
            return {
                "result": litellm_video_response,
                "kwargs": kwargs,
            }

        if "generateContent" in url_route:
            model = GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)

            # Use Gemini config for transformation
            instance_of_gemini_llm = litellm.GoogleAIStudioGeminiConfig()
            litellm_model_response: ModelResponse = instance_of_gemini_llm.transform_response(
                model=model,
                messages=[{"role": "user", "content": "no-message-pass-through-endpoint"}],
                raw_response=httpx_response,
                model_response=litellm.ModelResponse(),
                logging_obj=logging_obj,
                optional_params={},
                litellm_params={},
                api_key="",
                request_data={},
                encoding=litellm.encoding,
            )
            kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
                litellm_model_response=litellm_model_response,
                model=model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                custom_llm_provider="gemini",
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }
        else:
            return {
                "result": None,
                "kwargs": kwargs,
            }

    # NOTE: streamed Gemini has deliberately no handler here.
    #
    # `HttpPassThroughEndpointHelpers.get_endpoint_type` classifies any URL
    # containing `generateContent` / `streamGenerateContent` as
    # `EndpointType.VERTEX_AI` — Gemini's AI Studio host included — so
    # `PassThroughStreamingHandler` reassembles and prices streamed Gemini via
    # `VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks`.
    # That path is already correct for Gemini: it parses the chunks with the
    # very same iterator (both providers share
    # `vertex_and_google_ai_studio_gemini.ModelResponseIterator`), and
    # `VertexPassthroughLoggingHandler._get_custom_llm_provider_from_url`
    # resolves `generativelanguage.googleapis.com` to the `gemini` provider, so
    # the spend row is attributed and priced identically.
    #
    # A `_handle_logging_gemini_collected_chunks` used to live here with no call
    # sites. It was removed rather than wired up: adding an `EndpointType.GEMINI`
    # member plus a dispatch branch would have produced a second code path
    # computing the same number, and unreachable code that looks like coverage
    # is worse than none — it implies a guarantee that does not exist. See
    # `test_streamed_gemini_is_costed_by_the_vertex_path` for the regression
    # guard that replaces it.

    @staticmethod
    def extract_model_from_url(url: str) -> str:
        pattern = r"/models/([^:]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return "unknown"

    @staticmethod
    def _create_gemini_response_logging_payload_for_generate_content(
        litellm_model_response: Union[ModelResponse, TextCompletionResponse],
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str,
    ):
        """
        Create the standard logging object for Gemini passthrough generateContent (streaming and non-streaming)
        """

        response_cost = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
            custom_llm_provider="gemini",
        )

        kwargs["response_cost"] = response_cost
        kwargs["model"] = model
        kwargs["custom_llm_provider"] = custom_llm_provider

        # pretty print standard logging object
        verbose_proxy_logger.debug("kwargs= %s", kwargs)

        # set litellm_call_id to logging response object
        litellm_model_response.id = logging_obj.litellm_call_id
        logging_obj.model = litellm_model_response.model or model
        logging_obj.model_call_details["model"] = logging_obj.model
        logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
        logging_obj.model_call_details["response_cost"] = response_cost
        return kwargs
