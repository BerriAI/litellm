import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator as GeminiModelResponseIterator,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import (
    ModelResponse,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from ..success_handler import PassThroughEndpointLogging
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
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

    @staticmethod
    def _handle_logging_gemini_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        model: Optional[str],
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Takes raw chunks from Gemini passthrough endpoint and logs them in litellm callbacks

        - Builds complete response from chunks
        - Creates standard logging object
        - Logs in litellm callbacks
        """
        kwargs: Dict[str, Any] = {}
        model = model or GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)
        complete_streaming_response = GeminiPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=all_chunks,
            litellm_logging_obj=litellm_logging_obj,
            model=model,
            url_route=url_route,
        )

        if complete_streaming_response is None:
            verbose_proxy_logger.error(
                "Unable to build complete streaming response for Gemini passthrough endpoint, not logging..."
            )
            return {
                "result": None,
                "kwargs": kwargs,
            }

        kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
            litellm_model_response=complete_streaming_response,
            model=model,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=litellm_logging_obj,
            custom_llm_provider="gemini",
        )

        return {
            "result": complete_streaming_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _parse_gemini_streaming_json(
        all_chunks: List[str],
        gemini_iterator: GeminiModelResponseIterator,
    ) -> List[Any]:
        """
        Parse Gemini streaming chunks from fragmented JSON strings.
        
        Gemini's streaming format sends a single JSON array that may be split across
        multiple lines. This method:
        1. Joins all fragmented string chunks into complete JSON
        2. Parses the JSON array/object
        3. Transforms each item using Gemini's chunk_parser
        
        Args:
            all_chunks: Raw string chunks from the streaming response
            gemini_iterator: GeminiModelResponseIterator instance for parsing
            
        Returns:
            List of parsed chunks in OpenAI format, or empty list if parsing fails
        """
        parsed_chunks = []
        
        verbose_proxy_logger.debug(f"Gemini streaming: Processing {len(all_chunks)} raw chunks")
        
        # Gemini streaming response is a single JSON array that may be split across lines
        # Join all chunks back together to reconstruct the complete JSON
        combined_chunk = "".join(all_chunks)
        
        # Parse the combined JSON string
        try:
            dict_chunk = json.loads(combined_chunk)
            verbose_proxy_logger.debug(f"Parsed JSON object: {type(dict_chunk)}")
            
            # Gemini returns an array of response objects
            if isinstance(dict_chunk, list):
                for item in dict_chunk:
                    try:
                        # Call chunk_parser directly with the dict, not _common_chunk_parsing_logic
                        parsed_chunk = gemini_iterator.chunk_parser(chunk=item)
                        if parsed_chunk is not None:
                            parsed_chunks.append(parsed_chunk)
                    except Exception as e:
                        verbose_proxy_logger.error(f"Error parsing Gemini chunk item: {e}", exc_info=True)
                        continue
            else:
                # Single object response
                parsed_chunk = gemini_iterator.chunk_parser(chunk=dict_chunk)
                if parsed_chunk is not None:
                    parsed_chunks.append(parsed_chunk)
                    
        except json.JSONDecodeError as e:
            verbose_proxy_logger.error(f"Failed to parse Gemini streaming response as JSON: {e}")
            return []
        
        verbose_proxy_logger.debug(f"Total parsed chunks: {len(parsed_chunks)}")
        return parsed_chunks

    @staticmethod
    def _build_complete_streaming_response(
        all_chunks: List[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        url_route: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        if "generateContent" not in url_route and "streamGenerateContent" not in url_route:
            return None
            
        gemini_iterator: Any = GeminiModelResponseIterator(
            streaming_response=None,
            sync_stream=False,
            logging_obj=litellm_logging_obj,
        )
        
        # Parse the streaming chunks
        parsed_chunks = GeminiPassthroughLoggingHandler._parse_gemini_streaming_json(
            all_chunks=all_chunks,
            gemini_iterator=gemini_iterator,
        )

        all_openai_chunks = []
        for parsed_chunk in parsed_chunks:
            if parsed_chunk is None:
                continue
            all_openai_chunks.append(parsed_chunk)

        complete_streaming_response = litellm.stream_chunk_builder(
            chunks=all_openai_chunks,
            logging_obj=litellm_logging_obj,
        )

        return complete_streaming_response

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
            custom_llm_provider=custom_llm_provider,
        )

        kwargs["response_cost"] = response_cost
        kwargs["model"] = model
        kwargs["custom_llm_provider"] = custom_llm_provider

        # pretty print standard logging object
        verbose_proxy_logger.debug("kwargs= %s", json.dumps(kwargs, indent=4))

        # set litellm_call_id to logging response object
        litellm_model_response.id = logging_obj.litellm_call_id
        logging_obj.model = litellm_model_response.model or model
        logging_obj.model_call_details["model"] = logging_obj.model
        logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
        logging_obj.model_call_details["response_cost"] = response_cost
        return kwargs
