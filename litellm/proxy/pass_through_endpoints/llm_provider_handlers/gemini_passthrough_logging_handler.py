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
from litellm.types.llms.openai import LiteLLMBatch
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
        # Handle batch endpoints
        if "batchGenerateContent" in url_route:
            return GeminiPassthroughLoggingHandler.batch_handler(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                **kwargs,
            )
        
        # Handle generateContent endpoints
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
    def _build_complete_streaming_response(
        all_chunks: List[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        url_route: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        parsed_chunks = []
        if "generateContent" in url_route or "streamGenerateContent" in url_route:
            gemini_iterator: Any = GeminiModelResponseIterator(
                streaming_response=None,
                sync_stream=False,
                logging_obj=litellm_logging_obj,
            )
            chunk_parsing_logic: Any = gemini_iterator._common_chunk_parsing_logic
            parsed_chunks = [chunk_parsing_logic(chunk) for chunk in all_chunks]
        else:
            return None

        if len(parsed_chunks) == 0:
            return None

        all_openai_chunks = []
        for parsed_chunk in parsed_chunks:
            if parsed_chunk is None:
                continue
            all_openai_chunks.append(parsed_chunk)

        complete_streaming_response = litellm.stream_chunk_builder(chunks=all_openai_chunks)

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
            custom_llm_provider="gemini",
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

    @staticmethod
    def batch_handler(  # noqa: PLR0915
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle Gemini batch API passthrough logging.
        Creates a managed object for cost tracking when batch job is successfully created.
        Supports both file-based and inline batch types.
        """
        from litellm._uuid import uuid
        from litellm.types.llms.openai import LiteLLMBatch, Choices
        from litellm.proxy._types import SpecialEnums
        import base64

        try:
            _json_response = httpx_response.json()
            
            # Only handle successful batch job creation/retrieval (POST/GET requests)
            if httpx_response.status_code == 200 and "name" in _json_response:
                # Extract batch ID and model from the response
                batch_id = _json_response.get("name", "")  # e.g., "batches/abc123"
                
                model_name = GeminiPassthroughLoggingHandler._extract_model_from_batch_response(_json_response, url_route)
                litellm_batch_response = GeminiPassthroughLoggingHandler._transform_gemini_batch_to_litellm_batch(_json_response)
                actual_model_id = GeminiPassthroughLoggingHandler._get_actual_model_id_from_router(model_name)
                unified_id_string = SpecialEnums.LITELLM_MANAGED_BATCH_COMPLETE_STR.value.format(actual_model_id, batch_id)
                unified_object_id = base64.urlsafe_b64encode(unified_id_string.encode()).decode().rstrip("=")
                
                GeminiPassthroughLoggingHandler._store_batch_managed_object(
                    unified_object_id=unified_object_id,
                    batch_object=litellm_batch_response,
                    model_object_id=batch_id,
                    logging_obj=logging_obj,
                    **kwargs,
                )

                
                # Create a batch job response for logging
                litellm_model_response = ModelResponse()
                litellm_model_response.id = str(uuid.uuid4())
                litellm_model_response.model = model_name
                litellm_model_response.object = "batch"
                litellm_model_response.created = int(start_time.timestamp())
                
                # Get batch state
                batch_state = _json_response.get("state", {}).get("name", "JOB_STATE_PENDING")
                
                # Add batch-specific metadata
                litellm_model_response.choices = [Choices(
                    finish_reason="batch_pending" if batch_state == "JOB_STATE_PENDING" else "batch_complete",
                    index=0,
                    message={
                        "role": "assistant",
                        "content": f"Batch job {batch_id} - Status: {batch_state}",
                        "tool_calls": None,
                        "function_call": None,
                        "provider_specific_fields": {
                            "batch_job_id": batch_id,
                            "batch_job_state": batch_state,
                            "unified_object_id": unified_object_id
                        }
                    }
                )]
                
                # Set response cost to 0 initially (will be updated when batch completes)
                response_cost = 0.0
                kwargs["response_cost"] = response_cost
                kwargs["model"] = model_name
                kwargs["batch_id"] = batch_id
                if unified_object_id:
                    kwargs["unified_object_id"] = unified_object_id
                kwargs["batch_job_state"] = batch_state
                
                logging_obj.model = model_name
                logging_obj.model_call_details["model"] = logging_obj.model
                logging_obj.model_call_details["response_cost"] = response_cost
                logging_obj.model_call_details["batch_id"] = batch_id
                
                return {
                    "result": litellm_model_response,
                    "kwargs": kwargs,
                }
            else:
                # Handle non-successful responses
                litellm_model_response = ModelResponse()
                litellm_model_response.id = str(uuid.uuid4())
                litellm_model_response.model = "gemini_batch"
                litellm_model_response.object = "batch"
                litellm_model_response.created = int(start_time.timestamp())
                
                # Add error-specific metadata
                litellm_model_response.choices = [Choices(
                    finish_reason="batch_error",
                    index=0,
                    message={
                        "role": "assistant",
                        "content": f"Batch job creation/retrieval failed. Status: {httpx_response.status_code}",
                        "tool_calls": None,
                        "function_call": None,
                        "provider_specific_fields": {
                            "batch_job_state": "JOB_STATE_FAILED",
                            "status_code": httpx_response.status_code
                        }
                    }
                )]
                
                kwargs["response_cost"] = 0.0
                kwargs["model"] = "gemini_batch"
                kwargs["batch_job_state"] = "JOB_STATE_FAILED"
                
                return {
                    "result": litellm_model_response,
                    "kwargs": kwargs,
                }
                
        except Exception as e:
            verbose_proxy_logger.error(f"Error in gemini batch_handler: {e}")
            # Return basic response on error
            litellm_model_response = ModelResponse()
            litellm_model_response.id = str(uuid.uuid4())
            litellm_model_response.model = "gemini_batch"
            litellm_model_response.object = "batch"
            litellm_model_response.created = int(start_time.timestamp())
            
            # Add error-specific metadata
            litellm_model_response.choices = [Choices(
                finish_reason="batch_error",
                index=0,
                message={
                    "role": "assistant",
                    "content": f"Error processing batch job: {str(e)}",
                    "tool_calls": None,
                    "function_call": None,
                    "provider_specific_fields": {
                        "batch_job_state": "JOB_STATE_FAILED",
                        "error": str(e)
                    }
                }
            )]
            
            kwargs["response_cost"] = 0.0
            kwargs["model"] = "gemini_batch"
            kwargs["batch_job_state"] = "JOB_STATE_FAILED"
            
            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }

    @staticmethod
    def _extract_model_from_batch_response(batch_response: dict, url_route: str) -> str:
        """Extract model name from Gemini batch response or URL"""
        # Try to get model from response metadata
        model = batch_response.get("model", "")
        if model:
            # Extract just the model name from full path like "models/gemini-2.5-flash"
            if "/" in model:
                model = model.split("/")[-1]
            return model
        
        # Fallback: try to extract from URL
        return GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)

    @staticmethod
    def _transform_gemini_batch_to_litellm_batch(gemini_response: dict) -> LiteLLMBatch:
        """Transform Gemini batch API response to LiteLLMBatch format"""
        from litellm.llms.gemini.batches.handler import GeminiBatchesAPI
        
        # Use the centralized transformation method from GeminiBatchesAPI
        gemini_api = GeminiBatchesAPI()
        return gemini_api._transform_gemini_batch_response(gemini_response)

    @staticmethod
    def _store_batch_managed_object(
        unified_object_id: str,
        batch_object: LiteLLMBatch,
        model_object_id: str,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> None:
        """
        Store batch managed object for cost tracking.
        This will be picked up by the check_batch_cost polling mechanism.
        """
        try:           
            # Get the managed files hook from the logging object
            from litellm.proxy.proxy_server import proxy_logging_obj
            
            managed_files_hook = proxy_logging_obj.get_proxy_hook("managed_files")
            if managed_files_hook is not None and hasattr(managed_files_hook, 'store_unified_object_id'):
                # Create a mock user API key dict for the managed object storage
                from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles
                
                user_api_key_dict = UserAPIKeyAuth(
                    user_id=kwargs.get("user_id", "default-user"),
                    api_key="",
                    team_id=None,
                    team_alias=None,
                    user_role=LitellmUserRoles.CUSTOMER,
                    user_email=None,
                    max_budget=None,
                    spend=0.0,
                    models=[],
                    tpm_limit=None,
                    rpm_limit=None,
                    budget_duration=None,
                    budget_reset_at=None,
                    max_parallel_requests=None,
                    allowed_model_region=None,
                    metadata={},
                    key_alias=None,
                    permissions={},
                    model_max_budget={},
                    model_spend={},
                )
                
                # Store the unified object for batch cost tracking
                import asyncio
                asyncio.create_task(
                    managed_files_hook.store_unified_object_id(
                        unified_object_id=unified_object_id,
                        file_object=batch_object,
                        litellm_parent_otel_span=None,
                        model_object_id=model_object_id,
                        file_purpose="batch",
                        user_api_key_dict=user_api_key_dict,
                    )
                )
                
                verbose_proxy_logger.info(
                    f"Stored Gemini batch managed object with unified_object_id={unified_object_id}, batch_id={model_object_id}"
                )
            else:
                verbose_proxy_logger.warning("Managed files hook not available, cannot store batch object for cost tracking")
                
        except Exception as e:
            verbose_proxy_logger.error(f"Error storing Gemini batch managed object: {e}")

    @staticmethod
    def _get_actual_model_id_from_router(model_name: str) -> str:
        """Get actual model ID from router or use model name as fallback"""
        try:
            from litellm.proxy.proxy_server import llm_router
            
            if llm_router is not None:
                # Use the existing get_model_ids method from router
                model_ids = llm_router.get_model_ids(model_name=model_name)
                if model_ids and len(model_ids) > 0:
                    actual_model_id = model_ids[0]
                    verbose_proxy_logger.info(f"Found model ID in router: {actual_model_id}")
                    return actual_model_id
                else:
                    verbose_proxy_logger.warning(f"Model not found in router, using model name: {model_name}")
                    return model_name
            else:
                verbose_proxy_logger.warning(f"Router not available, using model name: {model_name}")
                return model_name
        except Exception as e:
            verbose_proxy_logger.error(f"Error getting model ID from router: {e}")
            return model_name
