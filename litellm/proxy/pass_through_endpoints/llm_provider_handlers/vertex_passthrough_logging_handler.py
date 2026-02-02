import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast
from urllib.parse import urlparse

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator as VertexModelResponseIterator,
)
from litellm.llms.vertex_ai.vector_stores.search_api.transformation import (
    VertexSearchAPIVectorStoreConfig,
)
from litellm.llms.vertex_ai.videos.transformation import VertexAIVideoConfig
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import (
    Choices,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    SpecialEnums,
    StandardPassThroughResponseObject,
    TextCompletionResponse,
)

vertex_search_api_config = VertexSearchAPIVectorStoreConfig()
if TYPE_CHECKING:
    from litellm.types.utils import LiteLLMBatch

    from ..success_handler import PassThroughEndpointLogging
else:
    PassThroughEndpointLogging = Any
    LiteLLMBatch = Any

# Define EndpointType locally to avoid import issues
EndpointType = Any


class VertexPassthroughLoggingHandler:

    @staticmethod
    def vertex_passthrough_handler(
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: Optional[dict] = None,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        if "predictLongRunning" in url_route:
            model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)
            
            vertex_video_config = VertexAIVideoConfig()
            litellm_video_response = vertex_video_config.transform_video_create_response(
                model=model,
                raw_response=httpx_response,
                logging_obj=logging_obj,
                custom_llm_provider="vertex_ai",
                request_data=request_body,
            )
            
            logging_obj.model = model
            logging_obj.model_call_details["model"] = model
            logging_obj.model_call_details["custom_llm_provider"] = "vertex_ai"
            logging_obj.custom_llm_provider = "vertex_ai"
            
            response_cost = litellm.completion_cost(
                completion_response=litellm_video_response,
                model=model,
                custom_llm_provider="vertex_ai",
                call_type="create_video",
            )
            
            # Set response_cost in _hidden_params to prevent recalculation
            if not hasattr(litellm_video_response, "_hidden_params"):
                litellm_video_response._hidden_params = {}
            litellm_video_response._hidden_params["response_cost"] = response_cost
            
            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            kwargs["custom_llm_provider"] = "vertex_ai"
            logging_obj.model_call_details["response_cost"] = response_cost
            
            return {
                "result": litellm_video_response,
                "kwargs": kwargs,
            }
        
        elif "generateContent" in url_route:
            model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)

            instance_of_vertex_llm = litellm.VertexGeminiConfig()
            litellm_model_response: ModelResponse = (
                instance_of_vertex_llm.transform_response(
                    model=model,
                    messages=[
                        {"role": "user", "content": "no-message-pass-through-endpoint"}
                    ],
                    raw_response=httpx_response,
                    model_response=litellm.ModelResponse(),
                    logging_obj=logging_obj,
                    optional_params={},
                    litellm_params={},
                    api_key="",
                    request_data={},
                    encoding=litellm.encoding,
                )
            )
            kwargs = VertexPassthroughLoggingHandler._create_vertex_response_logging_payload_for_generate_content(
                litellm_model_response=litellm_model_response,
                model=model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                custom_llm_provider=VertexPassthroughLoggingHandler._get_custom_llm_provider_from_url(
                    url_route
                ),
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }

        elif "predict" in url_route:
            return VertexPassthroughLoggingHandler._handle_predict_response(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                kwargs=kwargs,
            )
        elif "rawPredict" in url_route or "streamRawPredict" in url_route:
            from litellm.llms.vertex_ai.vertex_ai_partner_models import (
                get_vertex_ai_partner_model_config,
            )

            model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)
            vertex_publisher_or_api_spec = VertexPassthroughLoggingHandler._get_vertex_publisher_or_api_spec_from_url(
                url_route
            )

            _json_response = httpx_response.json()

            litellm_prediction_response = ModelResponse()

            if vertex_publisher_or_api_spec is not None:
                vertex_ai_partner_model_config = get_vertex_ai_partner_model_config(
                    model=model,
                    vertex_publisher_or_api_spec=vertex_publisher_or_api_spec,
                )
                litellm_prediction_response = (
                    vertex_ai_partner_model_config.transform_response(
                        model=model,
                        raw_response=httpx_response,
                        model_response=litellm_prediction_response,
                        logging_obj=logging_obj,
                        request_data={},
                        encoding=litellm.encoding,
                        optional_params={},
                        litellm_params={},
                        api_key="",
                        messages=[
                            {
                                "role": "user",
                                "content": "no-message-pass-through-endpoint",
                            }
                        ],
                    )
                )

            kwargs = VertexPassthroughLoggingHandler._create_vertex_response_logging_payload_for_generate_content(
                litellm_model_response=litellm_prediction_response,
                model="vertex_ai/" + model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                custom_llm_provider="vertex_ai",
            )

            return {
                "result": litellm_prediction_response,
                "kwargs": kwargs,
            }
        elif "search" in url_route:

            litellm_vs_response = (
                vertex_search_api_config.transform_search_vector_store_response(
                    response=httpx_response,
                    litellm_logging_obj=logging_obj,
                )
            )
            response_cost = litellm.completion_cost(
                completion_response=litellm_vs_response,
                model="vertex_ai/search_api",
                custom_llm_provider="vertex_ai",
                call_type="vector_store_search",
            )

            standard_pass_through_response_object: StandardPassThroughResponseObject = {
                "response": cast(dict, litellm_vs_response),
            }

            kwargs["response_cost"] = response_cost
            kwargs["model"] = "vertex_ai/search_api"
            logging_obj.model_call_details.setdefault("litellm_params", {})
            logging_obj.model_call_details["litellm_params"][
                "base_model"
            ] = "vertex_ai/search_api"
            logging_obj.model_call_details["response_cost"] = response_cost

            return {
                "result": standard_pass_through_response_object,
                "kwargs": kwargs,
            }
        elif "batchPredictionJobs" in url_route:
            return VertexPassthroughLoggingHandler.batch_prediction_jobs_handler(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                **kwargs,
            )
        else:
            return {
                "result": None,
                "kwargs": kwargs,
            }

    @staticmethod
    def _handle_predict_response(
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        kwargs: dict,
    ) -> PassThroughEndpointLoggingTypedDict:
        """Handle predict endpoint responses (embeddings, image generation)."""
        from litellm.llms.vertex_ai.image_generation.image_generation_handler import (
            VertexImageGeneration,
        )
        from litellm.llms.vertex_ai.multimodal_embeddings.transformation import (
            VertexAIMultimodalEmbeddingConfig,
        )
        from litellm.types.utils import PassthroughCallTypes

        vertex_image_generation_class = VertexImageGeneration()

        model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)

        _json_response = httpx_response.json()

        litellm_prediction_response: Union[
            ModelResponse, EmbeddingResponse, ImageResponse
        ] = ModelResponse()
        if vertex_image_generation_class.is_image_generation_response(
            _json_response
        ):
            litellm_prediction_response = (
                vertex_image_generation_class.process_image_generation_response(
                    _json_response,
                    model_response=litellm.ImageResponse(),
                    model=model,
                )
            )

            logging_obj.call_type = (
                PassthroughCallTypes.passthrough_image_generation.value
            )
        elif VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
            json_response=_json_response,
        ):
            # Use multimodal embedding transformation
            vertex_multimodal_config = VertexAIMultimodalEmbeddingConfig()
            litellm_prediction_response = (
                vertex_multimodal_config.transform_embedding_response(
                    model=model,
                    raw_response=httpx_response,
                    model_response=litellm.EmbeddingResponse(),
                    logging_obj=logging_obj,
                    api_key="",
                    request_data={},
                    optional_params={},
                    litellm_params={},
                )
            )
        else:
            litellm_prediction_response = litellm.vertexAITextEmbeddingConfig.transform_vertex_response_to_openai(
                response=_json_response,
                model=model,
                model_response=litellm.EmbeddingResponse(),
            )
        if isinstance(litellm_prediction_response, litellm.EmbeddingResponse):
            litellm_prediction_response.model = model

        logging_obj.model = model
        logging_obj.model_call_details["model"] = logging_obj.model
        logging_obj.model_call_details["custom_llm_provider"] = "vertex_ai"
        logging_obj.custom_llm_provider = "vertex_ai"
        response_cost = litellm.completion_cost(
            completion_response=litellm_prediction_response,
            model=model,
            custom_llm_provider="vertex_ai",
        )

        kwargs["response_cost"] = response_cost
        kwargs["model"] = model
        kwargs["custom_llm_provider"] = "vertex_ai"
        logging_obj.model_call_details["response_cost"] = response_cost

        return {
            "result": litellm_prediction_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _handle_logging_vertex_collected_chunks(
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
        Takes raw chunks from Vertex passthrough endpoint and logs them in litellm callbacks

        - Builds complete response from chunks
        - Creates standard logging object
        - Logs in litellm callbacks
        """
        kwargs: Dict[str, Any] = {}
        model = model or VertexPassthroughLoggingHandler.extract_model_from_url(
            url_route
        )
        complete_streaming_response = (
            VertexPassthroughLoggingHandler._build_complete_streaming_response(
                all_chunks=all_chunks,
                litellm_logging_obj=litellm_logging_obj,
                model=model,
                url_route=url_route,
            )
        )

        if complete_streaming_response is None:
            verbose_proxy_logger.error(
                "Unable to build complete streaming response for Vertex passthrough endpoint, not logging..."
            )
            return {
                "result": None,
                "kwargs": kwargs,
            }

        kwargs = VertexPassthroughLoggingHandler._create_vertex_response_logging_payload_for_generate_content(
            litellm_model_response=complete_streaming_response,
            model=model,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=litellm_logging_obj,
            custom_llm_provider=VertexPassthroughLoggingHandler._get_custom_llm_provider_from_url(
                url_route
            ),
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
            vertex_iterator: Any = VertexModelResponseIterator(
                streaming_response=None,
                sync_stream=False,
                logging_obj=litellm_logging_obj,
            )
            chunk_parsing_logic: Any = vertex_iterator._common_chunk_parsing_logic
            parsed_chunks = [chunk_parsing_logic(chunk) for chunk in all_chunks]
        elif "rawPredict" in url_route or "streamRawPredict" in url_route:
            from litellm.llms.anthropic.chat.handler import ModelResponseIterator
            from litellm.llms.base_llm.base_model_iterator import (
                BaseModelResponseIterator,
            )

            vertex_iterator = ModelResponseIterator(
                streaming_response=None,
                sync_stream=False,
            )
            chunk_parsing_logic = vertex_iterator.chunk_parser
            for chunk in all_chunks:
                dict_chunk = BaseModelResponseIterator._string_to_dict_parser(chunk)
                if dict_chunk is None:
                    continue
                parsed_chunks.append(chunk_parsing_logic(dict_chunk))
        else:
            return None
        if len(parsed_chunks) == 0:
            return None
        all_openai_chunks = []
        for parsed_chunk in parsed_chunks:
            if parsed_chunk is None:
                continue
            all_openai_chunks.append(parsed_chunk)

        complete_streaming_response = litellm.stream_chunk_builder(
            chunks=all_openai_chunks
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
    def extract_model_name_from_vertex_path(vertex_model_path: str) -> str:
        """
        Extract the actual model name from a Vertex AI model path.
        
        Examples:
        - publishers/google/models/gemini-2.5-flash -> gemini-2.5-flash
        - projects/PROJECT_ID/locations/LOCATION/models/MODEL_ID -> MODEL_ID
        
        Args:
            vertex_model_path: The full Vertex AI model path
            
        Returns:
            The extracted model name for use with LiteLLM
        """
        # Handle publishers/google/models/ format
        if "publishers/" in vertex_model_path and "models/" in vertex_model_path:
            # Extract everything after the last models/
            parts = vertex_model_path.split("models/")
            if len(parts) > 1:
                return parts[-1]
        
        # Handle projects/PROJECT_ID/locations/LOCATION/models/MODEL_ID format
        elif "projects/" in vertex_model_path and "models/" in vertex_model_path:
            # Extract everything after the last models/
            parts = vertex_model_path.split("models/")
            if len(parts) > 1:
                return parts[-1]
        
        # If no recognized pattern, return the original path
        return vertex_model_path

    @staticmethod
    def _get_vertex_publisher_or_api_spec_from_url(url: str) -> Optional[str]:
        # Check for specific Vertex AI partner publishers
        if "/publishers/mistralai/" in url:
            return "mistralai"
        elif "/publishers/anthropic/" in url:
            return "anthropic"
        elif "/publishers/ai21/" in url:
            return "ai21"
        elif "/endpoints/openapi/" in url:
            return "openapi"
        return None

    @staticmethod
    def _get_custom_llm_provider_from_url(url: str) -> str:
        parsed_url = urlparse(url)
        if parsed_url.hostname and parsed_url.hostname.endswith(
            "generativelanguage.googleapis.com"
        ):
            return litellm.LlmProviders.GEMINI.value
        return litellm.LlmProviders.VERTEX_AI.value

    @staticmethod
    def _is_multimodal_embedding_response(json_response: dict) -> bool:
        """
        Detect if the response is from a multimodal embedding request.

        Check if the response contains multimodal embedding fields:
            - Docs: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api#response-body


        Args:
            json_response: The JSON response from Vertex AI

        Returns:
            bool: True if this is a multimodal embedding response
        """
        # Check if response contains multimodal embedding fields
        if "predictions" in json_response:
            predictions = json_response["predictions"]
            for prediction in predictions:
                if isinstance(prediction, dict):
                    # Check for multimodal embedding response fields
                    if any(
                        key in prediction
                        for key in [
                            "textEmbedding",
                            "imageEmbedding",
                            "videoEmbeddings",
                        ]
                    ):
                        return True

        return False

    @staticmethod
    def _create_vertex_response_logging_payload_for_generate_content(
        litellm_model_response: Union[ModelResponse, TextCompletionResponse],
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str,
    ) -> dict:
        """
        Create the standard logging object for Vertex passthrough generateContent (streaming and non-streaming)

        """

        response_cost = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
            custom_llm_provider="vertex_ai",
        )

        kwargs["response_cost"] = response_cost
        kwargs["model"] = model

        # pretty print standard logging object
        verbose_proxy_logger.debug("kwargs= %s", kwargs)

        # set litellm_call_id to logging response object
        litellm_model_response.id = logging_obj.litellm_call_id
        logging_obj.model = litellm_model_response.model or model
        logging_obj.model_call_details["model"] = logging_obj.model
        logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
        return kwargs

    @staticmethod
    def batch_prediction_jobs_handler(  # noqa: PLR0915
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
        Handle batch prediction jobs passthrough logging.
        Creates a managed object for cost tracking when batch job is successfully created.
        """
        import base64

        from litellm._uuid import uuid
        from litellm.llms.vertex_ai.batches.transformation import (
            VertexAIBatchTransformation,
        )

        try:
            _json_response = httpx_response.json()
            
            # Only handle successful batch job creation (POST requests)
            if httpx_response.status_code == 200 and "name" in _json_response:
                # Transform Vertex AI response to LiteLLM batch format
                litellm_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
                    response=_json_response
                )
                
                # Extract batch ID and model from the response
                batch_id = VertexAIBatchTransformation._get_batch_id_from_vertex_ai_batch_response(_json_response)
                model_name = _json_response.get("model", "unknown")
                
                # Create unified object ID for tracking
                # Format: base64(litellm_proxy;model_id:{};llm_batch_id:{})
                actual_model_id = VertexPassthroughLoggingHandler.get_actual_model_id_from_router(model_name)

                unified_id_string = SpecialEnums.LITELLM_MANAGED_BATCH_COMPLETE_STR.value.format(actual_model_id, batch_id)
                unified_object_id = base64.urlsafe_b64encode(unified_id_string.encode()).decode().rstrip("=")
                
                # Store the managed object for cost tracking
                # This will be picked up by check_batch_cost polling mechanism
                VertexPassthroughLoggingHandler._store_batch_managed_object(
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
                litellm_model_response.object = "batch_prediction_job"
                litellm_model_response.created = int(start_time.timestamp())
                
                # Add batch-specific metadata to indicate this is a pending batch job
                litellm_model_response.choices = [Choices(
                    finish_reason="stop",
                    index=0,
                    message={
                        "role": "assistant",
                        "content": f"Batch prediction job {batch_id} created and is pending. Status will be updated when the batch completes.",
                        "tool_calls": None,
                        "function_call": None,
                        "provider_specific_fields": {
                            "batch_job_id": batch_id,
                            "batch_job_state": "JOB_STATE_PENDING",
                            "unified_object_id": unified_object_id
                        }
                    }
                )]
                
                # Set response cost to 0 initially (will be updated when batch completes)
                response_cost = 0.0
                kwargs["response_cost"] = response_cost
                kwargs["model"] = model_name
                kwargs["batch_id"] = batch_id
                kwargs["unified_object_id"] = unified_object_id
                kwargs["batch_job_state"] = "JOB_STATE_PENDING"
                
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
                litellm_model_response.model = "vertex_ai_batch"
                litellm_model_response.object = "batch_prediction_job"
                litellm_model_response.created = int(start_time.timestamp())
                
                # Add error-specific metadata
                litellm_model_response.choices = [Choices(
                    finish_reason="stop",
                    index=0,
                    message={
                        "role": "assistant",
                        "content": f"Batch prediction job creation failed. Status: {httpx_response.status_code}",
                        "tool_calls": None,
                        "function_call": None,
                        "provider_specific_fields": {
                            "batch_job_state": "JOB_STATE_FAILED",
                            "status_code": httpx_response.status_code
                        }
                    }
                )]
                
                kwargs["response_cost"] = 0.0
                kwargs["model"] = "vertex_ai_batch"
                kwargs["batch_job_state"] = "JOB_STATE_FAILED"
                
                return {
                    "result": litellm_model_response,
                    "kwargs": kwargs,
                }
                
        except Exception as e:
            verbose_proxy_logger.error(f"Error in batch_prediction_jobs_handler: {e}")
            # Return basic response on error
            litellm_model_response = ModelResponse()
            litellm_model_response.id = str(uuid.uuid4())
            litellm_model_response.model = "vertex_ai_batch"
            litellm_model_response.object = "batch_prediction_job"
            litellm_model_response.created = int(start_time.timestamp())
            
            # Add error-specific metadata
            litellm_model_response.choices = [Choices(
                finish_reason="stop",
                index=0,
                message={
                    "role": "assistant",
                    "content": f"Error creating batch prediction job: {str(e)}",
                    "tool_calls": None,
                    "function_call": None,
                    "provider_specific_fields": {
                        "batch_job_state": "JOB_STATE_FAILED",
                        "error": str(e)
                    }
                }
            )]
            
            kwargs["response_cost"] = 0.0
            kwargs["model"] = "vertex_ai_batch"
            kwargs["batch_job_state"] = "JOB_STATE_FAILED"
            
            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }

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
            # This is a bit of a hack, but we need access to the proxy logging system
            from litellm.proxy.proxy_server import proxy_logging_obj
            
            managed_files_hook = proxy_logging_obj.get_proxy_hook("managed_files")
            if managed_files_hook is not None and hasattr(managed_files_hook, 'store_unified_object_id'):
                # Create a mock user API key dict for the managed object storage
                from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
                user_api_key_dict = UserAPIKeyAuth(
                    user_id=kwargs.get("user_id", "default-user"),
                    api_key="",
                    team_id=None,
                    team_alias=None,
                    user_role=LitellmUserRoles.CUSTOMER,  # Use proper enum value
                    user_email=None,
                    max_budget=None,
                    spend=0.0,  # Set to 0.0 instead of None
                    models=[],  # Set to empty list instead of None
                    tpm_limit=None,
                    rpm_limit=None,
                    budget_duration=None,
                    budget_reset_at=None,
                    max_parallel_requests=None,
                    allowed_model_region=None,
                    metadata={},  # Set to empty dict instead of None
                    key_alias=None,
                    permissions={},  # Set to empty dict instead of None
                    model_max_budget={},  # Set to empty dict instead of None
                    model_spend={},  # Set to empty dict instead of None
                )
                
                # Store the unified object for batch cost tracking
                import asyncio
                asyncio.create_task(
                    managed_files_hook.store_unified_object_id(  # type: ignore
                        unified_object_id=unified_object_id,
                        file_object=batch_object,
                        litellm_parent_otel_span=None,
                        model_object_id=model_object_id,
                        file_purpose="batch",
                        user_api_key_dict=user_api_key_dict,
                    )
                )
                
                verbose_proxy_logger.info(
                    f"Stored batch managed object with unified_object_id={unified_object_id}, batch_id={model_object_id}"
                )
            else:
                verbose_proxy_logger.warning("Managed files hook not available, cannot store batch object for cost tracking")
                
        except Exception as e:
            verbose_proxy_logger.error(f"Error storing batch managed object: {e}")

    @staticmethod
    def get_actual_model_id_from_router(model_name: str) -> str:
        from litellm.proxy.proxy_server import llm_router
        
        if llm_router is not None:
            # Try to find the model in the router by the extracted model name
            extracted_model_name = VertexPassthroughLoggingHandler.extract_model_name_from_vertex_path(model_name)
            
            # Use the existing get_model_ids method from router
            model_ids = llm_router.get_model_ids(model_name=extracted_model_name)
            if model_ids and len(model_ids) > 0:
                # Use the first model ID found
                actual_model_id = model_ids[0]
                verbose_proxy_logger.info(f"Found model ID in router: {actual_model_id}")
                return actual_model_id
            else:
                # Fallback to constructed model name
                actual_model_id = extracted_model_name
                verbose_proxy_logger.warning(f"Model not found in router, using constructed name: {actual_model_id}")
                return actual_model_id
        else:
            # Fallback if router is not available
            extracted_model_name = VertexPassthroughLoggingHandler.extract_model_name_from_vertex_path(model_name)
            verbose_proxy_logger.warning(f"Router not available, using constructed model name: {extracted_model_name}")
            return extracted_model_name

