import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, cast

import httpx
from httpx import Headers, Response

from litellm.llms.base_llm.batches.transformation import BaseBatchesConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, CreateBatchRequest
from litellm.types.utils import LiteLLMBatch, LlmProviders, ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class AnthropicBatchesConfig(BaseBatchesConfig):
    def __init__(self):
        from ..chat.transformation import AnthropicConfig
        from ..common_utils import AnthropicModelInfo

        self.anthropic_chat_config = AnthropicConfig()  # initialize once
        self.anthropic_model_info = AnthropicModelInfo()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        """Return the LLM provider type for this configuration."""
        return LlmProviders.ANTHROPIC

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
        """Validate and prepare environment-specific headers and parameters."""
        # Resolve api_key from environment if not provided
        api_key = api_key or self.anthropic_model_info.get_api_key()
        if api_key is None:
            raise ValueError(
                "Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params"
            )
        _headers = {
            "accept": "application/json",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": api_key,
        }
        # Add beta header for message batches
        if "anthropic-beta" not in headers:
            headers["anthropic-beta"] = "message-batches-2024-09-24"
        headers.update(_headers)
        return headers

    def get_complete_batch_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: Dict,
        litellm_params: Dict,
        data: CreateBatchRequest,
    ) -> str:
        """Get the complete URL for batch creation request."""
        api_base = api_base or self.anthropic_model_info.get_api_base(api_base)
        if not api_base.endswith("/v1/messages/batches"):
            api_base = f"{api_base.rstrip('/')}/v1/messages/batches"
        return api_base

    def transform_create_batch_request(
        self,
        model: str,
        create_batch_data: CreateBatchRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> Union[bytes, str, Dict[str, Any]]:
        """
        Transform the batch creation request to Anthropic format.
        
        Not currently implemented - placeholder to satisfy abstract base class.
        """
        raise NotImplementedError("Batch creation not yet implemented for Anthropic")

    def transform_create_batch_response(
        self,
        model: Optional[str],
        raw_response: httpx.Response,
        logging_obj: LoggingClass,
        litellm_params: dict,
    ) -> LiteLLMBatch:
        """
        Transform Anthropic MessageBatch creation response to LiteLLM format.
        
        Not currently implemented - placeholder to satisfy abstract base class.
        """
        raise NotImplementedError("Batch creation not yet implemented for Anthropic")

    def get_retrieve_batch_url(
        self,
        api_base: Optional[str],
        batch_id: str,
        optional_params: Dict,
        litellm_params: Dict,
    ) -> str:
        """
        Get the complete URL for batch retrieval request.
        
        Args:
            api_base: Base API URL (optional, will use default if not provided)
            batch_id: Batch ID to retrieve
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            
        Returns:
            Complete URL for Anthropic batch retrieval: {api_base}/v1/messages/batches/{batch_id}
        """
        api_base = api_base or self.anthropic_model_info.get_api_base(api_base)
        return f"{api_base.rstrip('/')}/v1/messages/batches/{batch_id}"

    def transform_retrieve_batch_request(
        self,
        batch_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> Union[bytes, str, Dict[str, Any]]:
        """
        Transform batch retrieval request for Anthropic.
        
        For Anthropic, the URL is constructed by get_retrieve_batch_url(),
        so this method returns an empty dict (no additional request params needed).
        """
        # No additional request params needed - URL is handled by get_retrieve_batch_url
        return {}

    def transform_retrieve_batch_response(
        self,
        model: Optional[str],
        raw_response: httpx.Response,
        logging_obj: LoggingClass,
        litellm_params: dict,
    ) -> LiteLLMBatch:
        """Transform Anthropic MessageBatch retrieval response to LiteLLM format."""
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise ValueError(f"Failed to parse Anthropic batch response: {e}")

        # Map Anthropic MessageBatch to OpenAI Batch format
        batch_id = response_data.get("id", "")
        processing_status = response_data.get("processing_status", "in_progress")
        
        # Map Anthropic processing_status to OpenAI status
        status_mapping: Dict[str, Literal["validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"]] = {
            "in_progress": "in_progress",
            "canceling": "cancelling",
            "ended": "completed",
        }
        openai_status = status_mapping.get(processing_status, "in_progress")

        # Parse timestamps
        def parse_timestamp(ts_str: Optional[str]) -> Optional[int]:
            if not ts_str:
                return None
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                return int(dt.timestamp())
            except Exception:
                return None

        created_at = parse_timestamp(response_data.get("created_at"))
        ended_at = parse_timestamp(response_data.get("ended_at"))
        expires_at = parse_timestamp(response_data.get("expires_at"))
        cancel_initiated_at = parse_timestamp(response_data.get("cancel_initiated_at"))
        archived_at = parse_timestamp(response_data.get("archived_at"))

        # Extract request counts
        request_counts_data = response_data.get("request_counts", {})
        from openai.types.batch import BatchRequestCounts
        request_counts = BatchRequestCounts(
            total=sum([
                request_counts_data.get("processing", 0),
                request_counts_data.get("succeeded", 0),
                request_counts_data.get("errored", 0),
                request_counts_data.get("canceled", 0),
                request_counts_data.get("expired", 0),
            ]),
            completed=request_counts_data.get("succeeded", 0),
            failed=request_counts_data.get("errored", 0),
        )

        return LiteLLMBatch(
            id=batch_id,
            object="batch",
            endpoint="/v1/messages",
            errors=None,
            input_file_id="None",
            completion_window="24h",
            status=openai_status,
            output_file_id=batch_id,
            error_file_id=None,
            created_at=created_at or int(time.time()),
            in_progress_at=created_at if processing_status == "in_progress" else None,
            expires_at=expires_at,
            finalizing_at=None,
            completed_at=ended_at if processing_status == "ended" else None,
            failed_at=None,
            expired_at=archived_at if archived_at else None,
            cancelling_at=cancel_initiated_at if processing_status == "canceling" else None,
            cancelled_at=ended_at if processing_status == "canceling" and ended_at else None,
            request_counts=request_counts,
            metadata={},
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, Headers]
    ) -> "BaseLLMException":
        """Get the appropriate error class for Anthropic."""
        from ..common_utils import AnthropicError

        # Convert Dict to Headers if needed
        if isinstance(headers, dict):
            headers_obj: Optional[Headers] = Headers(headers)
        else:
            headers_obj = headers if isinstance(headers, Headers) else None

        return AnthropicError(status_code=status_code, message=error_message, headers=headers_obj)

    def transform_response(
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        from litellm.cost_calculator import BaseTokenUsageProcessor
        from litellm.types.utils import Usage

        response_text = raw_response.text.strip()
        all_usage: List[Usage] = []

        try:
            # Split by newlines and try to parse each line as JSON
            lines = response_text.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    response_json = json.loads(line)
                    # Update model_response with the parsed JSON
                    completion_response = response_json["result"]["message"]
                    transformed_response = (
                        self.anthropic_chat_config.transform_parsed_response(
                            completion_response=completion_response,
                            raw_response=raw_response,
                            model_response=model_response,
                        )
                    )

                    transformed_response_usage = getattr(
                        transformed_response, "usage", None
                    )
                    if transformed_response_usage:
                        all_usage.append(cast(Usage, transformed_response_usage))
                except json.JSONDecodeError:
                    continue

            ## SUM ALL USAGE
            combined_usage = BaseTokenUsageProcessor.combine_usage_objects(all_usage)
            setattr(model_response, "usage", combined_usage)

            return model_response
        except Exception as e:
            raise e
