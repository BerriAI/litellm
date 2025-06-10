from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.utils import ProviderConfigManager

if TYPE_CHECKING:
    from ..success_handler import PassThroughEndpointLogging
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


class BedrockPassthroughLoggingHandler:
    @staticmethod
    def bedrock_passthrough_handler(
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
        Transforms Bedrock response to LiteLLM response, generates a standard logging object so downstream logging can be handled
        """
        model = kwargs.get("model", "unknown")
        
        # Get the appropriate Bedrock configuration
        bedrock_config = ProviderConfigManager.get_provider_chat_config(
            model=model, provider=litellm.LlmProviders.BEDROCK
        )
        
        if bedrock_config is None:
            verbose_proxy_logger.error(
                f"No Bedrock configuration found for model {model}"
            )
            return {
                "result": None,
                "kwargs": kwargs,
            }
        
        # Use existing LiteLLM transformation infrastructure
        litellm_model_response = bedrock_config.transform_response(
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

        kwargs = BedrockPassthroughLoggingHandler._create_bedrock_response_logging_payload(
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
    def _create_bedrock_response_logging_payload(
        litellm_model_response: litellm.ModelResponse,
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
    ):
        """
        Create the standard logging object for Bedrock passthrough

        handles streaming and non-streaming responses
        """
        try:
            response_cost = litellm.completion_cost(
                completion_response=litellm_model_response,
                model=model,
            )
            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            kwargs["custom_llm_provider"] = "bedrock"
            
            passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = (
                kwargs.get("passthrough_logging_payload")
            )
            if passthrough_logging_payload:
                user = BedrockPassthroughLoggingHandler._get_user_from_metadata(
                    kwargs.get("metadata", {})
                )
                if user:
                    kwargs.setdefault("litellm_params", {})
                    kwargs["litellm_params"].update(
                        {"proxy_server_request": {"body": {"user": user}}}
                    )

            # set litellm_call_id to logging response object
            litellm_model_response.id = logging_obj.litellm_call_id
            litellm_model_response.model = model
            logging_obj.model_call_details["model"] = model
            logging_obj.model_call_details["custom_llm_provider"] = "bedrock"
            
            return kwargs
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error creating Bedrock response logging payload: %s", e
            )
            return kwargs
    
    @staticmethod
    def _get_user_from_metadata(metadata: dict) -> Optional[str]:
        """Extract user ID from metadata if available"""
        return metadata.get("user")
    
    @staticmethod
    def _should_log_request(kwargs: dict) -> bool:
        """Determine if this request should be logged"""
        return kwargs.get("should_log", True)