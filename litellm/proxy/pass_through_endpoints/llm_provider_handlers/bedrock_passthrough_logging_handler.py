import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional
from urllib.parse import urlparse

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import ModelResponse, Usage

if TYPE_CHECKING:
    from ..success_handler import PassThroughEndpointLogging
    from ..types import EndpointType
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
        Transforms Bedrock response to LiteLLM response format, generates a standard logging object 
        so downstream logging can be handled with cost calculation.
        """
        try:
            # Extract model from kwargs (passed from bedrock_proxy_route)
            model = kwargs.get("model", "unknown")
            
            verbose_proxy_logger.debug(
                f"BedrockPassthroughLoggingHandler: Processing response for model {model}"
            )
            
            # Transform Bedrock response to LiteLLM format
            litellm_model_response = BedrockPassthroughLoggingHandler._transform_bedrock_response_to_litellm(
                response_body=response_body,
                model=model,
                url_route=url_route,
            )
            
            # Calculate response cost
            response_cost = BedrockPassthroughLoggingHandler._calculate_bedrock_cost(
                litellm_model_response=litellm_model_response,
                model=model,
            )
            
            # Update kwargs with cost and model information
            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            kwargs["custom_llm_provider"] = "bedrock"
            
            # Get passthrough logging payload and update it
            passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload] = (
                kwargs.get("passthrough_logging_payload")
            )
            if passthrough_logging_payload:
                user = BedrockPassthroughLoggingHandler._get_user_from_metadata(
                    kwargs.get("metadata", {})
                )
                if user:
                    passthrough_logging_payload["user"] = user
                    
                passthrough_logging_payload["model"] = model
                passthrough_logging_payload["custom_llm_provider"] = "bedrock"
                passthrough_logging_payload["response_cost"] = response_cost
            
            verbose_proxy_logger.debug(
                f"BedrockPassthroughLoggingHandler: Calculated cost {response_cost} for model {model}"
            )
            
            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }
            
        except Exception as e:
            verbose_proxy_logger.exception(
                f"BedrockPassthroughLoggingHandler: Error processing response: {str(e)}"
            )
            # Return original values on error
            return {
                "result": None,
                "kwargs": kwargs,
            }
    
    @staticmethod
    def _transform_bedrock_response_to_litellm(
        response_body: dict, 
        model: str,
        url_route: str,
    ) -> ModelResponse:
        """
        Transform Bedrock response format to LiteLLM ModelResponse format.
        Handles different Bedrock model response formats.
        """
        try:
            litellm_response = ModelResponse()
            litellm_response.model = model
            
            # Handle different Bedrock response formats based on model family
            if "anthropic" in model.lower():
                litellm_response = BedrockPassthroughLoggingHandler._transform_anthropic_bedrock_response(
                    response_body, model, litellm_response
                )
            elif "amazon.titan" in model.lower():
                litellm_response = BedrockPassthroughLoggingHandler._transform_titan_bedrock_response(
                    response_body, model, litellm_response
                )
            elif "ai21" in model.lower():
                litellm_response = BedrockPassthroughLoggingHandler._transform_ai21_bedrock_response(
                    response_body, model, litellm_response
                )
            elif "cohere" in model.lower():
                litellm_response = BedrockPassthroughLoggingHandler._transform_cohere_bedrock_response(
                    response_body, model, litellm_response
                )
            elif "meta.llama" in model.lower():
                litellm_response = BedrockPassthroughLoggingHandler._transform_llama_bedrock_response(
                    response_body, model, litellm_response
                )
            else:
                # Generic transformation for unknown models
                litellm_response = BedrockPassthroughLoggingHandler._transform_generic_bedrock_response(
                    response_body, model, litellm_response
                )
                
            return litellm_response
            
        except Exception as e:
            verbose_proxy_logger.exception(
                f"BedrockPassthroughLoggingHandler: Error transforming response: {str(e)}"
            )
            # Return minimal response on error
            return ModelResponse(model=model)
    
    @staticmethod
    def _transform_anthropic_bedrock_response(
        response_body: dict, 
        model: str, 
        litellm_response: ModelResponse
    ) -> ModelResponse:
        """Transform Anthropic Claude responses from Bedrock"""
        from litellm.types.utils import Choices, Message
        
        # Anthropic responses typically have 'content' array
        content = response_body.get("content", [])
        if content and len(content) > 0:
            message_content = content[0].get("text", "")
        else:
            message_content = response_body.get("completion", "")
        
        message = Message(content=message_content, role="assistant")
        choice = Choices(
            message=message,
            index=0,
            finish_reason=response_body.get("stop_reason", "stop")
        )
        litellm_response.choices = [choice]
        
        # Extract usage information
        usage_data = response_body.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)
        )
        litellm_response = ModelResponse(
            id=litellm_response.id,
            choices=litellm_response.choices,
            created=litellm_response.created,
            model=litellm_response.model,
            object=litellm_response.object,
            usage=usage
        )
        
        return litellm_response
    
    @staticmethod
    def _transform_titan_bedrock_response(
        response_body: dict, 
        model: str, 
        litellm_response: ModelResponse
    ) -> ModelResponse:
        """Transform Amazon Titan responses from Bedrock"""
        from litellm.types.utils import Choices, Message
        
        # Titan responses have 'results' array
        results = response_body.get("results", [])
        if results and len(results) > 0:
            message_content = results[0].get("outputText", "")
        else:
            message_content = response_body.get("outputText", "")
        
        message = Message(content=message_content, role="assistant")
        choice = Choices(
            message=message,
            index=0,
            finish_reason=response_body.get("completionReason", "stop")
        )
        litellm_response.choices = [choice]
        
        # Extract usage information
        usage_data = response_body.get("inputTextTokenCount", 0)
        output_tokens = response_body.get("results", [{}])[0].get("tokenCount", 0) if response_body.get("results") else 0
        
        usage = Usage(
            prompt_tokens=usage_data,
            completion_tokens=output_tokens,
            total_tokens=usage_data + output_tokens
        )
        litellm_response = ModelResponse(
            id=litellm_response.id,
            choices=litellm_response.choices,
            created=litellm_response.created,
            model=litellm_response.model,
            object=litellm_response.object,
            usage=usage
        )
        
        return litellm_response
    
    @staticmethod
    def _transform_ai21_bedrock_response(
        response_body: dict, 
        model: str, 
        litellm_response: ModelResponse
    ) -> ModelResponse:
        """Transform AI21 responses from Bedrock"""
        from litellm.types.utils import Choices, Message
        
        # AI21 responses have 'completions' array
        completions = response_body.get("completions", [])
        if completions and len(completions) > 0:
            message_content = completions[0].get("data", {}).get("text", "")
        else:
            message_content = ""
        
        message = Message(content=message_content, role="assistant")
        choice = Choices(
            message=message,
            index=0,
            finish_reason=response_body.get("finishReason", {}).get("reason", "stop")
        )
        litellm_response.choices = [choice]
        
        # AI21 doesn't always provide detailed usage info
        usage = Usage(
            prompt_tokens=0,  # Not typically provided
            completion_tokens=0,  # Not typically provided
            total_tokens=0
        )
        litellm_response = ModelResponse(
            id=litellm_response.id,
            choices=litellm_response.choices,
            created=litellm_response.created,
            model=litellm_response.model,
            object=litellm_response.object,
            usage=usage
        )
        
        return litellm_response
    
    @staticmethod
    def _transform_cohere_bedrock_response(
        response_body: dict, 
        model: str, 
        litellm_response: ModelResponse
    ) -> ModelResponse:
        """Transform Cohere responses from Bedrock"""
        from litellm.types.utils import Choices, Message
        
        # Cohere responses have 'generations' array
        generations = response_body.get("generations", [])
        if generations and len(generations) > 0:
            message_content = generations[0].get("text", "")
        else:
            message_content = response_body.get("text", "")
        
        message = Message(content=message_content, role="assistant")
        choice = Choices(
            message=message,
            index=0,
            finish_reason=response_body.get("finish_reason", "COMPLETE")
        )
        litellm_response.choices = [choice]
        
        # Cohere usage information
        prompt_tokens = response_body.get("prompt", {}).get("tokens", 0) if response_body.get("prompt") else 0
        completion_tokens = len(message_content.split()) if message_content else 0  # Rough estimate
        
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        litellm_response = ModelResponse(
            id=litellm_response.id,
            choices=litellm_response.choices,
            created=litellm_response.created,
            model=litellm_response.model,
            object=litellm_response.object,
            usage=usage
        )
        
        return litellm_response
    
    @staticmethod
    def _transform_llama_bedrock_response(
        response_body: dict, 
        model: str, 
        litellm_response: ModelResponse
    ) -> ModelResponse:
        """Transform Meta Llama responses from Bedrock"""
        from litellm.types.utils import Choices, Message
        
        # Llama responses typically have 'generation'
        message_content = response_body.get("generation", "")
        if not message_content:
            # Try alternative response format
            message_content = response_body.get("outputs", [{}])[0].get("text", "") if response_body.get("outputs") else ""
        
        message = Message(content=message_content, role="assistant")
        choice = Choices(
            message=message,
            index=0,
            finish_reason=response_body.get("stop_reason", "stop")
        )
        litellm_response.choices = [choice]
        
        # Llama usage information
        prompt_tokens = response_body.get("prompt_token_count", 0)
        completion_tokens = response_body.get("generation_token_count", 0)
        
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        litellm_response = ModelResponse(
            id=litellm_response.id,
            choices=litellm_response.choices,
            created=litellm_response.created,
            model=litellm_response.model,
            object=litellm_response.object,
            usage=usage
        )
        
        return litellm_response
    
    @staticmethod
    def _transform_generic_bedrock_response(
        response_body: dict, 
        model: str, 
        litellm_response: ModelResponse
    ) -> ModelResponse:
        """Generic transformation for unknown Bedrock model responses"""
        from litellm.types.utils import Choices, Message
        
        # Try common response fields
        message_content = (
            response_body.get("text", "") or
            response_body.get("generated_text", "") or
            response_body.get("completion", "") or
            response_body.get("response", "") or
            str(response_body)
        )
        
        message = Message(content=message_content, role="assistant")
        choice = Choices(
            message=message,
            index=0,
            finish_reason="stop"
        )
        litellm_response.choices = [choice]
        
        # Generic usage (minimal)
        completion_tokens = len(message_content.split()) if message_content else 0
        usage = Usage(
            prompt_tokens=0,
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens
        )
        litellm_response = ModelResponse(
            id=litellm_response.id,
            choices=litellm_response.choices,
            created=litellm_response.created,
            model=litellm_response.model,
            object=litellm_response.object,
            usage=usage
        )
        
        return litellm_response
    
    @staticmethod
    def _calculate_bedrock_cost(
        litellm_model_response: ModelResponse, 
        model: str
    ) -> Optional[float]:
        """Calculate cost for Bedrock response using LiteLLM's cost calculation"""
        try:
            response_cost = litellm.completion_cost(
                completion_response=litellm_model_response,
                model=model,
            )
            return response_cost
        except Exception as e:
            verbose_proxy_logger.debug(
                f"BedrockPassthroughLoggingHandler: Error calculating cost for model {model}: {str(e)}"
            )
            return None
    
    @staticmethod
    def _get_user_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
        """Extract user information from metadata"""
        return metadata.get("user_api_key_user_id") or metadata.get("user")
    
    @staticmethod
    def _should_log_request(url_route: str) -> bool:
        """Determine if this Bedrock request should be logged"""
        # Log all Bedrock runtime requests
        parsed_url = urlparse(url_route)
        hostname = parsed_url.hostname or ""
        if "bedrock-runtime" in hostname or "bedrock-agent-runtime" in hostname:
            return True
        return False