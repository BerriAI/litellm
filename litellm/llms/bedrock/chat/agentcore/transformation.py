"""
Transformation for Bedrock AgentCore

https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agentcore_InvokeAgentRuntime.html
"""

import json
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import httpx

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonAgentCoreConfig(BaseConfig, BaseAWSLLM):
    def __init__(self, **kwargs):
        BaseConfig.__init__(self, **kwargs)
        BaseAWSLLM.__init__(self, **kwargs)

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Bedrock AgentCore has 0 OpenAI compatible params
        """
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to AgentCore params
        """
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete url for the request
        """
        ### SET RUNTIME ENDPOINT ###
        aws_bedrock_runtime_endpoint = optional_params.get(
            "aws_bedrock_runtime_endpoint", None
        )
        
        # Extract ARN from model string
        agent_runtime_arn = self._get_agent_runtime_arn(model)
        
        # Parse ARN to get region
        region = self._extract_region_from_arn(agent_runtime_arn)
        
        endpoint_url, _ = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            aws_region_name=region,
            endpoint_type="agentcore",
        )

        # AgentCore uses a different endpoint structure
        # The actual endpoint will be constructed based on the service
        # For now, we'll use the base endpoint and let AWS SDK handle the routing
        endpoint_url = f"{endpoint_url}/invoke-agent-runtime"

        return endpoint_url

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        return self._sign_request(
            service_name="bedrock-agentcore",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
            api_key=api_key,
        )

    def _get_agent_runtime_arn(self, model: str) -> str:
        """
        Extract ARN from model string
        model = "agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC"
        returns: "arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC"
        """
        parts = model.split("/", 1)
        if len(parts) != 2 or parts[0] != "agentcore":
            raise ValueError(
                "Invalid model format. Expected format: 'model=bedrock/agentcore/arn:aws:bedrock-agentcore:region:account:runtime/runtime_id'"
            )
        return parts[1]

    def _extract_region_from_arn(self, arn: str) -> str:
        """
        Extract region from ARN
        arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC
        returns: us-west-2
        """
        parts = arn.split(":")
        if len(parts) >= 4:
            return parts[3]
        raise ValueError(f"Invalid ARN format: {arn}")

    def _get_runtime_session_id(self, optional_params: dict) -> str:
        """
        Get or generate runtime session ID (must be 33+ chars)
        """
        session_id = optional_params.get("runtimeSessionId", None)
        if session_id:
            return session_id
        
        # Generate a session ID with 33+ characters
        return f"litellm-session-{str(uuid.uuid4())}"

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request to AgentCore format
        """
        # Use the last message content as the prompt
        prompt = convert_content_list_to_str(messages[-1])
        
        # Create the payload
        payload_dict = {"prompt": prompt}
        payload_json = json.dumps(payload_dict)
        
        # Get agent runtime ARN
        agent_runtime_arn = self._get_agent_runtime_arn(model)
        
        # Get or generate session ID
        runtime_session_id = self._get_runtime_session_id(optional_params)
        
        request_data = {
            "agentRuntimeArn": agent_runtime_arn,
            "runtimeSessionId": runtime_session_id,
            "payload": payload_json,
        }
        
        # Add optional qualifier if provided
        if "qualifier" in optional_params:
            request_data["qualifier"] = optional_params["qualifier"]
        
        return request_data

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform the AgentCore response to LiteLLM ModelResponse format
        """
        try:
            # Parse the response
            response_data = raw_response.json()
            verbose_logger.debug(f"AgentCore response data: {response_data}")
            
            # Extract the response content
            # The actual structure may vary based on AgentCore API
            # Adjust this based on actual response format
            content = ""
            if "response" in response_data:
                if isinstance(response_data["response"], dict):
                    content = response_data["response"].get("text", "")
                else:
                    content = str(response_data["response"])
            else:
                content = str(response_data)
            
            # Create the message
            message = Message(content=content, role="assistant")
            
            # Create choices
            choice = Choices(finish_reason="stop", index=0, message=message)
            
            # Update model response
            model_response.choices = [choice]
            model_response.model = model
            
            # Add usage information if available
            if "usage" in response_data:
                usage = Usage(
                    prompt_tokens=response_data["usage"].get("inputTokens", 0),
                    completion_tokens=response_data["usage"].get("outputTokens", 0),
                    total_tokens=response_data["usage"].get("inputTokens", 0)
                    + response_data["usage"].get("outputTokens", 0),
                )
                setattr(model_response, "usage", usage)
            
            return model_response

        except Exception as e:
            verbose_logger.error(
                f"Error processing Bedrock AgentCore response: {str(e)}"
            )
            raise BedrockError(
                message=f"Error processing response: {str(e)}",
                status_code=raw_response.status_code,
            )

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
        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message)

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        return True

