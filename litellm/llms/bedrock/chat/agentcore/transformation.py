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
        
        # Build the base endpoint URL for AgentCore
        # Note: We don't use get_runtime_endpoint as AgentCore has its own endpoint structure
        if aws_bedrock_runtime_endpoint:
            base_url = aws_bedrock_runtime_endpoint
        else:
            base_url = f"https://bedrock-agentcore.{region}.amazonaws.com"
        
        # Based on boto3 client.invoke_agent_runtime, the path is:
        # /runtimes/{URL-ENCODED-ARN}/invocations?qualifier=<value>
        from urllib.parse import quote
        encoded_arn = quote(agent_runtime_arn, safe='')
        endpoint_url = f"{base_url}/runtimes/{encoded_arn}/invocations"
        
        # Add qualifier as query parameter if provided
        if "qualifier" in optional_params:
            endpoint_url = f"{endpoint_url}?qualifier={optional_params['qualifier']}"

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
        Transform the request to AgentCore format.
        
        Based on boto3's implementation:
        - Session ID goes in header: X-Amzn-Bedrock-AgentCore-Runtime-Session-Id
        - Qualifier goes as query parameter
        - Only the payload goes in the request body
        """
        # Use the last message content as the prompt
        prompt = convert_content_list_to_str(messages[-1])
        
        # Create the payload - this is what goes in the body (raw JSON)
        payload_dict = {"prompt": prompt}
        
        # Get or generate session ID - this goes in the header
        runtime_session_id = self._get_runtime_session_id(optional_params)
        headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] = runtime_session_id
        
        # The request data is the payload dict (will be JSON encoded by the HTTP handler)
        # Qualifier will be handled as a query parameter in get_complete_url
        
        return payload_dict

    def _parse_sse_stream(self, response_text: str) -> dict:
        """
        Parse Server-Sent Events (SSE) stream format.
        Each line starts with 'data:' followed by JSON.
        """
        lines = response_text.strip().split('\n')
        
        final_message = None
        usage_data = None
        content_blocks = []
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('data:'):
                continue
            
            # Remove 'data:' prefix and parse JSON
            json_str = line[5:].strip()
            if not json_str:
                continue
                
            try:
                data = json.loads(json_str)
                
                # Some lines contain JSON strings instead of JSON objects - skip those
                if not isinstance(data, dict):
                    verbose_logger.debug(f"Skipping non-dict data: {type(data)}")
                    continue
                
                # Check for final message with complete content
                if "message" in data and isinstance(data["message"], dict):
                    final_message = data["message"]
                
                # Check for usage metadata
                if "event" in data and isinstance(data["event"], dict):
                    event = data["event"]
                    if "metadata" in event and "usage" in event["metadata"]:
                        usage_data = event["metadata"]["usage"]
                    
                    # Collect content deltas for building the response
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        if "text" in delta:
                            content_blocks.append(delta["text"])
                
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                verbose_logger.debug(f"Skipping non-JSON line: {line[:100]}")
                continue
        
        # If we have a final message, use that; otherwise build from content blocks
        if final_message and "content" in final_message:
            content_list = final_message["content"]
            if content_list and isinstance(content_list, list):
                # Extract text from content blocks
                content = ""
                for block in content_list:
                    if isinstance(block, dict) and "text" in block:
                        content += block["text"]
            else:
                content = ""
        else:
            # Build content from collected deltas
            content = "".join(content_blocks)
        
        return {
            "content": content,
            "usage": usage_data,
            "final_message": final_message
        }

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
        Transform the AgentCore response to LiteLLM ModelResponse format.
        AgentCore returns responses as SSE (Server-Sent Events) stream.
        """
        try:
            # Parse the SSE stream
            response_text = raw_response.text
            verbose_logger.debug(f"AgentCore response (first 500 chars): {response_text[:500]}")
            
            parsed_data = self._parse_sse_stream(response_text)
            
            content = parsed_data["content"]
            usage_data = parsed_data["usage"]
            
            verbose_logger.debug(f"Parsed content length: {len(content)}")
            verbose_logger.debug(f"Usage data: {usage_data}")
            
            # Create the message
            message = Message(content=content, role="assistant")
            
            # Create choices
            choice = Choices(finish_reason="stop", index=0, message=message)
            
            # Update model response
            model_response.choices = [choice]
            model_response.model = model
            
            # Add usage information if available
            if usage_data:
                usage = Usage(
                    prompt_tokens=usage_data.get("inputTokens", 0),
                    completion_tokens=usage_data.get("outputTokens", 0),
                    total_tokens=usage_data.get("totalTokens", 0),
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

