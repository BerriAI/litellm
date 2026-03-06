"""
Transformation for Bedrock Moonshot AI (Kimi K2) models.

Supports the Kimi K2 Thinking model available on Amazon Bedrock.
Model format: bedrock/moonshot.kimi-k2-thinking-v1:0

Reference: https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-bedrock-fully-managed-open-weight-models/
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union
import re

import httpx

from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.moonshot.chat.transformation import MoonshotChatConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.types.utils import ModelResponse

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonMoonshotConfig(AmazonInvokeConfig, MoonshotChatConfig):
    """
    Configuration for Bedrock Moonshot AI (Kimi K2) models.
    
    Reference:
        https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-bedrock-fully-managed-open-weight-models/
        https://platform.moonshot.ai/docs/api/chat
    
    Supported Params for the Amazon / Moonshot models:
    - `max_tokens` (integer) max tokens
    - `temperature` (float) temperature for model (0-1 for Moonshot)
    - `top_p` (float) top p for model
    - `stream` (bool) whether to stream responses
    - `tools` (list) tool definitions (supported on kimi-k2-thinking)
    - `tool_choice` (str|dict) tool choice specification (supported on kimi-k2-thinking)
    
    NOT Supported on Bedrock:
    - `stop` sequences (Bedrock doesn't support stopSequences field for this model)
    
    Note: The kimi-k2-thinking model DOES support tool calls, unlike kimi-thinking-preview.
    """

    def __init__(self, **kwargs):
        AmazonInvokeConfig.__init__(self, **kwargs)
        MoonshotChatConfig.__init__(self, **kwargs)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "bedrock"

    def _get_model_id(self, model: str) -> str:
        """
        Extract the actual model ID from the LiteLLM model name.
        
        Removes routing prefixes like:
        - bedrock/invoke/moonshot.kimi-k2-thinking -> moonshot.kimi-k2-thinking
        - invoke/moonshot.kimi-k2-thinking -> moonshot.kimi-k2-thinking
        - moonshot.kimi-k2-thinking -> moonshot.kimi-k2-thinking
        """
        # Remove bedrock/ prefix if present
        if model.startswith("bedrock/"):
            model = model[8:]
        
        # Remove invoke/ prefix if present
        if model.startswith("invoke/"):
            model = model[7:]
        
        # Remove any provider prefix (e.g., moonshot/)
        if "/" in model and not model.startswith("arn:"):
            parts = model.split("/", 1)
            if len(parts) == 2:
                model = parts[1]
        
        return model

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Get the supported OpenAI params for Moonshot AI models on Bedrock.
        
        Bedrock-specific limitations:
        - stopSequences field is not supported on Bedrock (unlike native Moonshot API)
        - functions parameter is not supported (use tools instead)
        - tool_choice doesn't support "required" value
        
        Note: kimi-k2-thinking DOES support tool calls (unlike kimi-thinking-preview)
        The parent MoonshotChatConfig class handles the kimi-thinking-preview exclusion.
        """
        excluded_params: List[str] = ["functions", "stop"]  # Bedrock doesn't support stopSequences
        
        base_openai_params = super(MoonshotChatConfig, self).get_supported_openai_params(model=model)
        final_params: List[str] = []
        for param in base_openai_params:
            if param not in excluded_params:
                final_params.append(param)
        
        return final_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Moonshot AI parameters for Bedrock.
        
        Handles Moonshot AI specific limitations:
        - tool_choice doesn't support "required" value
        - Temperature <0.3 limitation for n>1
        - Temperature range is [0, 1] (not [0, 2] like OpenAI)
        """
        return MoonshotChatConfig.map_openai_params(
            self,
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request for Bedrock Moonshot AI models.
        
        Uses the Moonshot transformation logic which handles:
        - Converting content lists to strings (Moonshot doesn't support list format)
        - Adding tool_choice="required" message if needed
        - Temperature and parameter validation

        """
        # Filter out AWS credentials using the existing method from BaseAWSLLM
        self._get_boto_credentials_from_optional_params(optional_params, model)
        
        # Strip routing prefixes to get the actual model ID
        clean_model_id = self._get_model_id(model)
        
        # Use Moonshot's transform_request which handles message transformation
        # and tool_choice="required" workaround
        return MoonshotChatConfig.transform_request(
            self,
            model=clean_model_id,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def _extract_reasoning_from_content(self, content: str) -> tuple[Optional[str], str]:
        """
        Extract reasoning content from <reasoning> tags in the response.
        
        Moonshot AI's Kimi K2 Thinking model returns reasoning in <reasoning> tags.
        This method extracts that content and returns it separately.
        
        Args:
            content: The full content string from the API response
            
        Returns:
            tuple: (reasoning_content, main_content)
        """
        if not content:
            return None, content
        
        # Match <reasoning>...</reasoning> tags
        reasoning_match = re.match(
            r"<reasoning>(.*?)</reasoning>\s*(.*)", 
            content, 
            re.DOTALL
        )
        
        if reasoning_match:
            reasoning_content = reasoning_match.group(1).strip()
            main_content = reasoning_match.group(2).strip()
            return reasoning_content, main_content
        
        return None, content

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: "ModelResponse",
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> "ModelResponse":
        """
        Transform the response from Bedrock Moonshot AI models.
        
        Moonshot AI uses OpenAI-compatible response format, but returns reasoning
        content in <reasoning> tags. This method:
        1. Calls parent class transformation
        2. Extracts reasoning content from <reasoning> tags
        3. Sets reasoning_content on the message object
        """
        # First, get the standard transformation
        model_response = MoonshotChatConfig.transform_response(
            self,
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
        
        # Extract reasoning content from <reasoning> tags
        if model_response.choices and len(model_response.choices) > 0:
            for choice in model_response.choices:
                # Only process Choices (not StreamingChoices) which have message attribute
                if isinstance(choice, Choices) and choice.message and choice.message.content:
                    reasoning_content, main_content = self._extract_reasoning_from_content(
                        choice.message.content
                    )
                    
                    if reasoning_content:
                        # Set the reasoning_content field
                        choice.message.reasoning_content = reasoning_content
                        # Update the main content without reasoning tags
                        choice.message.content = main_content
        
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BedrockError:
        """Return the appropriate error class for Bedrock."""
        return BedrockError(status_code=status_code, message=error_message)
