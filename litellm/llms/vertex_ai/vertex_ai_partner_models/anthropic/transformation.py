# What is this?
## Handler file for calling claude-3 on vertex ai
from typing import Any, Dict, List, Optional

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ....anthropic.chat.transformation import AnthropicConfig


class VertexAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url=" https://cloud.google.com/vertex-ai/"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


def get_anthropic_beta_from_headers(headers: Dict) -> List[str]:
    """
    Extract anthropic-beta header values and convert them to a list.
    Supports comma-separated values from user headers.

    Used by Vertex AI Anthropic transformation for consistent handling
    of anthropic-beta headers that should be passed to Vertex AI.

    Args:
        headers (dict): Request headers dictionary

    Returns:
        List[str]: List of anthropic beta feature strings, empty list if no header
    """
    anthropic_beta_header = headers.get("anthropic-beta")
    if not anthropic_beta_header:
        return []

    # Split comma-separated values and strip whitespace
    return [beta.strip() for beta in anthropic_beta_header.split(",")]


class VertexAIAnthropicConfig(AnthropicConfig):
    """
    Reference:https://docs.anthropic.com/claude/reference/messages_post

    Note that the API for Claude on Vertex differs from the Anthropic API documentation in the following ways:

    - `model` is not a valid parameter. The model is instead specified in the Google Cloud endpoint URL.
    - `anthropic_version` is a required parameter and must be set to "vertex-2023-10-16".

    The class `VertexAIAnthropicConfig` provides configuration for the VertexAI's Anthropic API interface. Below are the parameters:

    - `max_tokens` Required (integer) max tokens,
    - `anthropic_version` Required (string) version of anthropic for bedrock - e.g. "bedrock-2023-05-31"
    - `system` Optional (string) the system prompt, conversion from openai format to this is handled in factory.py
    - `temperature` Optional (float) The amount of randomness injected into the response
    - `top_p` Optional (float) Use nucleus sampling.
    - `top_k` Optional (int) Only sample from the top K options for each subsequent token
    - `stop_sequences` Optional (List[str]) Custom text sequences that cause the model to stop generating

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "vertex_ai"

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        data = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        data.pop("model", None)  # vertex anthropic doesn't accept 'model' parameter
        
        # Handle anthropic_beta from user headers
        anthropic_beta_list = get_anthropic_beta_from_headers(headers)
        
        # Auto-add computer-use beta if computer use tools are present
        tools = data.get("tools", [])
        if tools:
            for tool in tools:
                tool_type = tool.get("type", "")
                if tool_type.startswith("computer_"):
                    # Auto-add the computer-use beta header
                    if "computer-use-2024-10-22" not in anthropic_beta_list:
                        anthropic_beta_list.append("computer-use-2024-10-22")
                    break
        
        # Remove duplicates while preserving order
        if anthropic_beta_list:
            unique_betas = []
            seen = set()
            for beta in anthropic_beta_list:
                if beta not in seen:
                    unique_betas.append(beta)
                    seen.add(beta)
            data["anthropic_beta"] = unique_betas
        
        return data

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
        response = super().transform_response(
            model,
            raw_response,
            model_response,
            logging_obj,
            request_data,
            messages,
            optional_params,
            litellm_params,
            encoding,
            api_key,
            json_mode,
        )
        response.model = model

        return response

    @classmethod
    def is_supported_model(cls, model: str, custom_llm_provider: str) -> bool:
        """
        Check if the model is supported by the VertexAI Anthropic API.
        """
        if (
            custom_llm_provider != "vertex_ai"
            and custom_llm_provider != "vertex_ai_beta"
        ):
            return False
        if "claude" in model.lower():
            return True
        elif model in litellm.vertex_anthropic_models:
            return True
        return False
