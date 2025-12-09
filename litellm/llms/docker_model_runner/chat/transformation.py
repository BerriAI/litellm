"""
Translates from OpenAI's `/v1/chat/completions` to Docker Model Runner's `/engines/{engine}/v1/chat/completions`

Docker Model Runner API Reference: https://docs.docker.com/ai/model-runner/api-reference/
"""

from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class DockerModelRunnerChatConfig(OpenAIGPTConfig):
    """
    Configuration for Docker Model Runner API.
    
    Docker Model Runner uses URLs in the format: /engines/{engine}/v1/chat/completions
    The engine name (e.g., "llama.cpp") is part of the API endpoint path.
    """

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Docker Model Runner is OpenAI-compatible, so we use standard message transformation.
        """
        messages = handle_messages_with_content_list_to_str_conversion(messages)
        if is_async:
            return super()._transform_messages(
                messages=messages, model=model, is_async=True
            )
        else:
            return super()._transform_messages(
                messages=messages, model=model, is_async=False
            )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get API base and key for Docker Model Runner.
        
        Default API base: http://localhost:22088/engines/llama.cpp
        The engine path should be included in the api_base.
        """
        api_base = (
            api_base
            or get_secret_str("DOCKER_MODEL_RUNNER_API_BASE")
            or "http://localhost:22088/engines/llama.cpp"
        )  # type: ignore
        # Docker Model Runner may not require authentication for local instances
        dynamic_api_key = api_key or get_secret_str("DOCKER_MODEL_RUNNER_API_KEY") or "dummy-key"
        return api_base, dynamic_api_key

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
        Build the complete URL for Docker Model Runner API.
        
        Docker Model Runner uses URLs in the format: /engines/{engine}/v1/chat/completions
        
        The engine name should be specified in the api_base:
            - api_base="http://model-runner.docker.internal/engines/llama.cpp"
            - Default: "http://localhost:22088/engines/llama.cpp"
        
        Args:
            api_base: Base URL for the Docker Model Runner instance including engine path
            api_key: API key (may not be required for local instances)
            model: Model name (e.g., "llama-3.1")
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            stream: Whether streaming is enabled
            
        Returns:
            Complete URL for the API call
        """
        if not api_base:
            api_base = "http://localhost:22088/engines/llama.cpp"
        
        # Remove trailing slashes from api_base
        api_base = api_base.rstrip("/")
 
        # Build the URL: {api_base}/v1/chat/completions
        # api_base is expected to already contain the engine path
        complete_url = f"{api_base}/v1/chat/completions"
        
        return complete_url

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for Docker Model Runner.
        
        Docker Model Runner is OpenAI-compatible and supports standard parameters.
        """
        return super().get_supported_openai_params(model=model)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Docker Model Runner parameters.
        
        Docker Model Runner is OpenAI-compatible, so most parameters map directly.
        """
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value

        return optional_params

