import dataclasses
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from attr import dataclass

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse
from ..common_utils import HuggingFaceError, _validate_environment

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


@dataclass
class HFRequestParameters:
    inputs: Union[str, Dict]
    headers: Dict[str, Any]
    base_url: Optional[str] = None
    provider: Optional[str] = None


@dataclass
class HFChatConfig(OpenAIGPTConfig):
    """
    Reference: https://huggingface.co/docs/huggingface_hub/guides/inference
    """

    frequency_penalty: Optional[int] = None
    tool_choice: Optional[Union[str, dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    tools: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None
    logprobs: Optional[int] = None
    tool_prompt: Optional[str] = None
    stream_options: Optional[dict] = None

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        tools: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
        logprobs: Optional[int] = None,
        tool_prompt: Optional[str] = None,
        stream_options: Optional[dict] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    def get_base_url(self, model: str, base_url: Optional[str]) -> Optional[str]:
        """
        Get the API base for the Huggingface API.

        Do not add the chat/embedding/rerank extension here. Let the handler do this.
        """
        if model.startswith(("http://", "https://")):
            base_url = model
        elif base_url is None:
            base_url = os.getenv("HF_API_BASE") or os.getenv("HUGGINGFACE_API_BASE", "")
        return base_url

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return HuggingFaceError(
            status_code=status_code, message=error_message, headers=headers
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        messages = self._transform_messages(messages=messages, model=model)
        return {"model": model, "messages": messages, **optional_params}

    def _prepare_request(
        self,
        model: str,
        messages: list,
        api_base: Optional[str],
        api_key: str,
        headers: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> HFRequestParameters:
        """Prepare request data for custom client usage."""
        headers = _validate_environment(
            api_key=api_key,
            headers=headers,
        )
        if api_base is not None:
            # Use direct completion URL when api_base is provided
            base_url = self.get_base_url(model=model, base_url=api_base)
            provider = None  # Not needed when using completion_url
        else:
            # Parse provider and model for HF inference API
            parts = model.split("/", 1)
            provider = parts[0] if len(parts) > 1 else "hf-inference"
            model = parts[1] if len(parts) > 1 else model
            base_url = None
        inputs = self.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        return HFRequestParameters(
            inputs=inputs,
            headers=headers,
            base_url=base_url,
            provider=provider,
        )

    def transform_stream_chunk(self, chunk: Any) -> ModelResponse:
        """Transform HF's ChatCompletionStreamOutput to ModelResponse"""
        # Create streaming choices
        streaming_choices = []
        for choice in chunk.choices:
            streaming_choices.append(dataclasses.asdict(choice))

        # Create ModelResponse for streaming
        response = ModelResponse(
            id=chunk.id,
            choices=streaming_choices,
            created=chunk.created,
            model=chunk.model,
            system_fingerprint=chunk.system_fingerprint,
            stream=True,
            usage = dataclasses.asdict(chunk.usage) if chunk.usage else None,
            object="chat.completion.chunk",
        )
        response._hidden_params = {
            "created_at": chunk.created,
            "usage": chunk.usage if chunk.usage else None
        }
        return response