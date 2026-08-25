"""
MiniMax OpenAI transformation config - extends OpenAI chat config for MiniMax's OpenAI-compatible API
"""

from typing import (
    TYPE_CHECKING,
    Any,  # noqa: TID251  # LiteLLMLoggingObj has no concrete public type; matches OpenAIGPTConfig's own alias
    Final,
)

import httpx

import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class MinimaxChatConfig(OpenAIGPTConfig):
    """
    MiniMax OpenAI configuration that extends OpenAIGPTConfig.
    MiniMax provides an OpenAI-compatible API at:
    - International: https://api.minimax.io/v1
    - China: https://api.minimaxi.com/v1

    Supported models:
    - MiniMax-M2.1
    - MiniMax-M2.1-lightning
    - MiniMax-M2
    """

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        """
        Get MiniMax API key from environment or parameters.
        """
        return api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(
        api_base: str | None = None,
    ) -> str:
        """
        Get MiniMax API base URL.
        Defaults to international endpoint: https://api.minimax.io/v1
        For China, set to: https://api.minimaxi.com/v1
        """
        return api_base or get_secret_str("MINIMAX_API_BASE") or "https://api.minimax.io/v1"

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        Get the complete URL for MiniMax OpenAI API.
        Override to ensure we use MiniMax's endpoint.
        """
        # Get the base URL (either provided or default MiniMax endpoint)
        base_url: Final = self.get_api_base(api_base=api_base)

        # Ensure it ends with /chat/completions
        if base_url.endswith("/chat/completions"):
            return base_url
        elif base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        elif base_url.endswith("/"):
            return f"{base_url}v1/chat/completions"
        else:
            return f"{base_url}/v1/chat/completions"

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: list[AllMessageValues],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> tuple[list[AllMessageValues], list[ChatCompletionToolParam] | None]:
        """
        Override to preserve cache_control for MiniMax.
        MiniMax supports cache_control - don't strip it.
        """
        # MiniMax supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for MiniMax.
        Adds reasoning_split and thinking to the list of supported params.
        """
        base_params: Final = super().get_supported_openai_params(model=model)
        additional_params: Final = ["reasoning_split"]

        # Add thinking parameter if model supports reasoning
        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider="minimax"):
                additional_params.append("thinking")
        except Exception:
            pass

        return base_params + additional_params

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,  # mutable-ok: matches parent  # pyright: ignore[reportMissingTypeArgument,reportUnknownParameterType]  # matches OpenAIGPTConfig.transform_response
        messages: list[AllMessageValues],  # mutable-ok: matches parent
        optional_params: dict,  # mutable-ok: matches parent  # pyright: ignore[reportMissingTypeArgument,reportUnknownParameterType]  # matches OpenAIGPTConfig.transform_response
        litellm_params: dict,  # mutable-ok: matches parent  # pyright: ignore[reportMissingTypeArgument,reportUnknownParameterType]  # matches OpenAIGPTConfig.transform_response
        encoding,  # pyright: ignore[reportAny,reportMissingParameterType]  # matches OpenAIGPTConfig.transform_response
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        """
        MiniMax M2.7 (reasoning_split unset/false, the default) can return
        its entire answer inside <think>...</think> with nothing trailing
        after the closing tag. The shared parser leaves `content` empty in
        that case, discarding the model's only real output.

        Scoped to MiniMax only: for other providers using <think> tags,
        content left empty after the tag is genuinely empty output, not a
        signal to promote reasoning_content into the visible channel —
        doing that generically risks leaking hidden reasoning for
        adversarial prompts that end right after </think>. MiniMax's docs
        confirm the whole-answer-in-<think> shape is expected behavior
        for this provider specifically.
        """
        response = super().transform_response(  # pyright: ignore[reportUnknownMemberType]  # super() inherits partially unknown param types from parent
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
        for choice in response.choices:
            message = choice.message
            reasoning_content = getattr(message, "reasoning_content", None)  # pyright: ignore[reportAny]  # Message deletes reasoning_content when None
            if reasoning_content and not (message.content or "").strip():
                message.content = reasoning_content
        return response
