"""
TokenHub Chat Completion Configuration

TokenHub (Tencent Cloud) provides an OpenAI-compatible API.
Reference: https://cloud.tencent.com/document/product/1823/130058

Supports:
- Streaming
- Function Calling / Tool Use
- Thinking / Chain-of-Thought (via thinking parameter)
- Prompt Cache
- Structured Output (response_format)
"""

from __future__ import annotations

from litellm import verbose_logger
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.utils import supports_reasoning


class TokenHubChatConfig(OpenAILikeChatConfig):
    """
    Configuration for TokenHub (Tencent Cloud MaaS) chat completions.

    Reference: https://cloud.tencent.com/document/product/1823/130058

    TokenHub is OpenAI-compatible and supports:
    - All standard OpenAI chat completion parameters
    - thinking parameter for chain-of-thought control (enabled/disabled/adaptive)
    - reasoning_effort for GPT-series models
    - Prompt caching via X-Session-ID header or prompt_cache_key body field
    """

    frequency_penalty: int | None = None
    function_call: str | dict | None = None
    functions: list | None = None
    logit_bias: dict | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: int | None = None
    stop: str | list | None = None
    temperature: int | None = None
    top_p: int | None = None
    response_format: dict | None = None

    def __init__(
        self,
        frequency_penalty: int | None = None,
        function_call: str | dict | None = None,
        functions: list | None = None,
        logit_bias: dict | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: int | None = None,
        stop: str | list | None = None,
        temperature: int | None = None,
        top_p: int | None = None,
        response_format: dict | None = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list[str]:
        """
        Returns the list of supported OpenAI params for TokenHub models.
        """
        params = [
            "frequency_penalty",
            "function_call",
            "functions",
            "logit_bias",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "tools",
            "tool_choice",
            "top_p",
            "parallel_tool_calls",
        ]

        # Only add thinking/reasoning params for models that support it
        if supports_reasoning(model=model, custom_llm_provider="tokenhub"):
            params.append("thinking")
            params.append("reasoning_effort")

        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        replace_max_completion_tokens_with_max_tokens: bool = False,
    ) -> dict:
        """
        Map OpenAI params to TokenHub params.

        Handles the thinking parameter specially:
        - For most models: thinking.type = enabled/disabled/adaptive
        - For GPT-series: reasoning_effort = none/low/medium/high/xhigh
        """
        # Handle thinking parameter
        thinking_param = non_default_params.pop("thinking", None)
        if thinking_param is not None:
            # Validate thinking param
            if isinstance(thinking_param, dict):
                thinking_type = thinking_param.get("type")
                if thinking_type in ("enabled", "disabled", "adaptive"):
                    # Pass thinking as extra_body for TokenHub
                    if "extra_body" not in optional_params:
                        optional_params["extra_body"] = {}
                    optional_params["extra_body"]["thinking"] = thinking_param

        # Handle reasoning_effort (for GPT-series on TokenHub)
        reasoning_effort = non_default_params.pop("reasoning_effort", None)
        if reasoning_effort is not None:
            if reasoning_effort in ("none", "low", "medium", "high", "xhigh"):
                if "extra_body" not in optional_params:
                    optional_params["extra_body"] = {}
                optional_params["extra_body"]["reasoning_effort"] = reasoning_effort
            else:
                verbose_logger.warning(
                    "TokenHub: invalid reasoning_effort value '%s' — "
                    "must be one of: none, low, medium, high, xhigh. "
                    "Parameter was not applied.",
                    reasoning_effort,
                )

        # Let parent handle standard OpenAI params
        return super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
            replace_max_completion_tokens_with_max_tokens=replace_max_completion_tokens_with_max_tokens,
        )
