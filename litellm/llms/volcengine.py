from typing import Optional, Union

from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig


class VolcEngineConfig(OpenAILikeChatConfig):
    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_completion_tokens",
            "max_tokens",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "function_call",
            "functions",
            "max_retries",
            "extra_headers",
            "thinking",
        ]  # works across all models

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        replace_max_completion_tokens_with_max_tokens: bool = True,
    ) -> dict:
        optional_params = super().map_openai_params(
            non_default_params,
            optional_params,
            model,
            drop_params,
            replace_max_completion_tokens_with_max_tokens,
        )

        if "thinking" in optional_params:
            thinking_value = optional_params.pop("thinking")
            
            # Handle disabled thinking case - don't add to extra_body if disabled
            if (
                thinking_value is not None 
                and isinstance(thinking_value, dict) 
                and thinking_value.get("type") == "disabled"
            ):
                # Skip adding thinking parameter when it's disabled
                pass
            else:
                # Add thinking parameter to extra_body for all other cases
                optional_params.setdefault("extra_body", {})["thinking"] = thinking_value

        return optional_params
