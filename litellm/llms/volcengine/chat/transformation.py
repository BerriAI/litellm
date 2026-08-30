from typing import Final

from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig


class VolcEngineChatConfig(OpenAILikeChatConfig):
    """
    Reference: https://www.volcengine.com/docs/82379/1494384
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
        locals_: Final = locals().copy()
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
            """
            The `thinking` parameters of VolcEngine model has different default values.
            See the docs for details.
            Refrence: https://www.volcengine.com/docs/82379/1449737#0002
            """
            thinking_value: Final = optional_params.pop("thinking")

            # Handle using thinking params case - add to extra_body if value is legal
            if (
                thinking_value is not None
                and isinstance(thinking_value, dict)
                and thinking_value.get("type", None) in ["enabled", "disabled", "auto"]  # legal values, see docs
            ):
                # Add thinking parameter to extra_body for all legal cases
                optional_params.setdefault("extra_body", {})["thinking"] = thinking_value
            else:
                # Skip adding thinking parameter when it's not set or has invalid value
                pass
        return optional_params
