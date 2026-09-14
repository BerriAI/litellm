from typing import Final

from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig


class BytePlusChatConfig(OpenAILikeChatConfig):
    """
    Reference: https://docs.byteplus.com/en/docs/ModelArk
    """

    @classmethod
    def get_config(cls) -> "BytePlusChatConfig":
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: matches BaseConfig interface
        return [  # mutable-ok: matches BaseConfig interface
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
        ]

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: matches BaseConfig interface
        optional_params: dict,  # mutable-ok: matches BaseConfig interface
        model: str,
        drop_params: bool,
        replace_max_completion_tokens_with_max_tokens: bool = True,
    ) -> dict:  # mutable-ok: matches BaseConfig interface
        mapped_params: Final = super().map_openai_params(
            non_default_params,
            optional_params,
            model,
            drop_params,
            replace_max_completion_tokens_with_max_tokens,
        )

        if "thinking" in mapped_params:
            thinking_val: Final = mapped_params.pop("thinking", None)
            if thinking_val is not None:
                if isinstance(thinking_val, bool):
                    mapped_params["extra_body"] = {  # mutable-ok: extra_body payload dict
                        "thinking": {  # mutable-ok: nested thinking dict
                            "type": "enabled" if thinking_val else "disabled"
                        },
                    }
                elif isinstance(thinking_val, dict):
                    mapped_params["extra_body"] = {"thinking": thinking_val}  # mutable-ok: extra_body payload dict

        return mapped_params
