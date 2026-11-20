from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig

_PARAMS: Final = TypeAdapter(Mapping[str, object])


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
        functions: Sequence[object] | None = None,
        logit_bias: dict | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: int | None = None,
        stop: str | Sequence[object] | None = None,
        temperature: int | None = None,
        top_p: int | None = None,
        response_format: dict | None = None,
    ) -> None:
        locals_: Final[Mapping[str, object]] = dict(locals())
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list[str]:
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
            "reasoning_effort",
        ]  # works across all models

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
        replace_max_completion_tokens_with_max_tokens: bool = True,
    ) -> dict[str, object]:  # mutable-ok: provider config overrides return mutable SDK parameter dictionaries
        mapped: Final = _PARAMS.validate_python(
            super().map_openai_params(
                dict(non_default_params),
                dict(optional_params),
                model,
                drop_params,
                replace_max_completion_tokens_with_max_tokens,
            )
        )

        thinking: Final = as_str_mapping(mapped.get("thinking"))
        thinking_type: Final = thinking.get("type") if thinking is not None else None
        effort: Final = mapped.get("reasoning_effort")
        resolved_thinking: Final = (
            thinking
            if thinking_type in ("enabled", "disabled", "auto")
            else {"type": "disabled"}
            if effort == "none"
            else None
        )
        excluded: Final = (
            frozenset(("thinking", "reasoning_effort"))
            if thinking_type == "disabled" or effort == "none"
            else frozenset(("thinking",))
        )
        forwarded: Final = {key: value for key, value in mapped.items() if key not in excluded}
        if resolved_thinking is None:
            return forwarded
        return {
            **forwarded,
            "extra_body": {
                **(as_str_mapping(mapped.get("extra_body")) or {}),
                "thinking": resolved_thinking,
            },
        }
