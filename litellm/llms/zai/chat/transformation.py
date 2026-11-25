from collections.abc import Mapping, Sequence
from typing import Final

from typing_extensions import TypeIs  # noqa: TID251  # TypeIs reaches typing only on 3.13

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.factory import response_schema_prompt
from litellm.litellm_core_utils.reasoning_effort_capability import (
    configured_reasoning_effort_rounding,
    declared_reasoning_efforts_for_map_key,
)
from litellm.litellm_core_utils.reasoning_effort_strength import (
    ReasoningEffortRounding,
    nearest_declared_reasoning_effort,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam, ChatCompletionUserMessage
from litellm.utils import ModelInfoLookupMiss, lookup_model_info_for_registration

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE: Final = "https://api.z.ai/api/paas/v4"
TOKEN_LIMIT_PARAMS: Final = frozenset({"max_tokens", "max_completion_tokens"})
REWRITTEN_PARAMS: Final = TOKEN_LIMIT_PARAMS | {"reasoning_effort"}
REASONING_OFF: Final = "none"
PROVIDER_DEFAULT: Final = "default"
THINKING_PARAMS: Final = frozenset({"thinking"})


def _is_json_object(
    value: object,
) -> TypeIs[Mapping[str, object]]:  # guard-ok: cost-map rows and request params are JSON
    return isinstance(value, Mapping)


def _json_field(value: object, key: str) -> object:
    return value.get(key) if _is_json_object(value) else None


def _exceeds_output_limit(model: str, max_tokens: object) -> bool:
    info: Final = lookup_model_info_for_registration(model=model, custom_llm_provider="zai")
    limit: Final = None if isinstance(info, ModelInfoLookupMiss) else info.get("max_output_tokens")
    return isinstance(max_tokens, int) and isinstance(limit, int) and max_tokens > limit


def _catalog_reasoning_efforts(model: str) -> tuple[str, ...] | None:
    provider_levels: Final = declared_reasoning_efforts_for_map_key(f"zai/{model}")
    return provider_levels if provider_levels is not None else declared_reasoning_efforts_for_map_key(model)


def _requested_level(requested: object) -> object:
    return _json_field(requested, "effort") if _is_json_object(requested) else requested


def _declared_effort(
    level: str, declared: Sequence[str], drop_params: bool, rounding: ReasoningEffortRounding
) -> str | None:
    rounded: Final = nearest_declared_reasoning_effort(level, declared, rounding)
    return rounded if rounded in declared or not drop_params else None


def _accepted_effort(
    model: str,
    requested: object,
    drop_params: bool,
    rounding: ReasoningEffortRounding,
    declared: Sequence[str] | None,
) -> object:
    level: Final = _requested_level(requested)
    if not isinstance(level, str):
        return None if drop_params else level
    if level == PROVIDER_DEFAULT:
        return None
    if declared is None:
        return level
    accepted: Final = _declared_effort(level, declared, drop_params, rounding)
    if accepted is not None and accepted != level:
        verbose_logger.debug(
            "zai: %s takes reasoning_effort %s, sending %s in place of %s", model, declared, accepted, level
        )
    return accepted


def _schema_without_native_support(model: str, response_format: object) -> Mapping[str, object] | None:
    if _json_field(response_format, "type") != "json_schema":
        return None
    if litellm.supports_response_schema(model=model, custom_llm_provider="zai"):
        return None
    schema: Final = _json_field(_json_field(response_format, "json_schema"), "schema")
    return schema if _is_json_object(schema) else None


class ZAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "zai"

    @property
    def prompts_unsupported_json_schema(self) -> bool:
        return True

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("ZAI_API_BASE") or ZAI_API_BASE
        dynamic_api_key: Final = api_key or get_secret_str("ZAI_API_KEY")
        return api_base, dynamic_api_key

    def get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        return self._get_openai_compatible_provider_info(api_base, api_key)

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: list[AllMessageValues],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> tuple[list[AllMessageValues], list[ChatCompletionToolParam] | None]:
        """
        Override to preserve cache_control for GLM/ZAI.
        GLM supports cache_control - don't strip it.
        """
        # GLM/ZAI supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list[str]:
        base_params: Final = [
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
        ]

        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider=self.custom_llm_provider):
                base_params.extend(("thinking", "reasoning_effort"))
        except Exception:
            pass

        return base_params

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
        reasoning_effort_rounding: ReasoningEffortRounding | None = None,
        declared_reasoning_efforts: Sequence[str] | None = None,
    ) -> dict[str, object]:  # mutable-ok: request pipeline adds parameters to the returned body
        supported_params: Final = self.get_supported_openai_params(model)
        sent: Final = {param: value for param, value in non_default_params.items() if param in supported_params}
        drops_unsupported: Final = litellm.drop_params or drop_params
        max_completion_tokens: Final = sent.get("max_completion_tokens")
        falls_back_to_max_tokens: Final = max_completion_tokens is None or (
            drops_unsupported and _exceeds_output_limit(model, max_completion_tokens)
        )
        max_tokens: Final = sent.get("max_tokens") if falls_back_to_max_tokens else max_completion_tokens
        requested_effort: Final = sent.get("reasoning_effort")
        rounding: Final = reasoning_effort_rounding or configured_reasoning_effort_rounding()
        declared: Final = (
            declared_reasoning_efforts if declared_reasoning_efforts is not None else _catalog_reasoning_efforts(model)
        )
        always_thinks: Final = declared is not None and REASONING_OFF not in declared
        drops_disabled_thinking: Final = (
            drops_unsupported and always_thinks and _json_field(sent.get("thinking"), "type") == "disabled"
        )
        unsent_params: Final = REWRITTEN_PARAMS | THINKING_PARAMS if drops_disabled_thinking else REWRITTEN_PARAMS
        turns_thinking_off: Final = (
            _requested_level(requested_effort) == REASONING_OFF and "thinking" not in sent and not always_thinks
        )
        reasoning_effort: Final = (
            None
            if requested_effort is None or turns_thinking_off
            else _accepted_effort(model, requested_effort, drops_unsupported, rounding, declared)
        )
        return {
            **optional_params,
            **{param: value for param, value in sent.items() if param not in unsent_params},
            **({} if max_tokens is None else {"max_tokens": max_tokens}),
            **({"thinking": {"type": "disabled"}} if turns_thinking_off else {}),
            **({} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}),
        }

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: OpenAIGPTConfig.transform_request signature
        optional_params: dict[str, object],  # mutable-ok: OpenAIGPTConfig.transform_request signature
        litellm_params: dict[str, object],  # mutable-ok: OpenAIGPTConfig.transform_request signature
        headers: dict[str, str],  # mutable-ok: OpenAIGPTConfig.transform_request signature
    ) -> dict[str, object]:  # mutable-ok: inherited contract returns the request body
        schema: Final = (
            _schema_without_native_support(model, optional_params.get("response_format"))
            if self.prompts_unsupported_json_schema
            else None
        )
        schema_message: Final = (
            ()
            if schema is None
            else (
                ChatCompletionUserMessage(
                    role="user", content=response_schema_prompt(model=model, response_schema=schema, llm_provider="zai")
                ),
            )
        )
        return super().transform_request(
            model,
            [*messages, *schema_message],
            optional_params if schema is None else {**optional_params, "response_format": {"type": "json_object"}},
            litellm_params,
            headers,
        )
