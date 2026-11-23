"""
Dynamic configuration class generator for JSON-based providers.
"""

import json
from collections.abc import Coroutine, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, overload

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from .json_loader import SimpleProviderConfig

if TYPE_CHECKING:
    from litellm.llms.openai_like.responses.transformation import OpenAILikeResponsesConfig
    from litellm.types.llms.openai import ChatCompletionToolParam

_PARAM_NAMES_ADAPTER: Final[TypeAdapter[tuple[str, ...]]] = TypeAdapter(tuple[str, ...])


def _values_outside_fixed(
    params: Mapping[str, object], fixed_values: Mapping[str, bool | int | float]
) -> Mapping[str, object]:
    return MappingProxyType(
        {param: value for param, value in params.items() if param in fixed_values and value != fixed_values[param]}
    )


def _describe(params: Mapping[str, object]) -> str:
    return ", ".join(f"{param}={json.dumps(value)}" for param, value in params.items())


def create_config_class(provider: SimpleProviderConfig):
    """Generate config class dynamically from JSON configuration"""

    # Choose base class
    base_class: Final[type] = OpenAIGPTConfig if provider.base_class == "openai_gpt" else OpenAILikeChatConfig

    class JSONProviderConfig(base_class):
        def _flattens_content_lists(self) -> bool:
            return bool(provider.special_handling.get("convert_content_list_to_string"))

        @overload
        def _transform_messages(
            self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
        ) -> Coroutine[object, object, list[AllMessageValues]]: ...

        @overload
        def _transform_messages(
            self,
            messages: list[AllMessageValues],
            model: str,
            is_async: Literal[False] = False,
        ) -> list[AllMessageValues]: ...

        def _transform_messages(
            self, messages: list[AllMessageValues], model: str, is_async: bool = False
        ) -> list[AllMessageValues] | Coroutine[object, object, list[AllMessageValues]]:
            """Transform messages based on special_handling config"""

            # Handle content list to string conversion if configured
            if self._flattens_content_lists():
                messages = handle_messages_with_content_list_to_str_conversion(messages)

            if is_async:
                return super()._transform_messages(messages=messages, model=model, is_async=True)
            else:
                return super()._transform_messages(messages=messages, model=model, is_async=False)

        def _messages_and_tools_in_cache_mark_dialect(
            self,
            model: str,
            messages: Sequence[AllMessageValues],
            tools: Sequence["ChatCompletionToolParam"] | None,
            prompt_cache_options: object,
            litellm_params: Mapping[str, object],
        ) -> tuple[Sequence[AllMessageValues], Sequence["ChatCompletionToolParam"] | None]:
            if not provider.special_handling_for(model).get("reads_prompt_cache_breakpoint"):
                return super()._messages_and_tools_in_cache_mark_dialect(
                    model, messages, tools, prompt_cache_options, litellm_params
                )
            from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook

            return self.remove_cache_control_flag_from_messages_and_tools(
                model=model,
                messages=[
                    AnthropicCacheControlHook.cache_control_as_prompt_cache_breakpoints(message) for message in messages
                ],
                tools=None if tools is None else list(tools),
            )

        def _get_openai_compatible_provider_info(
            self, api_base: str | None, api_key: str | None
        ) -> tuple[str | None, str | None]:
            """Get API base and key from JSON config"""

            # Resolve base URL
            resolved_base = api_base
            if not resolved_base and provider.api_base_env:
                resolved_base = get_secret_str(provider.api_base_env)
            if not resolved_base:
                resolved_base = provider.base_url

            # Resolve API key
            resolved_key: Final = api_key or provider.api_key_from(get_secret_str)

            return resolved_base, resolved_key

        def get_openai_compatible_provider_info(
            self,
            api_base: str | None,
            api_key: str | None,
        ) -> tuple[str | None, str | None]:
            return self._get_openai_compatible_provider_info(api_base, api_key)

        def get_complete_url(
            self,
            api_base: str | None,
            api_key: str | None,
            model: str,
            optional_params: dict,
            litellm_params: dict,
            stream: bool | None = None,
        ) -> str:
            """Build complete URL for the API endpoint"""
            if not api_base:
                api_base = provider.base_url

            if api_base is None:
                raise ValueError(f"api_base is required for provider {provider.slug}")

            if not api_base.endswith("/chat/completions"):
                api_base = f"{api_base}/chat/completions"

            return api_base

        def get_supported_openai_params(self, model: str) -> list:
            """Get supported OpenAI params, excluding tool-related params for models
            that don't support function calling."""
            from litellm.utils import supports_function_calling, supports_reasoning

            inherited_params: Final = _PARAM_NAMES_ADAPTER.validate_python(
                super().get_supported_openai_params(model=model)
            )
            supported_params: Final = [
                *(param for param in inherited_params if param not in provider.unsupported_params),
                *(param for param in provider.extra_supported_params if param not in inherited_params),
            ]

            _supports_fc: Final = supports_function_calling(model=model, custom_llm_provider=provider.slug)

            if not _supports_fc:
                tool_params: Final = [
                    "tools",
                    "tool_choice",
                    "function_call",
                    "functions",
                    "parallel_tool_calls",
                ]
                for param in tool_params:
                    if param in supported_params:
                        supported_params.remove(param)
                verbose_logger.debug(
                    "Model %s on provider %s does not support function calling — removed tool-related params from supported params.",
                    model,
                    provider.slug,
                )

            _supports_reasoning: Final = supports_reasoning(model=model, custom_llm_provider=provider.slug)
            if _supports_reasoning and "reasoning_effort" not in supported_params:
                supported_params.append("reasoning_effort")

            return supported_params

        def map_openai_params(
            self,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: bool,
        ) -> dict:
            """Apply parameter mappings and constraints"""

            supported_params: Final = self.get_supported_openai_params(model)
            rejected_values: Final = _values_outside_fixed(non_default_params, provider.fixed_value_params)
            if rejected_values and not (drop_params or litellm.drop_params is True):
                raise litellm.UnsupportedParamsError(
                    message=f"{provider.slug} does not support {_describe(rejected_values)}. It only accepts "
                    f"{_describe({param: provider.fixed_value_params[param] for param in rejected_values})}. "
                    "To drop unsupported openai params from the call, set `litellm.drop_params = True`",
                    llm_provider=provider.slug,
                    model=model,
                )

            # Apply supported params
            for param, value in non_default_params.items():
                if param in rejected_values:
                    continue
                # Check parameter mappings first
                if param in provider.param_mappings:
                    optional_params[provider.param_mappings[param]] = value
                elif param in supported_params:
                    optional_params[param] = value

            # Apply temperature constraints if present
            if "temperature" in optional_params:
                temp = optional_params["temperature"]
                constraints: Final = provider.constraints

                # Clamp to max
                if "temperature_max" in constraints:
                    temp = min(temp, constraints["temperature_max"])

                # Clamp to min
                if "temperature_min" in constraints:
                    temp = max(temp, constraints["temperature_min"])

                # Special case: temperature_min_with_n_gt_1
                if "temperature_min_with_n_gt_1" in constraints:
                    n: Final = optional_params.get("n", 1)
                    if n > 1 and temp < constraints["temperature_min_with_n_gt_1"]:
                        temp = constraints["temperature_min_with_n_gt_1"]

                optional_params["temperature"] = temp

            return optional_params

        @property
        def custom_llm_provider(self) -> str | None:
            return provider.slug

    return JSONProviderConfig


_responses_config_cache: Final[dict] = {}


def create_responses_config_class(provider: SimpleProviderConfig) -> "type[OpenAILikeResponsesConfig]":
    """Generate a Responses API config class dynamically from JSON configuration.

    Parallel to create_config_class() but for /v1/responses endpoints.
    Classes are cached per provider slug to avoid regeneration on every request.
    """
    if provider.slug in _responses_config_cache:
        return _responses_config_cache[provider.slug]

    from litellm.llms.openai_like.responses.transformation import (
        OpenAILikeResponsesConfig,
    )
    from litellm.types.llms.openai import ResponseInputParam
    from litellm.types.router import GenericLiteLLMParams

    class JSONProviderResponsesConfig(OpenAILikeResponsesConfig):
        @property
        def custom_llm_provider(self):
            return provider.slug

        def validate_environment(
            self,
            headers: dict,
            model: str,
            litellm_params: GenericLiteLLMParams | None,
        ) -> dict:
            litellm_params = litellm_params or GenericLiteLLMParams()
            api_key: Final = litellm_params.api_key or provider.api_key_from(get_secret_str)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            return headers

        def get_complete_url(
            self,
            api_base: str | None,
            litellm_params: dict,
        ) -> str:
            if not api_base:
                if provider.api_base_env:
                    api_base = get_secret_str(provider.api_base_env)
                if not api_base:
                    api_base = provider.base_url

            if api_base is None:
                raise ValueError(f"api_base is required for provider {provider.slug}")

            api_base = api_base.rstrip("/")
            return f"{api_base}/responses"

        def _reads_prompt_cache_breakpoints(
            self, model: str, litellm_params: GenericLiteLLMParams, prompt_cache_options: object
        ) -> bool:
            return bool(
                provider.special_handling_for(model).get("reads_prompt_cache_breakpoint")
            ) or super()._reads_prompt_cache_breakpoints(model, litellm_params, prompt_cache_options)

        def transform_responses_api_request(
            self,
            model: str,
            input: str | ResponseInputParam,
            response_api_optional_request_params: dict,
            litellm_params: GenericLiteLLMParams,
            headers: dict,
        ) -> dict:
            if provider.special_handling.get("force_store_false"):
                response_api_optional_request_params["store"] = False
            return super().transform_responses_api_request(
                model=model,
                input=input,
                response_api_optional_request_params=response_api_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )

    _responses_config_cache[provider.slug] = JSONProviderResponsesConfig
    return JSONProviderResponsesConfig
