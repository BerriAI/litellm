"""
Dynamic configuration class generator for JSON-based providers.
"""

from collections.abc import Coroutine, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, overload

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
    handle_messages_with_text_only_content_lists_as_str,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.router_utils.reasoning_effort_capability import (
    declared_reasoning_efforts_for_model,
    nearest_declared_reasoning_effort,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from .json_loader import SimpleProviderConfig

if TYPE_CHECKING:
    from litellm.llms.openai_like.responses.transformation import OpenAILikeResponsesConfig


def _flattened_messages(
    messages: Sequence[AllMessageValues], handling: Mapping[str, bool]
) -> Sequence[AllMessageValues]:
    if handling.get("convert_content_list_to_string"):
        return handle_messages_with_content_list_to_str_conversion(list(messages))
    if handling.get("convert_text_content_list_to_string"):
        return handle_messages_with_text_only_content_lists_as_str(messages)
    return messages


def _with_declared_reasoning_effort(
    params: Mapping[str, object], *, model: str, provider_slug: str, drop_params: bool
) -> Mapping[str, object]:
    requested: Final = params.get("reasoning_effort")
    declared: Final = declared_reasoning_efforts_for_model(model, provider_slug)
    if declared is None or not isinstance(requested, str):
        return params
    accepted: Final = nearest_declared_reasoning_effort(requested, declared)
    if drop_params and accepted not in declared:
        return MappingProxyType({key: value for key, value in params.items() if key != "reasoning_effort"})
    return MappingProxyType({**params, "reasoning_effort": accepted})


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
            flattened: Final = _flattened_messages(messages, provider.special_handling_for(model))
            if is_async:
                return super()._transform_messages(messages=list(flattened), model=model, is_async=True)
            else:
                return super()._transform_messages(messages=list(flattened), model=model, is_async=False)

        def _get_openai_compatible_provider_info(
            self, api_base: str | None, api_key: str | None
        ) -> tuple[str | None, str | None]:
            return self.get_model_provider_info(None, api_base, api_key)

        def get_openai_compatible_provider_info(
            self,
            api_base: str | None,
            api_key: str | None,
        ) -> tuple[str | None, str | None]:
            return self._get_openai_compatible_provider_info(api_base, api_key)

        def get_model_provider_info(
            self, model: str | None, api_base: str | None, api_key: str | None
        ) -> tuple[str | None, str | None]:
            resolved_base: Final = (
                api_base
                or (get_secret_str(provider.api_base_env) if provider.api_base_env else None)
                or provider.default_base_url(model)
            )
            resolved_key: Final = api_key or get_secret_str(provider.api_key_env)
            return resolved_base, resolved_key

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
            resolved_base: Final = api_base or provider.default_base_url(model)
            if resolved_base.endswith("/chat/completions"):
                return resolved_base
            return f"{resolved_base}/chat/completions"

        def get_supported_openai_params(self, model: str) -> list:
            """Get supported OpenAI params, excluding tool-related params for models
            that don't support function calling."""
            from litellm.utils import supports_function_calling, supports_reasoning

            supported_params: Final = super().get_supported_openai_params(model=model)

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
            requested_params: Final = _with_declared_reasoning_effort(
                non_default_params,
                model=model,
                provider_slug=provider.slug,
                drop_params=drop_params,
            )

            # Apply supported params
            for param, value in requested_params.items():
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
            api_key: Final = litellm_params.api_key or get_secret_str(provider.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            return headers

        def get_complete_url(
            self,
            api_base: str | None,
            litellm_params: dict,
        ) -> str:
            resolved_base: Final = (
                api_base
                or (get_secret_str(provider.api_base_env) if provider.api_base_env else None)
                or provider.base_url
            )
            return f"{resolved_base.rstrip('/')}/responses"

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
