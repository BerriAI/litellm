"""
Dynamic configuration class generator for JSON-based providers.
"""

from collections.abc import Coroutine, Mapping
from types import MappingProxyType
from typing import Any, Final, Literal, overload

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ServiceTier

from .json_loader import SimpleProviderConfig

_SERVICE_TIER_TO_COMPLETION_WINDOW: Final[Mapping[str, str]] = MappingProxyType(
    {
        ServiceTier.FLEX.value: "flex",
        ServiceTier.BALANCED.value: "balanced",
        ServiceTier.PRIORITY.value: "asap",
        "default": "asap",
    }
)


def _apply_service_tier_as_completion_window(body: Mapping[str, object]) -> dict[str, object]:
    service_tier: Final = body.get("service_tier")
    window: Final[str | None] = (
        _SERVICE_TIER_TO_COMPLETION_WINDOW.get(service_tier.lower()) if isinstance(service_tier, str) else None
    )
    raw_metadata: Final = body.get("metadata")
    metadata: Final[Mapping[str, object]] = raw_metadata if isinstance(raw_metadata, dict) else MappingProxyType({})
    new_body: Final[dict[str, object]] = {  # mutable-ok: transform_request returns a plain dict
        key: value for key, value in body.items() if key != "service_tier"
    }
    if window is None or "completion_window" in metadata:
        return new_body
    return {  # mutable-ok: transform_request returns a plain dict
        **new_body,
        "metadata": {**metadata, "completion_window": window},  # mutable-ok: transform_request returns a plain dict
    }


def _service_tier_as_completion_window_enabled(provider: SimpleProviderConfig) -> bool:
    return provider.special_handling.get("service_tier_as_completion_window") is True


def create_config_class(provider: SimpleProviderConfig):
    """Generate config class dynamically from JSON configuration"""

    # Choose base class
    base_class: Final[type] = OpenAIGPTConfig if provider.base_class == "openai_gpt" else OpenAILikeChatConfig

    class JSONProviderConfig(base_class):
        @overload
        def _transform_messages(
            self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
        ) -> Coroutine[Any, Any, list[AllMessageValues]]: ...

        @overload
        def _transform_messages(
            self,
            messages: list[AllMessageValues],
            model: str,
            is_async: Literal[False] = False,
        ) -> list[AllMessageValues]: ...

        def _transform_messages(
            self, messages: list[AllMessageValues], model: str, is_async: bool = False
        ) -> list[AllMessageValues] | Coroutine[Any, Any, list[AllMessageValues]]:
            """Transform messages based on special_handling config"""

            # Handle content list to string conversion if configured
            if provider.special_handling.get("convert_content_list_to_string"):
                messages = handle_messages_with_content_list_to_str_conversion(messages)

            if is_async:
                return super()._transform_messages(messages=messages, model=model, is_async=True)
            else:
                return super()._transform_messages(messages=messages, model=model, is_async=False)

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
            if not api_base:
                api_base = provider.base_url

            if api_base is None:
                raise ValueError(f"api_base is required for provider {provider.slug}")

            if not api_base.endswith("/chat/completions"):
                api_base = f"{api_base}/chat/completions"

            return api_base

        def transform_request(
            self,
            model: str,
            messages: list[AllMessageValues],
            optional_params: dict[str, object],
            litellm_params: dict[str, object],
            headers: dict[str, object],
        ) -> dict[str, object]:
            body: Final = super().transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
            if _service_tier_as_completion_window_enabled(provider):
                return _apply_service_tier_as_completion_window(body)
            return body

        def get_supported_openai_params(self, model: str) -> list:
            """Get supported OpenAI params, excluding tool-related params for models
            that don't support function calling."""
            from litellm.utils import supports_function_calling, supports_reasoning

            base_params: Final = super().get_supported_openai_params(model=model)

            _supports_fc: Final = supports_function_calling(model=model, custom_llm_provider=provider.slug)

            tool_params: Final = (
                () if _supports_fc else ("tools", "tool_choice", "function_call", "functions", "parallel_tool_calls")
            )
            if not _supports_fc:
                verbose_logger.debug(
                    "Model %s on provider %s does not support function calling — removed tool-related params from supported params.",
                    model,
                    provider.slug,
                )

            _supports_reasoning: Final = supports_reasoning(model=model, custom_llm_provider=provider.slug)
            extra_params: Final = (
                ("reasoning_effort",) if _supports_reasoning and "reasoning_effort" not in base_params else ()
            )

            excluded_params: Final = frozenset(tool_params) | frozenset(provider.unsupported_params)
            supported_params: Final = tuple(
                param for param in (*base_params, *extra_params) if param not in excluded_params
            )

            return list(supported_params)  # mutable-ok: get_supported_openai_params contract returns a list

        def map_openai_params(
            self,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: bool,
        ) -> dict:
            """Apply parameter mappings and constraints"""

            supported_params: Final = self.get_supported_openai_params(model)

            # Apply supported params
            for param, value in non_default_params.items():
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


def create_responses_config_class(provider: SimpleProviderConfig):
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
            if not api_base:
                if provider.api_base_env:
                    api_base = get_secret_str(provider.api_base_env)
                if not api_base:
                    api_base = provider.base_url

            if api_base is None:
                raise ValueError(f"api_base is required for provider {provider.slug}")

            api_base = api_base.rstrip("/")
            return f"{api_base}/responses"

        def transform_responses_api_request(
            self,
            model: str,
            input: str | ResponseInputParam,
            response_api_optional_request_params: dict[str, object],
            litellm_params: GenericLiteLLMParams,
            headers: dict[str, object],
        ) -> dict[str, object]:
            if provider.special_handling.get("force_store_false"):
                response_api_optional_request_params["store"] = False
            body: Final = super().transform_responses_api_request(
                model=model,
                input=input,
                response_api_optional_request_params=response_api_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )
            if _service_tier_as_completion_window_enabled(provider):
                return _apply_service_tier_as_completion_window(body)
            return body

    _responses_config_cache[provider.slug] = JSONProviderResponsesConfig
    return JSONProviderResponsesConfig
