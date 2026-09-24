"""
Dynamic configuration class generator for JSON-based providers.
"""

from collections.abc import Coroutine, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, overload

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ServiceTier

if TYPE_CHECKING:
    from litellm.llms.openai_like.responses.transformation import OpenAILikeResponsesConfig
    from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams
    from litellm.types.router import GenericLiteLLMParams

from .json_loader import SimpleProviderConfig

_SERVICE_TIER_TO_COMPLETION_WINDOW: Final[Mapping[str, str]] = MappingProxyType(
    {
        ServiceTier.FLEX.value: "flex",
        ServiceTier.BALANCED.value: "balanced",
        ServiceTier.PRIORITY.value: "asap",
        "default": "asap",
    }
)


def _completion_window_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _apply_service_tier_as_completion_window(
    body: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: transform_request returns a plain dict
    service_tier: Final = body.get("service_tier")
    mapped_window: Final[str | None] = (
        _SERVICE_TIER_TO_COMPLETION_WINDOW.get(service_tier.lower()) if isinstance(service_tier, str) else None
    )
    raw_metadata: Final = body.get("metadata")
    metadata: Final[Mapping[str, object]] = raw_metadata if isinstance(raw_metadata, dict) else MappingProxyType({})
    raw_extra_body: Final = body.get("extra_body")
    extra_body: Final[Mapping[str, object]] = (
        raw_extra_body if isinstance(raw_extra_body, dict) else MappingProxyType({})
    )
    raw_extra_metadata: Final = extra_body.get("metadata")
    extra_metadata: Final[Mapping[str, object]] = (
        raw_extra_metadata if isinstance(raw_extra_metadata, dict) else MappingProxyType({})
    )
    caller_window: Final[str | None] = _completion_window_str(
        extra_metadata.get("completion_window")
    ) or _completion_window_str(metadata.get("completion_window"))
    window: Final[str | None] = caller_window or mapped_window
    new_body: Final = MappingProxyType({key: value for key, value in body.items() if key != "service_tier"})
    if window is None or (caller_window is None and window == "asap" and body.get("background") is True):
        return dict(new_body)  # mutable-ok: transform_request returns a plain dict
    windowed_metadata: Final = {**metadata, "completion_window": window}  # mutable-ok: nested wire dict
    merged: Final = {**new_body, "metadata": windowed_metadata}  # mutable-ok: transform_request returns a plain dict
    if isinstance(raw_extra_metadata, dict) and "completion_window" not in extra_metadata:
        windowed_extra_metadata: Final = {  # mutable-ok: nested wire dict
            **extra_metadata,
            "completion_window": window,
        }
        extra_body_with_window: Final = {  # mutable-ok: nested wire dict
            **extra_body,
            "metadata": windowed_extra_metadata,
        }
        return {**merged, "extra_body": extra_body_with_window}  # mutable-ok: SDK merges extra_body into the wire body
    return merged


def _merge_extra_body_keeping_metadata(
    request: Mapping[str, object],
    extra_body: Mapping[str, object] | None,
) -> dict[str, object]:  # mutable-ok: wire request body is a plain dict
    """Shallow merge where ``metadata`` merges one level deep so a caller window
    inside ``extra_body.metadata`` wins over a mapped one but cannot wipe it out
    by replacing the whole metadata dict."""
    if not extra_body:
        return dict(request)  # mutable-ok: wire request body is a plain dict
    request_metadata: Final = request.get("metadata")
    extra_metadata: Final = extra_body.get("metadata")
    if not isinstance(request_metadata, dict) or not isinstance(extra_metadata, dict):
        return {**request, **extra_body}  # mutable-ok: request body sent over the wire
    merged_metadata: Final = {**request_metadata, **extra_metadata}  # mutable-ok: request body sent over the wire
    return {**request, **extra_body, "metadata": merged_metadata}  # mutable-ok: request body sent over the wire


def _service_tier_as_completion_window_enabled(provider: SimpleProviderConfig) -> bool:
    return provider.special_handling.get("service_tier_as_completion_window") is True


_SUPPORTED_SERVICE_TIERS: Final = frozenset((*_SERVICE_TIER_TO_COMPLETION_WINDOW, "auto"))


def _service_tier_completion_window_drop(
    provider: SimpleProviderConfig, service_tier: object, model: str, drop_params: bool | None
) -> bool:
    if not _service_tier_as_completion_window_enabled(provider):
        return False
    if service_tier is None or (isinstance(service_tier, str) and service_tier.lower() in _SUPPORTED_SERVICE_TIERS):
        return False
    if drop_params or litellm.drop_params:
        return True
    raise litellm.UnsupportedParamsError(
        status_code=400,
        message=(
            f"{provider.slug} does not support service_tier '{service_tier}'. "
            "Supported values: auto, default, flex, balanced, priority. "
            "To drop unsupported params set litellm.drop_params=True"
        ),
        model=model,
        llm_provider=provider.slug,
    )


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
            messages: list[AllMessageValues],  # mutable-ok: matches base signature
            optional_params: Mapping[str, object],
            litellm_params: Mapping[str, object],
            headers: Mapping[str, object],
        ) -> dict[str, object]:  # mutable-ok: matches base signature
            body: Final = super().transform_request(
                model=model,
                messages=messages,
                optional_params=dict(optional_params),  # mutable-ok: base signature requires dict
                litellm_params=dict(litellm_params),  # mutable-ok: base signature requires dict
                headers=dict(headers),  # mutable-ok: base signature requires dict
            )
            if _service_tier_as_completion_window_enabled(provider):
                return _apply_service_tier_as_completion_window(body)
            return body

        def merge_extra_body(
            self,
            request: Mapping[str, object],
            extra_body: Mapping[str, object] | None,
        ) -> dict[str, object]:  # mutable-ok: wire request body is a plain dict
            if _service_tier_as_completion_window_enabled(provider):
                return _merge_extra_body_keeping_metadata(request, extra_body)
            return super().merge_extra_body(request, extra_body)

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

            drop_service_tier: Final = _service_tier_completion_window_drop(
                provider, non_default_params.get("service_tier"), model, drop_params
            )
            params_to_map: Final = (
                {  # mutable-ok: drop_params strips the tier into a fresh dict
                    key: value for key, value in non_default_params.items() if key != "service_tier"
                }
                if drop_service_tier
                else non_default_params
            )
            # Apply supported params
            for param, value in params_to_map.items():
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


def _json_responses_complete_url(provider: SimpleProviderConfig, api_base: str | None) -> str:
    resolved: Final = (
        api_base or (get_secret_str(provider.api_base_env) if provider.api_base_env else None) or provider.base_url
    )
    if resolved is None:
        raise ValueError(f"api_base is required for provider {provider.slug}")
    return f"{resolved.rstrip('/')}/responses"


def _json_responses_request_body(
    provider: SimpleProviderConfig,
    config: "OpenAILikeResponsesConfig",
    model: str,
    input: "str | ResponseInputParam",
    params: "dict[str, object]",  # mutable-ok: matches base signature
    litellm_params: "GenericLiteLLMParams",
    headers: "dict[str, object]",  # mutable-ok: matches base signature
) -> "dict[str, object]":  # mutable-ok: matches base signature
    from litellm.llms.openai_like.responses.transformation import OpenAILikeResponsesConfig

    if provider.special_handling.get("force_store_false"):
        params["store"] = False
    body: Final = OpenAILikeResponsesConfig.transform_responses_api_request(
        config,
        model=model,
        input=input,
        response_api_optional_request_params=params,
        litellm_params=litellm_params,
        headers=headers,
    )
    if _service_tier_as_completion_window_enabled(provider):
        return _apply_service_tier_as_completion_window(body)
    return body


def _json_responses_map_params(
    provider: SimpleProviderConfig,
    config: "OpenAILikeResponsesConfig",
    params: "ResponsesAPIOptionalRequestParams",
    model: str,
    drop_params: bool,
) -> dict:
    from litellm.llms.openai_like.responses.transformation import OpenAILikeResponsesConfig

    mapped: Final = OpenAILikeResponsesConfig.map_openai_params(
        config, response_api_optional_params=params, model=model, drop_params=drop_params
    )
    if _service_tier_completion_window_drop(provider, params.get("service_tier"), model, drop_params):
        return {  # mutable-ok: drop_params strips the tier into a fresh dict
            key: value for key, value in mapped.items() if key != "service_tier"
        }
    return mapped


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
    from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams
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
            return _json_responses_complete_url(provider, api_base)

        def transform_responses_api_request(
            self,
            model: str,
            input: str | ResponseInputParam,
            response_api_optional_request_params: dict[str, object],  # mutable-ok: matches base signature
            litellm_params: GenericLiteLLMParams,
            headers: dict[str, object],  # mutable-ok: matches base signature
        ) -> dict[str, object]:  # mutable-ok: matches base signature
            return _json_responses_request_body(
                provider,
                self,
                model,
                input,
                response_api_optional_request_params,
                litellm_params,
                headers,
            )

        def merge_extra_body(
            self,
            request: Mapping[str, object],
            extra_body: Mapping[str, object] | None,
        ) -> dict[str, object]:  # mutable-ok: wire request body is a plain dict
            if _service_tier_as_completion_window_enabled(provider):
                return _merge_extra_body_keeping_metadata(request, extra_body)
            return super().merge_extra_body(request, extra_body)

        def map_openai_params(
            self,
            response_api_optional_params: ResponsesAPIOptionalRequestParams,
            model: str,
            drop_params: bool,
        ) -> dict:
            return _json_responses_map_params(provider, self, response_api_optional_params, model, drop_params)

    _responses_config_cache[provider.slug] = JSONProviderResponsesConfig
    return JSONProviderResponsesConfig
