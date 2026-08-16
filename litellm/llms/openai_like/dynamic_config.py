"""
Dynamic configuration class generator for JSON-based providers.
"""

from collections.abc import Coroutine, Mapping
from typing import Any, Final, Literal, overload

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from .json_loader import SimpleProviderConfig


def _clamp_temperature(temperature: float, n: int, constraints: Mapping[str, float]) -> float:
    capped: Final = (
        min(temperature, constraints["temperature_max"]) if "temperature_max" in constraints else temperature
    )
    floored: Final = max(capped, constraints["temperature_min"]) if "temperature_min" in constraints else capped
    floor_for_multiple_choices: Final = constraints.get("temperature_min_with_n_gt_1")
    if n > 1 and floor_for_multiple_choices is not None:
        return max(floored, floor_for_multiple_choices)
    return floored


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
            mapped: Final = {
                **optional_params,
                **{
                    provider.param_mappings.get(param, param): value
                    for param, value in non_default_params.items()
                    if param in provider.param_mappings or param in supported_params
                },
            }

            constrained: Final = (
                mapped
                if "temperature" not in mapped
                else {
                    **mapped,
                    "temperature": _clamp_temperature(
                        temperature=mapped["temperature"],
                        n=mapped.get("n", 1),
                        constraints=provider.constraints,
                    ),
                }
            )

            # The OpenAI SDK omits `stream` entirely when it is false, which makes
            # stream-by-default providers answer a non-streaming call with SSE. Pin it
            # on the wire through extra_body, which the SDK merges into the request body.
            if not provider.special_handling.get("send_explicit_stream_false") or constrained.get("stream"):
                return constrained
            requested_extra_body: Final = constrained.get("extra_body")
            extra_body: Final[dict] = requested_extra_body if isinstance(requested_extra_body, dict) else {}
            return {**constrained, "extra_body": {"stream": False, **extra_body}}

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
            response_api_optional_request_params: dict,
            litellm_params: GenericLiteLLMParams,
            headers: dict,
        ) -> dict:
            if provider.special_handling.get("force_store_false"):
                response_api_optional_request_params["store"] = False
            if provider.special_handling.get("send_explicit_stream_false"):
                response_api_optional_request_params.setdefault("stream", False)
            return super().transform_responses_api_request(
                model=model,
                input=input,
                response_api_optional_request_params=response_api_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )

    _responses_config_cache[provider.slug] = JSONProviderResponsesConfig
    return JSONProviderResponsesConfig
