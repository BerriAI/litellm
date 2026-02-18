"""
Dynamic configuration class generator for JSON-based providers.
"""

from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, overload

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from .json_loader import SimpleProviderConfig


def create_config_class(provider: SimpleProviderConfig):
    """Generate config class dynamically from JSON configuration"""

    # Choose base class
    base_class: type = (
        OpenAIGPTConfig if provider.base_class == "openai_gpt" else OpenAILikeChatConfig
    )

    class JSONProviderConfig(base_class):  # type: ignore[valid-type,misc]
        @overload
        def _transform_messages(
            self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
        ) -> Coroutine[Any, Any, List[AllMessageValues]]:
            ...

        @overload
        def _transform_messages(
            self,
            messages: List[AllMessageValues],
            model: str,
            is_async: Literal[False] = False,
        ) -> List[AllMessageValues]:
            ...

        def _transform_messages(
            self, messages: List[AllMessageValues], model: str, is_async: bool = False
        ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
            """Transform messages based on special_handling config"""
            
            # Handle content list to string conversion if configured
            if provider.special_handling.get("convert_content_list_to_string"):
                messages = handle_messages_with_content_list_to_str_conversion(messages)
            
            if is_async:
                return super()._transform_messages(
                    messages=messages, model=model, is_async=True
                )
            else:
                return super()._transform_messages(
                    messages=messages, model=model, is_async=False
                )

        def _get_openai_compatible_provider_info(
            self, api_base: Optional[str], api_key: Optional[str]
        ) -> Tuple[Optional[str], Optional[str]]:
            """Get API base and key from JSON config"""

            # Resolve base URL
            resolved_base = api_base
            if not resolved_base and provider.api_base_env:
                resolved_base = get_secret_str(provider.api_base_env)
            if not resolved_base:
                resolved_base = provider.base_url

            # Resolve API key
            resolved_key = api_key or get_secret_str(provider.api_key_env)

            return resolved_base, resolved_key

        def get_complete_url(
            self,
            api_base: Optional[str],
            api_key: Optional[str],
            model: str,
            optional_params: dict,
            litellm_params: dict,
            stream: Optional[bool] = None,
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
            from litellm.utils import supports_function_calling

            supported_params = super().get_supported_openai_params(model=model)

            _supports_fc = supports_function_calling(
                model=model, custom_llm_provider=provider.slug
            )

            if not _supports_fc:
                tool_params = ["tools", "tool_choice", "function_call", "functions", "parallel_tool_calls"]
                for param in tool_params:
                    if param in supported_params:
                        supported_params.remove(param)
                verbose_logger.debug(
                    f"Model {model} on provider {provider.slug} does not support "
                    f"function calling — removed tool-related params from supported params."
                )

            return supported_params

        def map_openai_params(
            self,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: bool,
        ) -> dict:
            """Apply parameter mappings and constraints"""

            supported_params = self.get_supported_openai_params(model)
            
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
                constraints = provider.constraints

                # Clamp to max
                if "temperature_max" in constraints:
                    temp = min(temp, constraints["temperature_max"])

                # Clamp to min
                if "temperature_min" in constraints:
                    temp = max(temp, constraints["temperature_min"])

                # Special case: temperature_min_with_n_gt_1
                if "temperature_min_with_n_gt_1" in constraints:
                    n = optional_params.get("n", 1)
                    if n > 1 and temp < constraints["temperature_min_with_n_gt_1"]:
                        temp = constraints["temperature_min_with_n_gt_1"]

                optional_params["temperature"] = temp

            return optional_params

        @property
        def custom_llm_provider(self) -> Optional[str]:
            return provider.slug

    return JSONProviderConfig
