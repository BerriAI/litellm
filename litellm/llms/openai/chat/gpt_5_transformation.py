"""Support for OpenAI gpt-5 model family."""

from typing import Optional

import litellm

from .gpt_transformation import OpenAIGPTConfig


class OpenAIGPT5Config(OpenAIGPTConfig):
    """Configuration for gpt-5 models including GPT-5-Codex variants.

    Handles OpenAI API quirks for the gpt-5 series like:

    - Mapping ``max_tokens`` -> ``max_completion_tokens``.
    - Dropping unsupported ``temperature`` values when requested.
    - Support for GPT-5-Codex models optimized for code generation.
    """

    @classmethod
    def is_model_gpt_5_model(cls, model: str) -> bool:
        return "gpt-5" in model

    @classmethod
    def is_model_gpt_5_codex_model(cls, model: str) -> bool:
        """Check if the model is specifically a GPT-5 Codex variant."""
        return "gpt-5-codex" in model
    
    @classmethod
    def is_model_gpt_5_1_model(cls, model: str) -> bool:
        """Check if the model is a gpt-5.1 variant.
        
        gpt-5.1 supports temperature when reasoning_effort="none",
        unlike gpt-5 which only supports temperature=1.
        """
        return "gpt-5.1" in model

    def get_supported_openai_params(self, model: str) -> list:
        from litellm.utils import supports_tool_choice

        base_gpt_series_params = super().get_supported_openai_params(model=model)
        gpt_5_only_params = ["reasoning_effort", "verbosity"]
        base_gpt_series_params.extend(gpt_5_only_params)
        if not supports_tool_choice(model=model):
            base_gpt_series_params.remove("tool_choice")

        non_supported_params = [
            "logprobs",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "top_logprobs",
            "stop",
        ]

        return [
            param
            for param in base_gpt_series_params
            if param not in non_supported_params
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        ################################################################
        # max_tokens is not supported for gpt-5 models on OpenAI API
        # Relevant issue: https://github.com/BerriAI/litellm/issues/13381
        ################################################################
        if "max_tokens" in non_default_params:
            optional_params["max_completion_tokens"] = non_default_params.pop(
                "max_tokens"
            )

        if "temperature" in non_default_params:
            temperature_value: Optional[float] = non_default_params.pop("temperature")
            if temperature_value is not None:
                is_gpt_5_1 = self.is_model_gpt_5_1_model(model)
                reasoning_effort = (
                    non_default_params.get("reasoning_effort") 
                    or optional_params.get("reasoning_effort")
                )
                
                # gpt-5.1 supports any temperature when reasoning_effort="none" (or not specified, as it defaults to "none")
                if is_gpt_5_1 and (reasoning_effort == "none" or reasoning_effort is None):
                    optional_params["temperature"] = temperature_value
                elif temperature_value == 1:
                    optional_params["temperature"] = temperature_value
                elif litellm.drop_params or drop_params:
                    pass
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message=(
                            "gpt-5 models (including gpt-5-codex) don't support temperature={}. "
                            "Only temperature=1 is supported. "
                            "For gpt-5.1, temperature is supported when reasoning_effort='none' (or not specified, as it defaults to 'none'). "
                            "To drop unsupported params set `litellm.drop_params = True`"
                        ).format(temperature_value),
                        status_code=400,
                    )
        return super()._map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
