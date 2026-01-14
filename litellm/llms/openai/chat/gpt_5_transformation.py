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
    def is_model_gpt_5_1_codex_max_model(cls, model: str) -> bool:
        """Check if the model is the gpt-5.1-codex-max variant."""
        model_name = model.split("/")[-1]  # handle provider prefixes
        return model_name == "gpt-5.1-codex-max"
    
    @classmethod
    def is_model_gpt_5_1_model(cls, model: str) -> bool:
        """Check if the model is a gpt-5.1 or gpt-5.2 chat variant.
        
        gpt-5.1/5.2 support temperature when reasoning_effort="none",
        unlike base gpt-5 which only supports temperature=1. Excludes
        pro variants which keep stricter knobs.
        """
        model_name = model.split("/")[-1]
        is_gpt_5_1 = model_name.startswith("gpt-5.1")
        is_gpt_5_2 = model_name.startswith("gpt-5.2") and "pro" not in model_name
        return is_gpt_5_1 or is_gpt_5_2

    @classmethod
    def is_model_gpt_5_2_pro_model(cls, model: str) -> bool:
        """Check if the model is the gpt-5.2-pro snapshot/alias."""
        model_name = model.split("/")[-1]
        return model_name.startswith("gpt-5.2-pro")

    @classmethod
    def is_model_gpt_5_2_model(cls, model: str) -> bool:
        """Check if the model is a gpt-5.2 variant (including pro)."""
        model_name = model.split("/")[-1]
        return model_name.startswith("gpt-5.2")

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
        reasoning_effort = (
            non_default_params.get("reasoning_effort")
            or optional_params.get("reasoning_effort")
        )
        if reasoning_effort is not None and reasoning_effort == "xhigh":
            if not (
                self.is_model_gpt_5_1_codex_max_model(model)
                or self.is_model_gpt_5_2_model(model)
            ):
                if litellm.drop_params or drop_params:
                    non_default_params.pop("reasoning_effort", None)
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message=(
                            "reasoning_effort='xhigh' is only supported for gpt-5.1-codex-max and gpt-5.2 models."
                        ),
                        status_code=400,
                    )

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
