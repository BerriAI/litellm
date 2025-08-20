"""Support for OpenAI gpt-5 model family."""

from typing import Optional

import litellm

from .gpt_transformation import OpenAIGPTConfig


class OpenAIGPT5Config(OpenAIGPTConfig):
    """Configuration for gpt-5 models.

    Handles OpenAI API quirks for the gpt-5 series like:

    - Mapping ``max_tokens`` -> ``max_completion_tokens``.
    - Dropping unsupported ``temperature`` values when requested.
    """

    @classmethod
    def is_model_gpt_5_model(cls, model: str) -> bool:
        return "gpt-5" in model

    def get_supported_openai_params(self, model: str) -> list:
        from litellm.utils import supports_tool_choice

        base_gpt_series_params = super().get_supported_openai_params(model=model)
        gpt_5_only_params = ["reasoning_effort"]
        base_gpt_series_params.extend(gpt_5_only_params)
        if not supports_tool_choice(model=model):
            base_gpt_series_params.remove("tool_choice")
        return base_gpt_series_params

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
                if temperature_value == 1:
                    optional_params["temperature"] = temperature_value
                elif litellm.drop_params or drop_params:
                    pass
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message=(
                            "gpt-5 models don't support temperature={}. Only temperature=1 is supported. To drop unsupported params set `litellm.drop_params = True`"
                        ).format(temperature_value),
                        status_code=400,
                    )
        return super()._map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
