"""Support for Azure OpenAI gpt-5 model family."""

from typing import List

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.openai.chat.gpt_5_transformation import (
    OpenAIGPT5Config,
    _get_effort_level,
)
from litellm.types.llms.openai import AllMessageValues

from .gpt_transformation import AzureOpenAIConfig


class AzureOpenAIGPT5Config(AzureOpenAIConfig, OpenAIGPT5Config):
    """Azure specific handling for gpt-5 models."""

    GPT5_SERIES_ROUTE = "gpt5_series/"

    @classmethod
    def _supports_reasoning_effort_level(cls, model: str, level: str) -> bool:
        """Override to handle gpt5_series/ prefix used for Azure routing.

        The parent class calls ``_supports_factory(model, custom_llm_provider=None)``
        which fails to resolve ``gpt5_series/gpt-5.1`` to the correct Azure model
        entry. Strip the prefix and prepend ``azure/`` so the lookup finds
        ``azure/gpt-5.1`` in model_prices_and_context_window.json.
        """
        if model.startswith(cls.GPT5_SERIES_ROUTE):
            model = "azure/" + model[len(cls.GPT5_SERIES_ROUTE) :]
        elif not model.startswith("azure/"):
            model = "azure/" + model
        return super()._supports_reasoning_effort_level(model, level)

    @classmethod
    def is_model_gpt_5_model(cls, model: str) -> bool:
        """Check if the Azure model string refers to a gpt-5 variant.

        Accepts both explicit gpt-5 model names and the ``gpt5_series/`` prefix
        used for manual routing.
        """
        # gpt-5-chat* is a chat model and shouldn't go through GPT-5 reasoning restrictions.
        return (
            "gpt-5" in model and "gpt-5-chat" not in model
        ) or "gpt5_series" in model

    def get_supported_openai_params(self, model: str) -> List[str]:
        """Get supported parameters for Azure OpenAI GPT-5 models.

        Azure OpenAI GPT-5.2/5.4 models support logprobs, unlike OpenAI's GPT-5.
        This overrides the parent class to add logprobs support back for gpt-5.2+.

        Reference:
        - Tested with Azure OpenAI GPT-5.2 (api-version: 2025-01-01-preview)
        - Azure returns logprobs successfully despite Microsoft's general
          documentation stating reasoning models don't support it.
        """
        params = OpenAIGPT5Config.get_supported_openai_params(self, model=model)

        # Azure supports tool_choice for GPT-5 deployments, but the base GPT-5 config
        # can drop it when the deployment name isn't in the OpenAI model registry.
        if "tool_choice" not in params:
            params.append("tool_choice")

        # Only gpt-5.2+ has been verified to support logprobs on Azure.
        # The base OpenAI class includes logprobs for gpt-5.1+, but Azure
        # hasn't verified support for gpt-5.1, so remove them unless gpt-5.2/5.4+.
        if self._supports_reasoning_effort_level(
            model, "none"
        ) and not self.is_model_gpt_5_2_model(model):
            params = [p for p in params if p not in ["logprobs", "top_logprobs"]]
        elif self.is_model_gpt_5_2_model(model):
            azure_supported_params = ["logprobs", "top_logprobs"]
            params.extend(azure_supported_params)

        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        api_version: str = "",
    ) -> dict:
        reasoning_effort_value = non_default_params.get(
            "reasoning_effort"
        ) or optional_params.get("reasoning_effort")
        effective_effort = _get_effort_level(reasoning_effort_value)

        # gpt-5.1/5.2/5.4 support reasoning_effort='none', but other gpt-5 models don't
        # See: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning
        supports_none = self._supports_reasoning_effort_level(model, "none")

        if effective_effort == "none" and not supports_none:
            if litellm.drop_params is True or (
                drop_params is not None and drop_params is True
            ):
                non_default_params = non_default_params.copy()
                optional_params = optional_params.copy()
                if (
                    _get_effort_level(non_default_params.get("reasoning_effort"))
                    == "none"
                ):
                    non_default_params.pop("reasoning_effort")
                if _get_effort_level(optional_params.get("reasoning_effort")) == "none":
                    optional_params.pop("reasoning_effort")
            else:
                raise UnsupportedParamsError(
                    status_code=400,
                    message=(
                        "Azure OpenAI does not support reasoning_effort='none' for this model. "
                        "Supported values are: 'low', 'medium', and 'high'. "
                        "To drop this parameter, set `litellm.drop_params=True` or for proxy:\n\n"
                        "`litellm_settings:\n drop_params: true`\n"
                        "Issue: https://github.com/BerriAI/litellm/issues/16704"
                    ),
                )

        result = OpenAIGPT5Config.map_openai_params(
            self,
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

        # Only drop reasoning_effort='none' for models that don't support it
        result_effort = _get_effort_level(result.get("reasoning_effort"))
        if result_effort == "none" and not supports_none:
            result.pop("reasoning_effort")

        # Azure Chat Completions: gpt-5.4+ does not support tools + reasoning together.
        # Drop reasoning_effort when both are present (OpenAI routes to Responses API; Azure does not).
        if self.is_model_gpt_5_4_plus_model(model):
            has_tools = bool(
                non_default_params.get("tools") or optional_params.get("tools")
            )
            if has_tools and result_effort not in (None, "none"):
                result.pop("reasoning_effort", None)

        return result

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        model = model.replace(self.GPT5_SERIES_ROUTE, "")
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
