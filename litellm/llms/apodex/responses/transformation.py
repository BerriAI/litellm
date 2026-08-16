"""
Apodex Responses API — OpenAI-compatible, with a model-aware parameter contract.

Apodex serves /v1/responses for both model families but they accept different
subsets, so the restrictions here are keyed off the model rather than applied
provider-wide:

- core models are a stateless subset: `store` is forced to false, and
  `previous_response_id` or `background` come back as HTTP 400
- the Deep Research tiers keep server-side state, so they take all three
- both default `stream` to true, so a non-streaming call has to say so

Ref: https://platform.apodex.ai/docs/responses-api
     https://platform.apodex.ai/docs/models
"""

from collections.abc import Mapping
from typing import Final

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import get_apodex_api_key, is_deep_research_model

# Rejected by the core models with HTTP 400: there is no server-side conversation
# to resume and requests are always executed inline.
_STATEFUL_PARAMS: Final = ("previous_response_id", "background")


class ApodexResponsesConfig(OpenAIResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.APODEX

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: matches the base-class signature
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:  # mutable-ok: matches the base-class signature
        """Resolve the Apodex key rather than inheriting OpenAI's OPENAI_API_KEY fallback,
        which would otherwise forward an unrelated OpenAI key to Apodex."""
        resolved_params: Final = litellm_params or GenericLiteLLMParams()
        api_key: Final = get_apodex_api_key(resolved_params.api_key)
        if api_key is None:
            return headers
        return {  # mutable-ok: matches the base-class signature
            **headers,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: matches the base-class signature
        inherited: Final = super().get_supported_openai_params(model)
        if is_deep_research_model(model):
            return inherited
        return [  # mutable-ok: matches the base-class signature
            param for param in inherited if param not in _STATEFUL_PARAMS
        ]

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: matches the base-class signature
        mapped: Final = super().map_openai_params(
            response_api_optional_params=response_api_optional_params,
            model=model,
            drop_params=drop_params,
        )

        stateless: Final = (
            mapped
            if is_deep_research_model(model)
            else self._enforce_stateless(mapped, model=model, drop_params=drop_params)
        )

        if stateless.get("stream"):
            return {**stateless}  # mutable-ok: JSON request body
        return {**stateless, "stream": False}  # mutable-ok: JSON request body

    @staticmethod
    def _enforce_stateless(params: Mapping[str, object], model: str, drop_params: bool) -> Mapping[str, object]:
        """Core models only: drop what the stateless subset rejects and pin store to false."""
        if params.get("store") is True and not (drop_params or litellm.drop_params):
            raise litellm.UnsupportedParamsError(
                message=(
                    f"apodex model {model} does not support store=True on /v1/responses: the endpoint is a "
                    "stateless subset. To drop this, set `litellm.drop_params = True`"
                ),
                status_code=400,
            )
        kept: Final = {  # mutable-ok: JSON request body
            key: value for key, value in params.items() if key not in _STATEFUL_PARAMS
        }
        return {**kept, "store": False}  # mutable-ok: JSON request body
