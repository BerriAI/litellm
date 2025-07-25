
"""
Helper functions for health check calls.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

class HealthCheckHelpers:

    @staticmethod
    async def ahealth_check_wildcard_models(
        model: str,
        custom_llm_provider: str,
        model_params: dict,
        litellm_logging_obj: "Logging",
    ) -> dict:
        from litellm import acompletion
        from litellm.litellm_core_utils.llm_request_utils import (
            pick_cheapest_chat_models_from_llm_provider,
        )

        # this is a wildcard model, we need to pick a random model from the provider
        cheapest_models = pick_cheapest_chat_models_from_llm_provider(
            custom_llm_provider=custom_llm_provider, n=3
        )
        if len(cheapest_models) == 0:
            raise Exception(
                f"Unable to health check wildcard model for provider {custom_llm_provider}. Add a model on your config.yaml or contribute here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
            )
        if len(cheapest_models) > 1:
            fallback_models = cheapest_models[
                1:
            ]  # Pick the last 2 models from the shuffled list
        else:
            fallback_models = None
        model_params["model"] = cheapest_models[0]
        model_params["litellm_logging_obj"] = litellm_logging_obj
        model_params["fallbacks"] = fallback_models
        model_params["max_tokens"] = 1
        await acompletion(**model_params)
        return {}
    

    @staticmethod
    def _update_model_params_with_health_check_tracking_information(
        model_params: dict,
    ) -> dict:
        """
        Updates the health check model params with tracking information.

        The following is added at this stage:
            1. `tags`: This helps identify health check calls in the DB.
            2. `user_api_key_auth`: This helps identify health check calls in the DB.
                We need this since the DB requires an API Key to track a log in the SpendLogs Table
        """
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
        _metadata_variable_name = "litellm_metadata"
        litellm_metadata = HealthCheckHelpers._get_metadata_for_health_check_call()
        model_params[_metadata_variable_name] = litellm_metadata
        model_params = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
            data=model_params,
            user_api_key_dict=UserAPIKeyAuth.get_litellm_internal_health_check_user_api_key_auth(),
            _metadata_variable_name=_metadata_variable_name,
        )
        return model_params
    
    @staticmethod
    def _get_metadata_for_health_check_call():
        """
        Returns the metadata for the health check call.
        """
        from litellm.constants import LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
        return {
            "tags": [LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME],
        }