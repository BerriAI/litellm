from typing import Optional

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class ModerationAPIUtils:

    @staticmethod
    def init_litellm_logging_obj_for_moderations_call(
        custom_llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the litellm_logging_obj for a moderations call

        Ensures the correct `custom_llm_provider`, model, and user are set in the litellm_logging_obj

        This will be used downstream when constructing the standard_logging_payload
        """
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = kwargs.get(
            "litellm_logging_obj", None
        )
        if litellm_logging_obj:
            custom_llm_provider = (
                custom_llm_provider or litellm.LlmProviders.OPENAI.value
            )
            litellm_logging_obj.update_environment_variables(
                model=model,
                user=kwargs.get("user", None),
                optional_params={},
                litellm_params={
                    **kwargs,
                },
                custom_llm_provider=custom_llm_provider,
            )
