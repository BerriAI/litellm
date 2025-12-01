from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth


class BaseTranslation(ABC):
    @abstractmethod
    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Any:
        """
        Process input messages with guardrails.

        Note: user_api_key_dict metadata should be available in the data dict.
        """
        pass

    @abstractmethod
    async def process_output_response(
        self,
        response: Any,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
    ) -> Any:
        """
        Process output response with guardrails.

        Args:
            response: The response object from the LLM
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata (passed separately since response doesn't contain it)
        """
        pass
