from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth


class BaseTranslation(ABC):
    @staticmethod
    def transform_user_api_key_dict_to_metadata(
        user_api_key_dict: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Transform user_api_key_dict to a metadata dict with prefixed keys.

        Converts keys like 'user_id' to 'user_api_key_user_id' to clearly indicate
        the source of the metadata.

        Args:
            user_api_key_dict: UserAPIKeyAuth object or dict with user information

        Returns:
            Dict with keys prefixed with 'user_api_key_'
        """
        if user_api_key_dict is None:
            return {}

        # Convert to dict if it's a Pydantic object
        user_dict = (
            user_api_key_dict.model_dump()
            if hasattr(user_api_key_dict, "model_dump")
            else user_api_key_dict
        )

        if not isinstance(user_dict, dict):
            return {}

        # Transform keys to be prefixed with 'user_api_key_'
        transformed = {}
        for key, value in user_dict.items():
            # Skip None values and internal fields
            if value is None or key.startswith("_"):
                continue

            # If key already has the prefix, use as-is, otherwise add prefix
            if key.startswith("user_api_key_"):
                transformed[key] = value
            else:
                transformed[f"user_api_key_{key}"] = value

        return transformed

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

    async def process_output_streaming_response(
        self,
        responses_so_far: List[Any],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional["UserAPIKeyAuth"] = None,
    ) -> Any:
        """
        Process output streaming response with guardrails.

        Optional to override in subclasses.
        """
        return responses_so_far
