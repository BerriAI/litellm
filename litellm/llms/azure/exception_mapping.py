from typing import Optional

from litellm.exceptions import ContentPolicyViolationError


class AzureOpenAIExceptionMapping:
    """
    Class for creating Azure OpenAI specific exceptions
    """
    @staticmethod
    def create_content_policy_violation_error(
        message: str,
        model: str,
        extra_information: str,
        original_exception: Exception,
    ) -> ContentPolicyViolationError:
        """
        Create a content policy violation error
        """    
        raise ContentPolicyViolationError(
            message=f"litellm.ContentPolicyViolationError: AzureException - {message}",
            llm_provider="azure",
            model=model,
            litellm_debug_info=extra_information,
            response=getattr(original_exception, "response", None),
            provider_specific_fields={
                "innererror": AzureOpenAIExceptionMapping._get_innererror_from_exception(original_exception)
            },
        )
    
    @staticmethod
    def _get_innererror_from_exception(original_exception: Exception) -> Optional[dict]:
        """
        Azure OpenAI returns the innererror in the body of the exception
        This method extracts the innererror from the exception
        """
        innererror = None
        body_dict = getattr(original_exception, "body", None) or {}
        if isinstance(body_dict, dict):
            innererror = body_dict.get("innererror")
        return innererror
            