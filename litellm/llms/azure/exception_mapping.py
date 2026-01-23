from typing import Any, Dict, Optional, Tuple

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
        azure_error, inner_error = AzureOpenAIExceptionMapping._extract_azure_error(
            original_exception
        )

        # Prefer the provider message/type/code when present.
        provider_message = (
            azure_error.get("message")
            if isinstance(azure_error, dict)
            else None
        ) or message
        provider_type = (
            azure_error.get("type") if isinstance(azure_error, dict) else None
        )
        provider_code = (
            azure_error.get("code") if isinstance(azure_error, dict) else None
        )

        # Keep the OpenAI-style body fields populated so downstream (proxy + SDK)
        # can surface `type` / `code` correctly.
        openai_style_body: Dict[str, Any] = {
            "message": provider_message,
            "type": provider_type or "invalid_request_error",
            "code": provider_code or "content_policy_violation",
            "param": None,
        }

        raise ContentPolicyViolationError(
            message=provider_message,
            llm_provider="azure",
            model=model,
            litellm_debug_info=extra_information,
            response=getattr(original_exception, "response", None),
            provider_specific_fields={
                # Preserve legacy key for backward compatibility.
                "innererror": inner_error,
                # Prefer Azure's current naming.
                "inner_error": inner_error,
                # Include the full Azure error object for clients that want it.
                "azure_error": azure_error or None,
            },
            body=openai_style_body,
        )

    @staticmethod
    def _extract_azure_error(
        original_exception: Exception,
    ) -> Tuple[Dict[str, Any], Optional[dict]]:
        """Extract Azure OpenAI error payload and inner error details.

        Azure error formats can vary by endpoint/version. Common shapes:
        - {"innererror": {...}} (legacy)
        - {"error": {"code": "...", "message": "...", "type": "...", "inner_error": {...}}}
        - {"code": "...", "message": "...", "type": "..."} (already flattened)
        """
        body_dict = getattr(original_exception, "body", None) or {}
        if not isinstance(body_dict, dict):
            return {}, None

        # Some SDKs place the payload under "error".
        azure_error: Dict[str, Any]
        if isinstance(body_dict.get("error"), dict):
            azure_error = body_dict.get("error", {})  # type: ignore[assignment]
        else:
            azure_error = body_dict

        inner_error = (
            azure_error.get("inner_error")
            or azure_error.get("innererror")
            or body_dict.get("innererror")
            or body_dict.get("inner_error")
        )

        return azure_error, inner_error
