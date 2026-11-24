from collections.abc import Mapping
from typing import Final

import httpx

from litellm.exceptions import ContentPolicyViolationError
from litellm.litellm_core_utils.core_helpers import as_str_mapping


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
        response: httpx.Response | None,
    ) -> ContentPolicyViolationError:
        """
        Create a content policy violation error
        """
        azure_error, inner_error = AzureOpenAIExceptionMapping._extract_azure_error(original_exception)

        # Prefer the provider message/type/code when present.
        provider_message: Final = (azure_error.get("message") if isinstance(azure_error, dict) else None) or message
        provider_type: Final = azure_error.get("type") if isinstance(azure_error, dict) else None
        provider_code: Final = azure_error.get("code") if isinstance(azure_error, dict) else None

        # Keep the OpenAI-style body fields populated so downstream (proxy + SDK)
        # can surface `type` / `code` correctly.
        openai_style_body: Final[dict[str, object]] = {
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
            response=response,
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
    ) -> tuple[Mapping[str, object], object]:
        """Extract Azure OpenAI error payload and inner error details.

        Azure error formats can vary by endpoint/version. Common shapes:
        - {"innererror": {...}} (legacy)
        - {"error": {"code": "...", "message": "...", "type": "...", "inner_error": {...}}}
        - {"code": "...", "message": "...", "type": "..."} (already flattened)
        """
        body: Final[object] = getattr(original_exception, "body", None) or {}
        body_dict: Final = as_str_mapping(body)
        if body_dict is None or not isinstance(body, dict):
            return {}, None

        # Some SDKs place the payload under "error".
        nested_error: Final = as_str_mapping(body_dict.get("error"))
        azure_error: Final = (
            nested_error if nested_error is not None and isinstance(body_dict.get("error"), dict) else body_dict
        )

        inner_error: Final[object] = (
            azure_error.get("inner_error")
            or azure_error.get("innererror")
            or body_dict.get("innererror")
            or body_dict.get("inner_error")
        )

        return azure_error, inner_error
