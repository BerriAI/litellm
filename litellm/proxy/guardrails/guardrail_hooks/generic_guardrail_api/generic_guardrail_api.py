# +-------------------------------------------------------------+
#
#           Use Generic Guardrail API for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import os
from typing import Any, Dict, List, Literal, Optional, Tuple

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GenericGuardrailAPIMetadata,
    GenericGuardrailAPIRequest,
    GenericGuardrailAPIResponse,
)

GUARDRAIL_NAME = "generic_guardrail_api"


class GenericGuardrailAPI(CustomGuardrail):
    """
    Generic Guardrail API integration for LiteLLM.

    This integration allows you to use any guardrail API that follows the
    LiteLLM Basic Guardrail API spec without needing to write custom integration code.

    The API should accept a POST request with:
    {
        "text": str,
        "request_body": dict,
        "additional_provider_specific_params": dict
    }

    And return:
    {
        "action": "BLOCKED" | "NONE" | "GUARDRAIL_INTERVENED",
        "blocked_reason": str (optional, only if action is BLOCKED),
        "text": str (optional, modified text if action is GUARDRAIL_INTERVENED)
    }
    """

    def __init__(
        self,
        headers: Optional[Dict[str, Any]] = None,
        api_base: Optional[str] = None,
        additional_provider_specific_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.headers = headers or {}
        base_url = api_base or os.environ.get("GENERIC_GUARDRAIL_API_BASE")

        if not base_url:
            raise ValueError(
                "api_base is required for Generic Guardrail API. "
                "Set GENERIC_GUARDRAIL_API_BASE environment variable or pass it in litellm_params"
            )

        # Append the endpoint path if not already present
        if not base_url.endswith("/beta/litellm_basic_guardrail_api"):
            base_url = base_url.rstrip("/")
            self.api_base = f"{base_url}/beta/litellm_basic_guardrail_api"
        else:
            self.api_base = base_url

        self.additional_provider_specific_params = (
            additional_provider_specific_params or {}
        )

        # Set supported event hooks
        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
            ]

        super().__init__(**kwargs)

        verbose_proxy_logger.debug(
            "Generic Guardrail API initialized with api_base: %s", self.api_base
        )

    def _extract_user_api_key_metadata(
        self, request_data: dict
    ) -> GenericGuardrailAPIMetadata:
        """
        Extract user API key metadata from request_data.

        Args:
            request_data: Request data dictionary that may contain user_api_key_dict

        Returns:
            GenericGuardrailAPIMetadata with extracted user information
        """
        metadata = GenericGuardrailAPIMetadata()

        # Extract user_api_key_dict from request_data
        user_api_key_dict = request_data.get("user_api_key_dict")
        if user_api_key_dict is None:
            return metadata

        # Map UserAPIKeyAuth fields to GenericGuardrailAPIMetadata fields
        # Use getattr to safely access fields that might not exist
        if hasattr(user_api_key_dict, "token"):
            metadata["user_api_key_hash"] = getattr(user_api_key_dict, "token", None)
        elif hasattr(user_api_key_dict, "api_key"):
            metadata["user_api_key_hash"] = getattr(user_api_key_dict, "api_key", None)

        metadata["user_api_key_alias"] = getattr(user_api_key_dict, "key_alias", None)
        metadata["user_api_key_user_id"] = getattr(user_api_key_dict, "user_id", None)
        metadata["user_api_key_user_email"] = getattr(
            user_api_key_dict, "user_email", None
        )
        metadata["user_api_key_team_id"] = getattr(user_api_key_dict, "team_id", None)
        metadata["user_api_key_team_alias"] = getattr(
            user_api_key_dict, "team_alias", None
        )
        metadata["user_api_key_end_user_id"] = getattr(
            user_api_key_dict, "end_user_id", None
        )
        metadata["user_api_key_org_id"] = getattr(user_api_key_dict, "org_id", None)

        verbose_proxy_logger.debug(
            "Generic Guardrail API: Extracted user metadata: %s",
            {k: v for k, v in metadata.items() if v is not None},
        )

        return metadata

    async def apply_guardrail(
        self,
        texts: List[str],
        request_data: dict,
        input_type: Literal["request", "response"],
        images: Optional[List[str]] = None,
    ) -> Tuple[List[str], Optional[List[str]]]:
        """
        Apply the Generic Guardrail API to the given text.

        This is the main method that gets called by the framework.

        Args:
            texts: List of texts to check
            request_data: Request data dictionary containing user_api_key_dict and other metadata
            input_type: Whether this is a "request" or "response" guardrail
            images: Optional list of images to check

        Returns:
            Tuple of (processed texts, processed images)

        Raises:
            Exception: If the guardrail blocks the request
        """
        verbose_proxy_logger.debug("Generic Guardrail API: Applying guardrail to text")

        # Use provided request_data or create an empty dict
        if request_data is None:
            request_data = {}

        request_body = request_data.get("body") or {}

        # Merge additional provider specific params from config and dynamic params
        additional_params = {**self.additional_provider_specific_params}

        # Get dynamic params from request if available
        dynamic_params = self.get_guardrail_dynamic_request_body_params(request_body)
        if dynamic_params:
            additional_params.update(dynamic_params)

        # Extract user API key metadata
        user_metadata = self._extract_user_api_key_metadata(request_data)

        # Create request payload
        guardrail_request = GenericGuardrailAPIRequest(
            texts=texts,
            request_data=user_metadata,
            images=images,
            additional_provider_specific_params=additional_params,
            input_type=input_type,
        )

        # Prepare headers
        headers = {"Content-Type": "application/json"}
        if self.headers:
            headers.update(self.headers)

        try:
            # Make the API request
            response = await self.async_handler.post(
                url=self.api_base,
                json=guardrail_request.to_dict(),
                headers=headers,
            )

            response.raise_for_status()
            response_json = response.json()

            verbose_proxy_logger.debug(
                "Generic Guardrail API response: %s", response_json
            )

            guardrail_response = GenericGuardrailAPIResponse.from_dict(response_json)

            # Handle the response
            if guardrail_response.action == "BLOCKED":
                # Block the request
                error_message = (
                    guardrail_response.blocked_reason or "Content violates policy"
                )
                verbose_proxy_logger.warning(
                    "Generic Guardrail API blocked request: %s", error_message
                )
                raise Exception(f"Content blocked by guardrail: {error_message}")

            elif guardrail_response.action == "GUARDRAIL_INTERVENED":
                # Content was modified by the guardrail
                if guardrail_response.texts:
                    verbose_proxy_logger.debug("Generic Guardrail API modified text")
                    return guardrail_response.texts, guardrail_response.images

            # Action is NONE or no modifications needed
            return (
                guardrail_response.texts or texts,
                guardrail_response.images or images,
            )

        except Exception as e:
            # Check if it's already an exception we raised
            if "Content blocked by guardrail" in str(e):
                raise
            verbose_proxy_logger.error(
                "Generic Guardrail API: failed to make request: %s", str(e)
            )
            raise Exception(f"Generic Guardrail API failed: {str(e)}")
