"""
Pass-Through Endpoint Message Handler for Unified Guardrails

This module provides a handler for passthrough endpoint requests.
It uses the field targeting configuration from litellm_logging_obj
to extract specific fields for guardrail processing.
"""

from typing import TYPE_CHECKING, Any, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.proxy._types import PassThroughGuardrailSettings

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class PassThroughEndpointHandler(BaseTranslation):
    """
    Handler for processing passthrough endpoint requests with guardrails.

    Uses passthrough_guardrails_config from litellm_logging_obj
    to determine which fields to extract for guardrail processing.
    """

    def _get_guardrail_settings(
        self,
        litellm_logging_obj: Optional["LiteLLMLoggingObj"],
        guardrail_name: Optional[str],
    ) -> Optional[PassThroughGuardrailSettings]:
        """
        Get the guardrail settings for a specific guardrail from logging_obj.
        """
        from litellm.proxy.pass_through_endpoints.passthrough_guardrails import (
            PassthroughGuardrailHandler,
        )

        if litellm_logging_obj is None:
            return None

        passthrough_config = getattr(
            litellm_logging_obj, "passthrough_guardrails_config", None
        )
        if not passthrough_config or not guardrail_name:
            return None

        return PassthroughGuardrailHandler.get_settings(
            passthrough_config, guardrail_name
        )

    def _extract_text_for_guardrail(
        self,
        data: dict,
        field_expressions: Optional[List[str]],
    ) -> str:
        """
        Extract text from data for guardrail processing.

        If field_expressions provided, extracts only those fields.
        Otherwise, returns the full payload as JSON.
        """
        from litellm.proxy.pass_through_endpoints.jsonpath_extractor import (
            JsonPathExtractor,
        )

        if field_expressions:
            text = JsonPathExtractor.extract_fields(
                data=data,
                jsonpath_expressions=field_expressions,
            )
            verbose_proxy_logger.debug(
                "PassThroughEndpointHandler: Extracted targeted fields: %s",
                text[:200] if text else None,
            )
            return text

        # Use entire payload, excluding internal fields
        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        payload_to_check = {
            k: v
            for k, v in data.items()
            if not k.startswith("_") and k not in ("metadata", "litellm_logging_obj")
        }
        verbose_proxy_logger.debug(
            "PassThroughEndpointHandler: Using full payload for guardrail"
        )
        return safe_dumps(payload_to_check)

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Any:
        """
        Process input by applying guardrails to targeted fields or full payload.
        """
        guardrail_name = guardrail_to_apply.guardrail_name
        verbose_proxy_logger.debug(
            "PassThroughEndpointHandler: Processing input for guardrail=%s",
            guardrail_name,
        )

        # Get field targeting settings for this guardrail
        settings = self._get_guardrail_settings(litellm_logging_obj, guardrail_name)
        field_expressions = settings.request_fields if settings else None

        # Extract text to check
        text_to_check = self._extract_text_for_guardrail(data, field_expressions)

        if not text_to_check:
            verbose_proxy_logger.debug(
                "PassThroughEndpointHandler: No text to check, skipping guardrail"
            )
            return data

        # Apply guardrail (pass-through doesn't modify the text, just checks it)
        _guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
            inputs={"texts": [text_to_check]},
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )

        return data

    async def process_output_response(
        self,
        response: Any,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional["LiteLLMLoggingObj"] = None,
        user_api_key_dict: Optional[Any] = None,
    ) -> Any:
        """
        Process output response by applying guardrails to targeted fields.

        Args:
            response: The response to process
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails
        """
        if not isinstance(response, dict):
            verbose_proxy_logger.debug(
                "PassThroughEndpointHandler: Response is not a dict, skipping"
            )
            return response

        guardrail_name = guardrail_to_apply.guardrail_name
        verbose_proxy_logger.debug(
            "PassThroughEndpointHandler: Processing output for guardrail=%s",
            guardrail_name,
        )

        # Get field targeting settings for this guardrail
        settings = self._get_guardrail_settings(litellm_logging_obj, guardrail_name)
        field_expressions = settings.response_fields if settings else None

        # Extract text to check
        text_to_check = self._extract_text_for_guardrail(response, field_expressions)

        if not text_to_check:
            return response

        # Create a request_data dict with response info and user API key metadata
        request_data: dict = (
            {"response": response}
            if not isinstance(response, dict)
            else response.copy()
        )

        # Add user API key metadata with prefixed keys
        user_metadata = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
        if user_metadata:
            request_data["litellm_metadata"] = user_metadata

        # Apply guardrail (pass-through doesn't modify the text, just checks it)
        _guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
            inputs={"texts": [text_to_check]},
            request_data=request_data,
            input_type="response",
            logging_obj=litellm_logging_obj,
        )

        return response
